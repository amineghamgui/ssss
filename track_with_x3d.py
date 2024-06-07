import time
import torch
import cv2
import colorsys
import numpy as np
import psutil
import imageio


from x3d_infrences import prediction
from ocsort import ocsort
from ultralytics import YOLO
from super_gradients.training import models
from super_gradients.training.models.detection_models.customizable_detector import CustomizableDetector
from super_gradients.training.pipelines.pipelines import DetectionPipeline
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'***************************************************************************************\n device : {device}')

def get_prediction(image_in, pipeline):
    # Preprocess
    preprocessed_image, processing_metadata = pipeline.image_processor.preprocess_image(image=image_in.copy())

    # Predict
    with torch.no_grad():
        torch_input = torch.Tensor(preprocessed_image).unsqueeze(0).to('cuda')
        model_output = pipeline.model(torch_input)
        prediction = pipeline._decode_model_output(model_output, model_input=torch_input)

    # Postprocess
    return pipeline.image_processor.postprocess_predictions(predictions=prediction[0], metadata=processing_metadata)


def img_detection_yolov8(model, image):

    results = model.predict(image, conf=0.6, classes=0, device=device)



    boxes = []
    labels = []
    probs = []

    for res in results:
        boxes = res.boxes.xyxy.cpu().numpy()
        labels = res.boxes.cls.cpu().numpy()
        probs = res.boxes.conf.cpu().numpy()

    # Combine boxes and confidence values into a single array
    xyxyc = np.hstack((boxes, np.c_[probs]))

    return xyxyc, labels




def img_detection_yolonas(model, image):
    # Measure detection task execution time
    start_time_detection = time.time()
    model_predictions = model.predict(image , fuse_model=False , conf=0.45 )

    end_time_detection = time.time()
    detection_time = end_time_detection - start_time_detection

    bboxes_xyxy = model_predictions[0].prediction.bboxes_xyxy.tolist()
    labels = model_predictions[0].prediction.labels.tolist()
    confidence = model_predictions[0].prediction.confidence.tolist()

    person_bboxes = []
    person_labels = []
    person_confidence = []

    for bbox, label, conf in zip(bboxes_xyxy, labels, confidence):
        if label != 0:
            continue

        person_bboxes.append(bbox)
        person_labels.append(int(label))
        person_confidence.append(float(conf))

    print('detection model :', labels, '\n person detection ', person_labels)

    return person_bboxes, person_labels, person_confidence ,detection_time



def get_color(number):
    hue = number * 30%180
    saturation = number * 103 % 256
    value = number * 50 % 256

    color = colorsys.hsv_to_rgb(hue / 179, saturation / 255, value / 255)

    return [int(c*255) for c in color]

def loading_models():

    tracker = ocsort.OCSort(det_thresh=0.30, max_age=30, min_hits=3)
    #yolo_nas = models.get("yolo_nas_m", pretrained_weights="coco").to(device)
    yolo_nas = models.get("yolo_nas_m", pretrained_weights="coco").cuda()
    yolo_nas.eval()



    yolo8 = YOLO('yolov8m.pt')



    print('detector : YOLO8  , tracker : Oc_sort')
    return tracker,yolo_nas


frames_a_concatener=[]
d=dict()
d_action=dict()
def tracking1(video_path):
    tracker,model = loading_models()
    deb=0
    target_size=""
    # make sure to set IOU and confidence in the pipeline constructor
    pipeline = DetectionPipeline(
        model=model,
        image_processor=model._image_processor,
        post_prediction_callback=model.get_post_prediction_callback(iou=0.25, conf=0.60),
        class_names=model._class_names,
        fuse_model=False,
        device=device,


    )


    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video stream or file")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_vd = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print (f'fps vd :{fps_vd}')
    print(f'frame_count vd :{frame_count}')
    print(f'w {frame_width} - h {frame_height}')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')


    out = cv2.VideoWriter('./oc_sort_NAS2.mp4', fourcc, fps_vd, (frame_width, frame_height))

    frames = []
    i = 0

    counter, fps, elasped = 0,0,0
    list_detection_time = []
    list_tracking_time = []

    class_name = ['person']
    iteration_time = []
    list_fps=[]
    
    diffirence_time_x3d=0

    
    while True:
        h,w = 0,0;

        start_time = time.time()
        ret, frame = cap.read()

          
        if ret:


            og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = og_frame.copy()
            num_personnes_total=0

            print('________________________________________')

            # Measure detection task execution time
            start_time_detection = time.time()


            # YOLO NAS
            pred = get_prediction(frame, pipeline)
            # print(f'pred ,{pred}')

            end_time_detection = time.time()
            detection_time = end_time_detection - start_time_detection
            list_detection_time.append(detection_time)
            print(f"Detection task execution time: {detection_time} seconds")

            labels = pred.labels
            xyxyc = np.hstack((pred.bboxes_xyxy,
                               np.c_[pred.confidence]))

            person_labels = []
            person_xyxyc = []

            for label, bbox_confidence in zip(labels, xyxyc):
                if label == 0:
                    person_labels.append(label)
                    person_xyxyc.append(bbox_confidence)

            person_labels = np.array(person_labels)
            person_xyxyc = np.array(person_xyxyc)
            print(labels)
            print(f'detection : {len(xyxyc)} -- person_detection : {len(person_xyxyc)}')

            
            if (len(person_xyxyc)!=0):

              tracks = tracker.update(person_xyxyc , person_labels )
              start_time_tracking = time.time()
              # update tracker
              tracks = tracker.update(person_xyxyc , person_labels )

              
              # draw tracks on frame
              for track in tracker.trackers:
                  
                  track_id = track.id
                  hits = track.hits
                  score = "{:.2f}".format(track.conf)
                  x1, y1, x2, y2 = np.round(track.get_state()).astype(int).squeeze()
                  
                  #Person detection zone
                  person_roi= frame[y1:y2, x1:x2]

                  
                  #Each person is defined with their own identifier in the dictionary d: d{id : [list of frames]}
                  #Analysis performed every 25 frames
                  if  person_roi.size:
                      if (track_id not in d):
                            d[track_id] = [] 
                            d[track_id].append(person_roi)
                            
                      else :
                            if(len(d[track_id])<25):
                                target_size = (d[track_id][0].shape[1], d[track_id][0].shape[0])  # Changer l'ordre de la taille cible
                                resized_roi = cv2.resize(person_roi, target_size)
                                d[track_id].append(resized_roi)
                            else:
                                    start_time_x3d=time.time()
                                   
                                    totalpred_withfps=prediction(model1,d[track_id])
                                    d_action[track_id]=totalpred_withfps[0]
                                    end_time_x3d=time.time()
                                    diffirence_time_x3d=end_time_x3d-start_time_x3d

                                    
                                    del d[track_id][0:24]
                # Define the colors
                  red_color = (255, 0, 0)  # Red color in BGR
                  green_color = (0, 255, 0)  # Green color in BGR

                # Blink control variables
                  blink_on = True
                  blink_duration = 0.5  # Duration to keep the rectangle on and off (in seconds)
                  last_blink_time = time.time()

                # Danger icon properties
                  icon_position = (10, 30)  # Position of the icon (top left corner)
                  icon_text = "Attention : présence de mouvements suspects dans la scène !!!!!!"  # Icon text, you can use any symbol you prefer

                # Inside your frame processing loop
                  if track_id in d_action:
                    print("amine")
                    current_time = time.time()
                    if current_time - last_blink_time >= blink_duration:
                        blink_on = not blink_on
                        last_blink_time = current_time

                    if d_action[track_id] == "SL":
                        color = red_color
                        if blink_on:  # Draw the rectangle and icon if blink_on is True
                            cv2.rectangle(og_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(og_frame,
                                        f"mouvement suspect",
                                        (int(x1) + 10, int(y1) - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        color,
                                        2,
                                        cv2.LINE_AA)

                    elif d_action[track_id] == "NR":
                        color = green_color
                        cv2.rectangle(og_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(og_frame,
                                    f"mouvement normal",
                                    (int(x1) + 10, int(y1) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color,
                                    2,
                                    cv2.LINE_AA)
                      

                  # Count
                  num_personnes_total = num_personnes_total + 1

              end_time_tracking = time.time()
              tracking_time = end_time_tracking - start_time_tracking
              list_tracking_time.append(tracking_time)
              print(f"Tracking task execution time: {tracking_time} seconds")



              current_time = time.time()
              elasped = (current_time - start_time)
              print("Temps écoulé:", elasped, "secondes")
              fps = 1 / elasped
              list_fps.append(fps)

              iteration_time.append(elasped)


            # Taille de la police et épaisseur du texte
              font_scale = 1
              thickness = 2

            # Couleur du texte et arrière-plan
              text_color = (0, 0, 0)
              background_color = (255, 255, 255)  # Couleur de l'arrière-plan (blanc ici)

            # Coordonnées pour afficher le FPS
              fps_position = (30, 60)

            # Coordonnées pour afficher le nombre de personnes
              num_personnes_position = (800, 650)

            # Mettez un rectangle blanc en arrière-plan pour le texte FPS
              (text_width, text_height) = cv2.getTextSize(
                f'FPS : {str(round(fps,2))}', cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
              cv2.rectangle(og_frame, (fps_position[0], fps_position[1] - text_height - 10),
                          (fps_position[0] + text_width + 10, fps_position[1] + 10), background_color, -1)
              cv2.putText(og_frame, f'FPS : {str(round(fps,2))}', fps_position, cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, text_color, thickness, cv2.LINE_AA)

            # Mettez un rectangle blanc en arrière-plan pour le texte Nombre de personnes
              (text_width, text_height) = cv2.getTextSize(
                f'Nombre de personnes : {num_personnes_total}', cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
              cv2.rectangle(og_frame, (num_personnes_position[0], num_personnes_position[1] - text_height - 10),
                          (num_personnes_position[0] + text_width + 10, num_personnes_position[1] + 10), background_color, -1)
              cv2.putText(og_frame, f'Nombre de personnes : {num_personnes_total}', num_personnes_position, cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, text_color, thickness, cv2.LINE_AA)


              frames.append(og_frame)


              out.write(cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR))

        else:
            break
    frame_width = d[0][0].shape[1]
    frame_height = d[0][0].shape[0]
    
    # frame_width1 = d[1][0].shape[1]
    # frame_height1 = d[1][0].shape[0]
    print("frame_width",frame_width)
    print("frame_height",)
    
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    #for i in d.keys():
    print("shape_______________________________",(frame_width,frame_height ))
    # print("shape_______________________________",(frame_width1,frame_height1))



    i=0
    
    #Mean time track / detection
    mean_detection_time = sum(list_detection_time) / len(list_detection_time)
    mean_tracking_time = sum(list_tracking_time) / len(list_tracking_time)
    print('------------------------------------------------------------------------')
    print(f'mean_detection_time {mean_detection_time} sec \n')
    print(f'mean_tracking_time {mean_tracking_time} sec \n ')

    #Mean fps
    average_iteration_time=(sum(iteration_time) - iteration_time[0]) / (len(iteration_time) - 1)
    mean_fps =1 / average_iteration_time
    mean_fps2=(sum(list_fps) - list_fps[0]) / (len(list_fps) - 1)

    print(f'mean_fps :{mean_fps}')
    print(f'mean_fps :{mean_fps2}')

    cap.release()
    out.release()
