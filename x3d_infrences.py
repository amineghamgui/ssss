import json
import urllib
import torch
from pytorchvideo.data.encoded_video import EncodedVideo

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
        CenterCropVideo,
        NormalizeVideo,
    )
from pytorchvideo.transforms import (
        ApplyTransformToKey,
        ShortSideScale,
        UniformTemporalSubsample
    )
    

def prediction(model, liste_frame):
    tensor_liste = [torch.from_numpy(arr) for arr in liste_frame]
    torch_tensor = torch.stack(tensor_liste)
    inputs=torch_tensor.permute(3, 0, 1, 2)
    i=dict()
    i["video"]=inputs
    video_data = transform(i)
    
    inputs = video_data["video"].to(device)

    start_time = time.time()
    # Make predictions
    preds = model(inputs[None, ...])
    
    # Apply softmax and get predicted classes
    post_act = torch.nn.Softmax(dim=1)
    
    preds = post_act(preds)
    
    pred_classes = preds.topk(k=1).indices[0]
    end_time = time.time()
    execution_time = end_time - start_time
    fps=10/execution_time
    print(preds.topk(k=1))
    # Map predicted classes to label names
    pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]
    return pred_class_names[0],fps
