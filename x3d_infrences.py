import json
import urllib
import torch
from pytorchvideo.data.encoded_video import EncodedVideo



import torch
model_name = 'x3d_s'
modelx3d_s = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
import torch.nn as nn
import json
import urllib
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

# Choose the `x3d_s` model


device = "cuda"
model1 = modelx3d_s.eval()
model1 = modelx3d_s.to(device)

mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
frames_per_second = 30
model_transform_params  = {
"x3d_xs": {
    "side_size": 182,
    "crop_size": 182,
    "num_frames": 4,
    "sampling_rate": 12,
},
"x3d_s": {
    "side_size": 182,
    "crop_size": 182,
    "num_frames": 13,
    "sampling_rate": 6,
},
"x3d_m": {
    "side_size": 256,
    "crop_size": 256,
    "num_frames": 16,
    "sampling_rate": 5,
}
}

# Get transform parameters based on model
transform_params = model_transform_params[model_name]

# Note that this transform is specific to the slow_R50 model.
transform =  ApplyTransformToKey(
key="video",
transform=Compose(
    [
        UniformTemporalSubsample(transform_params["num_frames"]),
        Lambda(lambda x: x/255.0),
        NormalizeVideo(mean, std),
        ShortSideScale(size=transform_params["side_size"]),
        CenterCropVideo(
            crop_size=(transform_params["crop_size"], transform_params["crop_size"])
        )
    ]
),
)


#    Load the weights to the model
layers = list(model1.blocks.children())
_layers = layers[:-1]  # Extrait de caractéristiques
classifier = layers[-1]  # Classificateur
num_classes = 2
classifier.proj = nn.Linear(in_features=classifier.proj.in_features, out_features=2, bias=True)
model_path = "/kaggle/input/x3dpoid/poid_amine_5.pth"

# Load the model
checkpoint_state_dict= torch.load(model_path,map_location=torch.device('cpu'))
model1.load_state_dict(checkpoint_state_dict)

kinetics_id_to_classname={1:'SL',0:'NR'}
model1.to(device)


# The duration of the input clip is also specific to the model.
clip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"])/frames_per_second

def prediction(model= model1, liste_frame):
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
