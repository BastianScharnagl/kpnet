import importlib
import cv2
import os
import json
import torch
from PIL import Image, ImageOps
import numpy as np

def instantiate_from_config(config):
    module, cls = config["class_path"].rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)(**config.get("init_args", dict()))

def get_state_dict(d):
    return d.get('state_dict', d)

def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict

# draw keypoints in opencv numpy format (B,H,W,C)
def draw_keypoints(image, keypoints, color):
    for b in range(len(keypoints)):
        for i in range(len(keypoints[b])):
            if i%2 != 0:
                continue

            x = int(keypoints[b][i])
            y = int(keypoints[b][i+1])

            cv2.circle(image[b], (x, y), 3, color, -1)

    #return image

def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size[0] - img.size[0]
    delta_height = desired_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding, color=(255, 255, 255))

def resize_with_padding(img, expected_size, color):
    if len(expected_size) == 1:
        expected_size = (expected_size, expected_size)
        
    img.thumbnail((expected_size[0], expected_size[1]))

    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding, fill=color)

def read_image(data_root, fashion_name, folder_name, file_id, size):
    ext = ".jpg"
    if folder_name in ["cloth_agnostic", "cloth_agnostic_mask", "parse_agnostic"]:
        ext = ".png"
    elif folder_name == "parsing_maps":
        ext = "_gray.png"
        
    path = os.path.join(data_root, fashion_name, folder_name, file_id + ext)
    
    img = Image.open(path)

    if ".png" in path:
        img.convert('RGB')
    
    color = (255, 255, 255)
    if folder_name in ["parsing_maps", "parse_mask", "parse_agnostic", "cloth_agnostic_mask", "cloth_agnostic_body"]:
        color = (0,0,0)

    img = resize_with_padding(img, size, color)
    
    img = np.array(img).astype(np.float32) / 255.0
    img = (img * 2.0) - 1.0

    return img

def read_keypoints(data_root, fashion_name, folder_name, file_id):
    ext = ".json"
    if folder_name == "openpose_json":
        ext = "_keypoints.json"

    with open(os.path.join(data_root, fashion_name, folder_name, file_id+ext), "r") as f:
        data = json.load(f)
        if folder_name == "openpose_json":
            keypoints = data["people"][0]["pose_keypoints_2d"]
            t_keypoints = torch.zeros([int(len(keypoints)/3*2)])
            # Logik in Dataset oder in Modell?
            for i in range(len(keypoints)-2):
                if i % 3 == 0:
                    x_k = keypoints[i]
                    y_k = keypoints[i+1]
                        
                    t_keypoints[i//3*2] = x_k
                    t_keypoints[(i//3*2)+1] = y_k
        elif folder_name in ["fashion_keypoints_original", "fashion_keypoints_stock"]:
            keypoints = data["keypoints"]
            fashion_class = data["class"]
            t_keypoints = torch.zeros([116])
            
            offset = None
            if fashion_class == "0":
                offset = 0
            elif fashion_class == "1":
                offset = 50

            for i in range(len(keypoints)-2):
                if i % 3 == 0:
                    x_k = keypoints[i]
                    y_k = keypoints[i+1]
                    t_keypoints[(i//3*2)+offset] = x_k
                    t_keypoints[(i//3*2)+1+offset] = y_k
                    
    return t_keypoints

def norm_keypoints(keypoints, scale):
    if type(keypoints) == torch.Tensor:
        size = keypoints.shape[0]
    else:
        size = len(keypoints)
        
    for i in range(len(keypoints)):
        if i%2 != 0:
            continue
        keypoints[i] = keypoints[i]/scale[0]
        keypoints[i+1] = keypoints[i+1]/scale[1]
    return keypoints

def translate_keypoints(keypoints, p0):
    if type(keypoints) == torch.Tensor:
        size = keypoints.shape[0]
    else:
        size = len(keypoints)
        
    for i in range(len(keypoints)):
        if i%2 != 0:
            continue
        keypoints[i] = keypoints[i]-p0[0]
        keypoints[i+1] = keypoints[i+1]-p0[1]
    return keypoints