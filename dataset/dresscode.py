import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split, default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from pytorch_lightning import LightningDataModule
import torch
import PIL
from PIL import Image, ImageDraw, ImageFont, ImageOps

import pickle
import pandas as pd
import random
import torchvision
from utils.utils import resize_with_padding, norm_keypoints, translate_keypoints         

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
    
def read_image(data_root, folder_name, file_id, size):
    ext = ".jpg"
    if folder_name in ["cloth_agnostic", "cloth_agnostic_mask", "parse_agnostic"]:
        ext = ".png"
    elif folder_name == "parsing_maps":
        ext = "_gray.png"
    elif folder_name == "human_image":
        ext = "_0.jpg"    
    elif folder_name == "fashion_image_stock":
        ext = "_1.jpg"
    path = os.path.join(data_root, folder_name, file_id + ext)
    
    img = Image.open(path)

    if ".png" in path:
        img.convert('RGB')
    
    color = (255, 255, 255)
    #if folder_name in ["parsing_maps", "parse_mask", "parse_agnostic", "cloth_agnostic_mask", "cloth_agnostic_body"]:
    #    color = (0,0,0)

    img = resize_with_padding(img, size, color)
    
    img = np.array(img).astype(np.float32) / 255.0
    img = (img * 2.0) - 1.0

    return img#.transpose(2,0,1) # for stable diffusion -> get_input with einops

def read_keypoints(data_root, folder_name, file_id):
    ext = ".json"
    if folder_name == "human_keypoints_posed":
        ext = "_0_keypoints.json"
    elif folder_name == "fashion_keypoints_stock":
        ext = "_1.json"
    elif folder_name == "fashion_keypoints_posed":
        ext = "_0.json"
        
    with open(os.path.join(data_root, folder_name, file_id+ext), "r") as f:
        data = json.load(f)
        if folder_name == "human_keypoints_posed":
            keypoints = data["people"][0]["pose_keypoints_2d"]
            t_keypoints = torch.zeros([int(len(keypoints)/3*2)])
            # Logik in Dataset oder in Modell?
            for i in range(len(keypoints)-2):
                if i % 3 == 0:
                    x_k = keypoints[i]
                    y_k = keypoints[i+1]
                        
                    t_keypoints[i//3*2] = x_k
                    t_keypoints[(i//3*2)+1] = y_k
        elif folder_name in ["fashion_keypoints_posed", "fashion_keypoints_stock"]:
            #print(data)
            keypoints = data["keypoints"]
            #fashion_class = data["class"]
            t_keypoints = torch.zeros([28])
            
            #offset = None
            #if fashion_class == "0":
            #    offset = 0
            #elif fashion_class == "1":
            #    offset = 50

            for i in range(len(keypoints)):
                    x_k = keypoints[i][0]
                    y_k = keypoints[i][1]
                    t_keypoints[i*2] = x_k
                    t_keypoints[i*2+1] = y_k
                    
    return t_keypoints

def read_smpl_data(data_root, folder_name, file_id):
    ext = "_0.json"

    with open(os.path.join(data_root, folder_name, file_id+ext), "r") as f:
        data = json.load(f)

        t_j = torch.zeros([98])
        for n_j in range(49):
            j_x = data["smpl_joints2d"][0][n_j][0]
            j_y = data["smpl_joints2d"][0][n_j][1]

            t_j[n_j*2] = j_x
            t_j[n_j*2+1] = j_y
    
        shape = torch.Tensor(data["pred_shape"][0])

    return (t_j, shape)

class DataModule(LightningDataModule):
    def __init__(self,
                 data_root,
                 image,
                 representations,
                 batch_size,
                 num_workers,
                 size=[768,1024],
                 translate=False,
                 pkl_path="pairs.pkl",
                 reload=False
                ):
        super().__init__()

        self.data_root = data_root
        
        self.image = image
        self.representations = representations
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.size = size  
        self.translate = translate
        
        self.pkl_path = pkl_path
        self.reload = reload
        
    def setup(self, stage=None):
        # load dataset
        dataset = Dataset(
            data_root=self.data_root,
            image=self.image,
            representations=self.representations,
            size = self.size,
            translate = self.translate,
            pkl_path = self.pkl_path,
            reload = self.reload
        )


        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset=dataset, lengths=[0.7, 0.2, 0.1], generator=torch.Generator().manual_seed(42))

        print("Dataset Split", len(self.train_dataset), len(self.val_dataset), len(self.test_dataset))
        
        
        self.datasets = {
            "train": self.train_dataset,
            "validation": self.val_dataset,
            "test": self.test_dataset
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=collate_fn)

class Dataset(Dataset):
    def __init__(self,
                 data_root,
                 image,
                 representations,
                 size,
                 translate=False,
                 pkl_path="pairs.pkl",
                 reload=False
                 ):

        self.data_root = data_root
        self.image = image
        self.representations = representations
        
        self.size = size
        self.translate = translate
        self.prepare_data(pkl_path, reload)

        print("Dataset Size:", self.__len__)
        
    def prepare_data(self, pickle_path="pairs.pkl", reload=False):
        self.indices = {}

        for folder_name in self.representations:
            files = os.listdir(os.path.join(self.data_root, folder_name))
            self.indices[folder_name] = []
            for file_name in files:
                name = os.path.splitext(file_name)[0]
                id = name.split("_")[0]

                self.indices[folder_name].append(id)

        # only choose overlapping indices
        self.set_indices = set(self.indices[self.representations[0]])

        for i in range(1, len(self.representations)):
            self.set_indices = (self.set_indices & set(self.indices[self.representations[i]]))
        
        self.list_indices = list(self.set_indices)
        self.list_indices.sort()
        
    def __len__(self):
        return len(self.list_indices)
        
    def __getitem__(self, i):
        try:
            output = {}
            id = self.list_indices[i]
            output["id"] = id
            for key in self.representations:
    
                if key in ["human_keypoints_posed", "fashion_keypoints_posed", "fashion_keypoints_stock"]:
                    output[key] = read_keypoints(self.data_root, key, id)
    
                # for testing
                elif key == "txt":
                    output[key] = "txt"
    
                elif key == "smpl_data":
                    joints, shape = read_smpl_data(self.data_root, key, id)
                    joints = norm_keypoints(joints, (768, 1024))
                    if self.translate:
                        joints = translate_keypoints(joints, (0.5, 0.5))
                    output[key] = torch.cat([joints, shape],0)
                elif key == "fashion_image_stock_latents":
                    output[key] = torchvision.io.read_image(os.path.join(self.data_root, key, id +"_1.png"))/255
                else:
                    output[key] = read_image(self.data_root, key, id, self.size)
    
            for key in self.representations:
                if key in ["human_keypoints_posed", "fashion_keypoints_posed", "fashion_keypoints_stock"]:
                    output[key] = norm_keypoints(output[key], (768, 1024))
                    if self.translate:
                        output[key] = translate_keypoints(output[key], (0.5, 0.5))
            
            return output
        except:
            return None
            
        