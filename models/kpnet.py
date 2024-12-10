from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torchvision
import importlib
import numpy as np
import cv2

from utils.utils import draw_keypoints, norm_keypoints, translate_keypoints

class KPNet(LightningModule):
    def __init__(self, encoder_layers=[], decoder_layers=[], input_keys=[], output_key=None, loss_fn:torch.nn.Module=torch.nn.MSELoss, learning_rate=1e-5):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.input_keys = input_keys
        self.output_key = output_key
        
        # ? Loss relativ zu den Körperpositionen z.B. Ärmel zu Handgelenk? 
        module, cls = loss_fn["class_path"].rsplit(".", 1)
        loss_cls = getattr(importlib.import_module(module, package=None), cls)
        args = loss_fn.get("init_args", dict())
        self.loss = loss_cls(**args)
        
        encoder_modules = []
        for i in range(len(encoder_layers)):
            module, cls = encoder_layers[i]["class_path"].rsplit(".", 1)
            layer_cls = getattr(importlib.import_module(module, package=None), cls)
            args = encoder_layers[i].get("init_args", dict())
            layer_obj = layer_cls(**args)
            encoder_modules.append(layer_obj)
            
        self.encoder = nn.Sequential(*encoder_modules)

        decoder_modules = []
        for i in range(len(decoder_layers)):
            module, cls = decoder_layers[i]["class_path"].rsplit(".", 1)
            layer_cls = getattr(importlib.import_module(module, package=None), cls)
            args = decoder_layers[i].get("init_args", dict())
            layer_obj = layer_cls(**args)
            decoder_modules.append(layer_obj)
            
        self.decoder = nn.Sequential(*decoder_modules)
    
    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        y =self.decoder(z)
        return y
        
    def forward(self, x):
        z = self.encode(x)
        y = self.decode(z)        
        return y

    def preprocess_input(self, batch):
        inputs = []
        for key in self.input_keys:
            inputs.append(batch[key])
        input = torch.cat(inputs, 1)
        return input
        
    def preprocess_output(self, batch):
        output = batch[self.output_key]
        return output
        
    def training_step(self, batch, batch_nb):
        x = self.preprocess_input(batch)
        y = self.preprocess_output(batch)

        y_hat = self(x)

        loss = self.loss(y_hat, y)
        
        self.log("train_loss", loss, prog_bar=True)
        self.batch = batch
        
        return loss
    
    def validation_step(self, batch, batch_nb):
        x = self.preprocess_input(batch)
        y = self.preprocess_output(batch)

        y_hat = self(x)

        loss = self.loss(y_hat, y)
        
        self.log("val_loss", loss, prog_bar=True)

        batch["y_hat"] = y_hat
        self.batch = batch
        
        return loss      
   
    def test_step(self, batch, batch_nb):
        x = self.preprocess_input(batch)
        y = self.preprocess_output(batch)

        y_hat = self(x)

        loss = self.loss(y_hat, y)
        
        self.log("test_loss", loss, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)