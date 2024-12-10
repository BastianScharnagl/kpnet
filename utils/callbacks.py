from pytorch_lightning.callbacks import Callback
import numpy as np
from utils.utils import norm_keypoints, translate_keypoints, draw_keypoints
import torch
import os
import torchvision
from torchvision.utils import save_image
from PIL import Image

class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=True,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx)): # batch_idx % self.batch_freq == 0

            img_np = ((np.array(batch["human_image"].cpu()+1)/2)*255).astype(np.uint8)
            size = (384, 512)
            #for key in self.input_keys:
            #    img_np = draw_keypoints(img_np, self.batch[key], size, (255,0,0))
    
            x = pl_module.preprocess_input(batch)
            batch["y_hat"] = pl_module(x)

            n_b = batch[pl_module.output_key].shape[0]
            n = n_b if n_b < 8 else 8
            
            
            for i in range(4):
                #img_np[i] = cv2.cvtColor(img_np[i], cv2.COLOR_RGB2BGR)
    
                #batch[pl_module.output_key][i] = translate_keypoints(batch[pl_module.output_key][i], (-batch["human_keypoints_posed"][i][2], -batch["human_keypoints_posed"][i][3]))
                batch[pl_module.output_key][i] = translate_keypoints(batch[pl_module.output_key][i], (-0.5, -0.5))
                batch[pl_module.output_key][i] = norm_keypoints(batch[pl_module.output_key][i], (1/768,1/1024))
                
                batch["y_hat"][i] = translate_keypoints(batch["y_hat"][i], (-0.5, -0.5))#(-batch["openpose_json"][i][2], -batch["openpose_json"][i][3]))            
                batch["y_hat"][i] = norm_keypoints(batch["y_hat"][i], (1/768,1/1024))
                
                #batch["human_keypoints_posed"][i] = translate_keypoints(batch["human_keypoints_posed"][i], (-0.5, -0.5))
                #batch["human_keypoints_posed"][i] = norm_keypoints(batch["human_keypoints_posed"][i], (1/768,1/1024))

            img_np = draw_keypoints(img_np[:4], batch["human_keypoints_posed"][i], (255,0,0))
    
            
            img_np = draw_keypoints(img_np[:4], batch[pl_module.output_key][:4], (0,255,0))
    
            img_np = draw_keypoints(img_np[:4], batch["y_hat"][:4], (0,0,255))
            
            img = (img_np[:4]).astype(np.uint8).transpose(0,3,1,2)
            tensor = torch.Tensor(img)

            grid = torchvision.utils.make_grid(tensor/255)
    
            filename = "gs-{:06}_e-{:06}_b-{:06}.png".format(
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx)
            
            path = os.path.join(pl_module.logger.save_dir, pl_module.logger.name, pl_module.logger.version, "images", filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            save_image(grid, path)
                
                #self.log_local(pl_module.logger.save_dir, split, tensor, pl_module.global_step, pl_module.current_epoch, batch_idx)


    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)

import sys
from argparse import ArgumentParser
from omegaconf import OmegaConf
import importlib
from pytorch_lightning import Trainer
import torch
from datetime import datetime
from utils.utils import instantiate_from_config, load_state_dict
import os



class ImageLoggerMLPAdapterVAE(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=True,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        config = "configs/stablediffusion/vae/v2_stock.yaml"
        checkpoint = "logs/stablediffusion/vae/v2_stock/02072024_134236/checkpoints/epoch=127-step=254720.ckpt"
        
        adaptee_config = OmegaConf.load(config)
        self.adaptee = instantiate_from_config(adaptee_config.model)
        print("load adaptee from ", checkpoint)
        self.adaptee.load_state_dict(torch.load(checkpoint, "cpu"), strict=False)
        self.adaptee.eval()
        self.adaptee.requires_grad_(False)
        self.adaptee.to("cuda:0")

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx)): # batch_idx % self.batch_freq == 0

            x = pl_module.preprocess_input(batch)
            #print("x", x.shape)
            y_hat = pl_module(x)
            #print("inf", y_hat.shape)

            y_hat = self.adaptee.decode(y_hat.to("cuda:0"))
            
            #y_hat = pl_module.decode_train(y_hat)
            #print("decode", y_hat.shape)
            grid = torchvision.utils.make_grid(y_hat[:4])
    
            filename = "output-gs-{:06}_e-{:06}_b-{:06}.png".format(
                pl_module.global_step,
                pl_module.current_epoch,
                batch["id"][0])
            
            path = os.path.join(pl_module.logger.save_dir, pl_module.logger.name, pl_module.logger.version, "images", filename)

            os.makedirs(os.path.split(path)[0], exist_ok=True)
            save_image(grid, path)
            """
            grid = torchvision.utils.make_grid(batch["fashion_image_stock"][:4])

            filename = "image-gs-{:06}_e-{:06}_b-{:06}.png".format(
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx)
            
            path = os.path.join(pl_module.logger.save_dir, pl_module.logger.name, pl_module.logger.version, "images", filename)

            os.makedirs(os.path.split(path)[0], exist_ok=True)
            save_image(grid, path)
            """
            #self.log_local(pl_module.logger.save_dir, split, tensor, pl_module.global_step, pl_module.current_epoch, batch_idx)


    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)

class ImageLoggerControlNet(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    def log_local(self, save_dir, name, version, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "image_log", split)
        for k in images:
            if images[k].shape[1] == 77: # cond
                continue
            
            try:
                grid = torchvision.utils.make_grid(images[k], nrow=4)
                if self.rescale:
                    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                #print(k, grid.shape)
                img = Image.fromarray((grid * 255).astype(np.uint8))
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
                path = os.path.join(save_dir, name, version, "images", filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                img.save(path)
            except:
                pass

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)
            print(images.keys())
            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)
                #if k == "latent_conditioning":
                #    images[k] = images[k].cuda()
                #    images[k] = pl_module.cond_stage_model.adaptee.decode(images[k])
            try:          
                for k in batch:
                    N = min(batch[k].shape[0], self.max_images)
                    images[k] = batch[k][:N]
                    if isinstance(images[k], torch.Tensor):
                        images[k] = images[k].detach().cpu().permute(0,3,1,2)
                        if self.clamp:
                            images[k] = torch.clamp(images[k], -1., 1.)
            except:
                pass
                
            self.log_local(pl_module.logger.save_dir, pl_module.logger.name, pl_module.logger.version, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")
