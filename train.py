from argparse import ArgumentParser
from omegaconf import OmegaConf
from utils.utils import instantiate_from_config
import datetime
import os
from pytorch_lightning import Trainer
import torch

try:
    torch.multiprocessing.set_start_method('spawn')
    print("spawned")
except RuntimeError:
    pass

torch.set_float32_matmul_precision("highest")
torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    
    parser = ArgumentParser()
    
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--test", action='store_true')
    
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config)
    
    data = instantiate_from_config(config.data)
    
    model = instantiate_from_config(OmegaConf.to_object(config.model))
    
    if args.checkpoint != None:
        sd = torch.load(args.checkpoint, map_location="cpu")
        if "state_dict" in sd.keys():
            model.load_state_dict(sd["state_dict"], strict=False)
        else:
            model.load_state_dict(sd, strict=False)

    config_path = args.config.replace("configs/","").replace(".yaml", "")
    
    save_dir = "logs"
    if args.test:
        save_dir = save_dir+"_test"
    
    logger_conf = OmegaConf.create(
        {
            "class_path": "pytorch_lightning.loggers.CSVLogger",
            "init_args": 
            {
                "save_dir": save_dir,
                "name": config_path, 
                "version": now
            }
        }
    )

    
    
    trainer_kwargs = dict()
    
    trainer_kwargs["logger"] = instantiate_from_config(logger_conf)    
    config.trainer.pop("logger")
    
    trainer_kwargs["callbacks"] = [instantiate_from_config(callback) for callback in config.trainer.callbacks]
    config.trainer.pop("callbacks")

    trainer = Trainer(**config.trainer, **trainer_kwargs)

    trainer.fit(model, data)
    
    trainer.test(model, data)
    
    OmegaConf.save(config=config, f=os.path.join(save_dir, config_path, now, "config.yaml"))