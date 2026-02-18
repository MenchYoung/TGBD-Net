import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
import numpy as np
import json
import random
from accelerate import Accelerator, DistributedDataParallelKwargs

from dataloader import get_bbox_dataloaders
from config import Config 
from model import TGBDModel, LamedConfig
from trainer import EGFRTrainer

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.benchmark = True 

def main():
    # --- 1. initialize Accelerate ---
    # find_unused_parameters=True 
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
    cfg = Config()
    

    if accelerator.is_main_process:
        if not hasattr(cfg, 'ALPHA_ALIGN'):
            cfg.ALPHA_ALIGN = 1.0 
        cfg.SAVE_DIR = os.path.join(cfg.SAVE_DIR, fr"Align_{cfg.ALPHA_ALIGN}_lr_{cfg.LR}_bs{cfg.BATCH_SIZE}_epoch{cfg.EPOCHS}")
        os.makedirs(cfg.SAVE_DIR, exist_ok=True)
        print(f"🚀 Start Training (Accelerate DDP)")
        print(f"📂 Save Dir: {cfg.SAVE_DIR}")
        print(f"✅ Accelerator is using {accelerator.num_processes} GPUs")

    fix_seed(cfg.SEED + accelerator.process_index) # different seed per rank for independent randomness

    # --- 2. Prepare Data ---
    if accelerator.is_main_process:
        print("\n--- [1/4] Loading Data... ---")
    
    # Get DataLoader
    dataloaders = get_bbox_dataloaders(
        csv_path=cfg.CSV_PATH,
        image_size=cfg.IMAGE_SIZE,
        batch_size=cfg.BATCH_SIZE, 
        num_workers=cfg.NUM_WORKERS 
    )
    
    # --- 3. Prepare Model ---
    if accelerator.is_main_process:
        print("\n--- [2/4] Loading Model... ---")
        
    with open(cfg.CONFIG_JSON_PATH, 'r') as f:
        swin_config_dict = json.load(f)
    swin_config = LamedConfig(**swin_config_dict)
    
    model = TGBDModel(
        vision_tower_config=swin_config,
        weights_path=cfg.WEIGHTS_PATH,
        num_classes=cfg.NUM_CLASSES,
        dropout_rate=cfg.DROPOUT_RATE
    )
    
    
    # --- 4. Prepare Trainer ---
    if accelerator.is_main_process:
        print("\n--- [3/4] Loading Trainer... ---")
        
    #  accelerator
    trainer = EGFRTrainer(model, dataloaders, cfg, accelerator, num_of_clusters=cfg.NUM_OF_CLUSTRERS)
    
    # --- 5. Start Training ---
    if accelerator.is_main_process:
        print("\n" + "="*60)
        print("🔥 Start Training!")
        print("="*60)
    
    trainer.fit()
    
    if accelerator.is_main_process:
        print("\n🎉 Mission Accomplished")

if __name__ == '__main__':
    main()