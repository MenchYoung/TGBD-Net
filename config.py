import torch

class Config:
    # --- 1. Directories ---
    CSV_PATH = r"/path/to/your/dataset.csv" 
    CONFIG_JSON_PATH = r"/path/to/your/config.json"
    WEIGHTS_PATH = r"/path/to/your/vision_tower_weights.pt"
    SAVE_DIR = "./experiment/TGBD_Net"

    # --- 2. Training Strategy ---
    EPOCHS = 135
    SEED = 42
    EARLY_STOP_PATIENCE = 50
    NUM_OF_CLUSTRERS = 4
    NUM_CLASSES = 2
    DROPOUT_RATE = 0.3
    
    # --- 3. Learning Strategy ---
    LR = 1e-4
    WARMUP_EPOCHS = 15
    UNFREEZE_EPOCH = 55
    ALPHA_ALIGN = 0.5
    
    # --- 4. Data and Hardware ---
    BATCH_SIZE = 8
    NUM_WORKERS = 2
    IMAGE_SIZE = (48, 256, 256)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

