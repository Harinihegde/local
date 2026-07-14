import torch

DENSITY_THRESHOLDS = {
    'sparse': (0, 15),
    'medium': (15, 40),
    'dense': (40, 1000)
}

BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
PATIENCE = 5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_ROOT = 'shanghaitech_data/shanghaitech_with_people_density_map/ShanghaiTech'