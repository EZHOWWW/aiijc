# %%
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import os

# %%
# --- Configuration ---
random_state = 42
DATA_DIR = 'data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
TRAIN_CSV = os.path.join(TRAIN_DIR, '_classes.csv')
VALID_CSV = os.path.join(VALID_DIR, '_classes.csv')

IMG_SIZE = 224
BATCH_SIZE = 32
# ImageNet mean and std for normalization
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# %%
class AnimalTracksDataset(Dataset):
    """Custom Dataset for loading animal tracks images."""
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        # Get column names for classes (e.g., 'Bear', 'Bird', ...)
        self.class_columns = self.df.columns[1:] 
        # Convert one-hot encoded labels to single class index
        self.labels = self.df[self.class_columns].values.argmax(axis=1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get image path and label
        img_path = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
            
        return image, label

# %%
# --- Transformations ---
# Define augmentations for training and validation sets.
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

valid_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

# --- Datasets and DataLoaders ---
# Create instances of our dataset
train_dataset = AnimalTracksDataset(csv_path=TRAIN_CSV, img_dir=TRAIN_DIR, transform=train_transform)
valid_dataset = AnimalTracksDataset(csv_path=VALID_CSV, img_dir=VALID_DIR, transform=valid_transform)

# Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# %%
# --- Verification (Optional) ---
# Let's check if everything works correctly.
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(valid_dataset)}")

# Fetch one batch to see its shape
images, labels = next(iter(train_loader))
print(f"Images batch shape: {images.shape}") # Should be [BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE]
print(f"Labels batch shape: {labels.shape}")   # Should be [BATCH_SIZE]
print(f"Class names: {train_dataset.class_columns.tolist()}")
