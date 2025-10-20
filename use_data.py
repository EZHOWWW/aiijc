import json
import os
from glob import glob
from typing import Any, Callable, Dict, List, Optional

import torch
import trimesh
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class MultimodalDataset(Dataset):
    """
    A PyTorch Dataset for loading multimodal data (3D models, images, text).

    This dataset is designed to handle the specific data structure outlined in
    the project's documentation (USE_DATA.md). It supports loading data from
    different datasets (e.g., CustomDataset, Thingi10k) and splits (train, val, test).

    The directory structure is expected to be:
    <root_dir>/<dataset_name>/[category]/models/<model_name>.stl
    <root_dir>/<dataset_name>/[category]/images/<model_name>_[number].<jpeg/png>
    <root_dir>/<dataset_name>/[category]/captions.json

    Args:
        root_dir (str): The root directory where all datasets are stored.
        dataset_name (str): The name of the dataset to load.
        split (str): The data split to use ('train', 'validation', or 'test').
        transform_3d (Optional[Callable]): A function/transform to apply to the 3D model data.
        transform_image (Optional[Callable]): A function/transform to apply to the image data.
        transform_text (Optional[Callable]): A function/transform to apply to the text data.
        load_images (bool): Whether to load images.
        load_models (bool): Whether to load 3D models.
    """

    def __init__(
        self,
        root_dir: str = "/mnt/new_disk/aiijc_data",
        dataset_name: str = "train",
        split: str = "train",
        transform_3d: Optional[Callable] = None,
        transform_image: Optional[Callable] = None,
        transform_text: Optional[Callable] = None,
        load_images: bool = True,
        load_models: bool = True,
    ):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.split = split
        self.transform_3d = transform_3d
        self.transform_image = transform_image
        self.transform_text = transform_text
        self.load_images = load_images
        self.load_models = load_models

        self.dataset_path = os.path.join(self.root_dir, self.dataset_name)
        self.split_path = os.path.join(self.dataset_path, self.split)

        self.samples = self._create_samples()

    def _create_samples(self) -> List[Dict[str, Any]]:
        """
        Scans the dataset directory and creates a list of samples.
        Each sample is a dictionary containing paths to the model, images, and text.
        """
        samples = []
        model_paths = glob(
            os.path.join(self.split_path, "**/models/*.stl"), recursive=True
        )

        # Load captions if they exist
        captions = {}
        caption_files = glob(
            os.path.join(self.split_path, "**/captions.json"), recursive=True
        )
        for caption_file in caption_files:
            with open(caption_file, "r") as f:
                captions.update(json.load(f))

        for model_path in model_paths:
            model_name = os.path.splitext(os.path.basename(model_path))[0]

            # Find corresponding images
            image_dir = os.path.join(
                os.path.dirname(os.path.dirname(model_path)), "images"
            )
            image_paths = glob(os.path.join(image_dir, f"{model_name}_*.*"))

            # Get corresponding text
            text_data = captions.get(model_name, [])
            if isinstance(text_data, str):
                text_data = [text_data]

            samples.append(
                {
                    "model_name": model_name,
                    "model_path": model_path,
                    "image_paths": image_paths,
                    "texts": text_data,
                }
            )
        return samples

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Fetches a single sample from the dataset.

        Args:
            idx (int): The index of the sample to fetch.

        Returns:
            A dictionary containing the processed data for the sample:
            {
                "model_name": str,
                "model": (Optional) Tensor for the 3D model,
                "images": (Optional) List of Tensors for the images,
                "texts": List of strings
            }
        """
        sample = self.samples[idx]

        model_data = None
        if self.load_models:
            # Load 3D model (e.g., as a point cloud)
            try:
                mesh = trimesh.load(sample["model_path"], force="mesh")
                # Example: sample 1024 points from the mesh surface
                model_data = mesh.sample(1024)
                if self.transform_3d:
                    model_data = self.transform_3d(model_data)
            except Exception as e:
                print(
                    f"Warning: Could not load model {sample['model_path']}. Error: {e}"
                )
                model_data = None  # Or a placeholder

        image_data = []
        if self.load_images:
            for img_path in sample["image_paths"]:
                try:
                    image = Image.open(img_path).convert("RGB")
                    if self.transform_image:
                        image = self.transform_image(image)
                    image_data.append(image)
                except Exception as e:
                    print(
                        f"Warning: Could not load image {img_path}. Error: {e}"
                    )

        text_data = sample["texts"]
        if self.transform_text:
            text_data = [self.transform_text(t) for t in text_data]

        return {
            "model_name": sample["model_name"],
            "model": model_data,
            "images": image_data,
            "texts": text_data,
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Creates a DataLoader for a given dataset.

    This is optimized for performance, especially when dealing with data on a slow HDD,
    by using multiple workers and pinning memory.

    Args:
        dataset (Dataset): The dataset to load data from.
        batch_size (int): The number of samples per batch.
        shuffle (bool): Whether to shuffle the data at every epoch.
        num_workers (int): The number of subprocesses to use for data loading.
        pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory
                           before returning them.

    Returns:
        A DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,  # Use a custom collate function for flexible batching
    )


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to handle variable numbers of images and texts per sample.
    """
    model_names = [item["model_name"] for item in batch]
    models = [item["model"] for item in batch if item["model"] is not None]
    images = [
        img for item in batch for img in item["images"]
    ]  # Flatten list of images
    texts = [
        txt for item in batch for txt in item["texts"]
    ]  # Flatten list of texts

    # Stack models and images if they are tensors
    if models:
        models = torch.stack(models)
    # Note: Stacking images might require them to be the same size.
    # The transform_image should handle this (e.g., resize and convert to tensor).
    if images and all(isinstance(i, torch.Tensor) for i in images):
        images = torch.stack(images)

    return {
        "model_name": model_names,
        "model": models,
        "images": images,
        "texts": texts,
    }


# --- Usage Example ---
if __name__ == "__main__":
    """
    This section demonstrates how to use the MultimodalDataset and create a DataLoader.

    NOTE: This example assumes you have a dummy data structure in /tmp/aiijc_data
    for demonstration purposes. You would replace this path with your actual
    data root directory.
    """
    # 1. Create a dummy dataset structure for testing
    DUMMY_ROOT = "/mnt/new_disk/aiijc_data"
    DUMMY_DATASET = "train"
    DUMMY_SPLIT = ""

    # Create directories
    model_dir = os.path.join(DUMMY_ROOT, DUMMY_DATASET, DUMMY_SPLIT, "models")
    image_dir = os.path.join(DUMMY_ROOT, DUMMY_DATASET, DUMMY_SPLIT, "images")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # Create a dummy STL file (using trimesh to create a simple box)
    # dummy_mesh = trimesh.creation.box(extents=[1, 1, 1])
    # dummy_mesh.export(os.path.join(model_dir, "model1.stl"))
    # dummy_mesh.export(os.path.join(model_dir, "model2.stl"))

    # Create dummy image files
    # Image.new('RGB', (224, 224), color = 'red').save(os.path.join(image_dir, "model1_01.png"))
    # Image.new('RGB', (224, 224), color = 'green').save(os.path.join(image_dir, "model1_02.png"))
    # Image.new('RGB', (224, 224), color = 'blue').save(os.path.join(image_dir, "model2_01.png"))

    # Create a dummy captions file
    # captions_path = os.path.join(DUMMY_ROOT, DUMMY_DATASET, DUMMY_SPLIT, "captions.json")
    # with open(captions_path, 'w') as f:
    #     json.dump({
    #         "model1": "a red and green thing",
    #         "model2": ["a blue thing", "another description for the blue thing"]
    #     }, f)

    # print("--- Dummy data created in /tmp/aiijc_data ---")

    # 2. Define transformations (e.g., using torchvision)
    from torchvision import transforms

    image_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    def point_cloud_transform(pc):
        # Example transform: convert to tensor
        return torch.from_numpy(pc).float()

    # 3. Instantiate the dataset
    print("--- Initializing Dataset ---")
    multimodal_dataset = MultimodalDataset(
        root_dir=DUMMY_ROOT,
        dataset_name=DUMMY_DATASET,
        split=DUMMY_SPLIT,
        transform_image=image_transform,
        transform_3d=point_cloud_transform,
    )

    # 4. Check a single sample
    print(f"Dataset size: {len(multimodal_dataset)}")
    sample = multimodal_dataset[0]
    print("--- Sample 0 ---")
    print(f"Model Name: {sample['model_name']}")
    if sample["model"] is not None:
        print(f"Model (Point Cloud) Shape: {sample['model'].shape}")
    if sample["images"]:
        print(f"Number of Images: {len(sample['images'])}")
        print(f"Image 0 Shape: {sample['images'][0].shape}")
    print(f"Texts: {sample['texts']}")

    # 5. Create the DataLoader
    print("--- Creating DataLoader ---")
    dataloader = create_dataloader(
        multimodal_dataset, batch_size=16, shuffle=True
    )

    # 6. Iterate over a batch
    print("--- Fetching a Batch ---")
    try:
        batch = next(iter(dataloader))
        print(f"Batch Model Names: {batch['model_name']}")
        if batch["model"] is not None and len(batch["model"]) > 0:
            print(f"Batch Model Tensor Shape: {batch['model'].shape}")
        if batch["images"] is not None and len(batch["images"]) > 0:
            print(f"Batch Images Tensor Shape: {batch['images'].shape}")
        print(f"Batch Texts: {batch['texts']}")
    except Exception as e:
        print(f"Error creating or iterating dataloader: {e}")
        print(
            "This can happen if libraries like torch/trimesh are not installed."
        )
