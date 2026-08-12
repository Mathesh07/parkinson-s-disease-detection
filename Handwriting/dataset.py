import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import (BATCH_SIZE, CSV_FILES, DATA_ROOT, IMAGE_FOLDERS,
                    IMAGE_SIZE, RAW_CLASS_LABELS, RANDOM_SEED,
                    SPLIT_MANIFEST_PATH)


@dataclass
class ImageSample:
    image_path: str
    label: int
    patient_id: int
    source_csv: str
    image_name: str
    task: str


def load_csv_datasets() -> List[pd.DataFrame]:
    """Load the spiral and meander CSV files from the NewHandPD folder."""
    frames = []
    for csv_name in CSV_FILES:
        csv_path = DATA_ROOT / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        df["source_csv"] = csv_name
        df["task"] = "spiral" if "Spiral" in csv_name else "meander"
        frames.append(df)
    return frames


def resolve_image_path(image_name: str) -> Optional[str]:
    """Resolve the image name from the CSV into a real file path."""
    if pd.isna(image_name) or not str(image_name).strip():
        return None
    image_name = str(image_name).strip()
    for task in IMAGE_FOLDERS.values():
        for folder in task.values():
            candidate = folder / image_name
            if candidate.exists():
                return str(candidate)
    return None


def build_manifest() -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Build a combined manifest from the spiral and meander CSV files."""
    frames = load_csv_datasets()
    manifest = pd.concat(frames, ignore_index=True)
    manifest["image_path"] = manifest["IMAGE_NAME"].apply(resolve_image_path)
    manifest["image_exists"] = manifest["image_path"].notna()
    manifest["label_name"] = manifest["CLASS_TYPE"].map(RAW_CLASS_LABELS)
    manifest["target"] = (manifest["CLASS_TYPE"] == 2).astype(int)
    manifest["source_label"] = manifest["CLASS_TYPE"]
    summary = summarize_dataset(manifest)
    return manifest, summary


def summarize_dataset(manifest: pd.DataFrame) -> Dict[str, object]:
    """Create a concise dataset report from the manifest."""
    summary = {
        "unique_patients": int(manifest["ID_PATIENT"].nunique()),
        "total_images": int(len(manifest)),
        "unique_classes": sorted(manifest["CLASS_TYPE"].dropna().unique().tolist()),
        "images_per_class": manifest["CLASS_TYPE"].value_counts().to_dict(),
        "patients_with_multiple_images": int((manifest["ID_PATIENT"].value_counts() > 1).sum()),
        "images_per_patient": manifest["ID_PATIENT"].value_counts().sort_values(ascending=False).to_dict(),
        "missing_images": int((~manifest["image_exists"]).sum()),
    }
    return summary


def print_dataset_summary(summary: Dict[str, object], manifest: pd.DataFrame) -> None:
    """Print a beginner-friendly dataset summary."""
    print("\n=== Dataset Inspection Summary ===")
    print(f"Unique patients: {summary['unique_patients']}")
    print(f"Total images in CSVs: {summary['total_images']}")
    print(f"Classes found: {summary['unique_classes']} (1=Healthy, 2=Parkinson)")
    print("Images per class:")
    for label, count in summary["images_per_class"].items():
        name = RAW_CLASS_LABELS.get(label, str(label))
        print(f"  - CLASS_TYPE {label} ({name}): {count} images")
    print(f"Patients with multiple images: {summary['patients_with_multiple_images']}")
    print(f"Missing image references: {summary['missing_images']}")
    if summary["missing_images"] > 0:
        missing_rows = manifest[~manifest["image_exists"]][["IMAGE_NAME", "source_csv", "CLASS_TYPE", "ID_PATIENT"]]
        print("Missing image references (skipped automatically):")
        print(missing_rows.to_string(index=False))


def create_patient_level_split(manifest: pd.DataFrame, test_fold: int = 0, val_fold: int = 1) -> pd.DataFrame:
    """Split the dataset by PATIENT ID so no patient appears in more than one split."""
    # Work at patient level to ensure zero patient leakage
    patient_level = manifest.groupby("ID_PATIENT", as_index=False).first()[["ID_PATIENT", "target"]].copy()
    X = np.zeros(len(patient_level))
    y = patient_level["target"].to_numpy()
    groups = patient_level["ID_PATIENT"].to_numpy()

    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    split_map = {}
    for fold_idx, (_, group_idx) in enumerate(splitter.split(X, y, groups)):
        patient_ids = patient_level.iloc[group_idx]["ID_PATIENT"].tolist()
        if fold_idx == test_fold:
            split_map["test"] = patient_ids
        elif fold_idx == val_fold:
            split_map["val"] = patient_ids

    patient_level["split"] = "train"
    patient_level.loc[patient_level["ID_PATIENT"].isin(split_map.get("test", [])), "split"] = "test"
    patient_level.loc[patient_level["ID_PATIENT"].isin(split_map.get("val", [])), "split"] = "val"

    if "split" in manifest.columns:
        manifest = manifest.drop(columns=["split"])

    manifest = manifest.merge(patient_level[["ID_PATIENT", "split"]], on="ID_PATIENT", how="left")
    manifest.to_csv(SPLIT_MANIFEST_PATH, index=False)
    return manifest


class HandwritingImageDataset(Dataset):
    """PyTorch Dataset loading handwriting images for Parkinson's detection."""

    def __init__(self, samples: List[ImageSample], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.label


def create_transforms(image_size: int = IMAGE_SIZE, train: bool = False):
    """Return image preprocessing and data augmentation transforms.
    
    Training:
    - Resize & Random Crop to image_size
    - Slight Random Rotation (+/- 10 deg)
    - Random Horizontal Flip
    - ImageNet Normalization
    
    Validation / Testing:
    - Deterministic Resize to image_size
    - ImageNet Normalization
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_samples(manifest: pd.DataFrame, split: str) -> List[ImageSample]:
    """Build list of ImageSample objects for a given split filtering missing images."""
    subset = manifest[(manifest["split"] == split) & (manifest["image_exists"])]
    return [
        ImageSample(
            image_path=row["image_path"],
            label=int(row["target"]),
            patient_id=int(row["ID_PATIENT"]),
            source_csv=row["source_csv"],
            image_name=row["IMAGE_NAME"],
            task=row["task"],
        )
        for _, row in subset.iterrows()
    ]


def create_dataloaders(manifest: pd.DataFrame, batch_size: int = BATCH_SIZE, image_size: int = IMAGE_SIZE, num_workers: int = 0):
    """Create train, validation, and test PyTorch DataLoaders with patient-isolated splits."""
    train_samples = build_samples(manifest, "train")
    val_samples = build_samples(manifest, "val")
    test_samples = build_samples(manifest, "test")

    train_dataset = HandwritingImageDataset(train_samples, transform=create_transforms(image_size=image_size, train=True))
    val_dataset = HandwritingImageDataset(val_samples, transform=create_transforms(image_size=image_size, train=False))
    test_dataset = HandwritingImageDataset(test_samples, transform=create_transforms(image_size=image_size, train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    manifest, summary = build_manifest()
    print_dataset_summary(summary, manifest)
    manifest = create_patient_level_split(manifest)
    print("\nSplit counts by patient split (images):")
    print(manifest["split"].value_counts().to_string())
    print("\nSplit counts by class and split:")
    print(pd.crosstab(manifest["split"], manifest["target"]).to_string())

