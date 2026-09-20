"""
BBBP Dataset loader and PyTorch Dataset class.

Downloads the Blood-Brain Barrier Penetration dataset from MoleculeNet.
Dataset: ~2000 molecules with binary labels (BBB+ or BBB-)
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict
from sklearn.model_selection import train_test_split
import requests
from pathlib import Path


class BBBDataset(Dataset):
    """PyTorch Dataset for BBB penetration data."""

    def __init__(
        self,
        smiles_list: List[str],
        labels: List[int],
        tokenizer,
        augment: bool = False,
        augmentation_factor: int = 3
    ):
        """
        Args:
            smiles_list: List of SMILES strings
            labels: List of binary labels (0 or 1)
            tokenizer: SMILESTokenizer instance
            augment: Whether to use SMILES augmentation
            augmentation_factor: How many variants per molecule
        """
        self.smiles_list = smiles_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.augment = augment
        self.augmentation_factor = augmentation_factor

    def __len__(self) -> int:
        """How many molecules in the dataset."""
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get one molecule.
        
        PyTorch calls this for each molecule during training.
        Returns: Dictionary with 'input_ids', 'attention_mask', 'labels', 'smiles'
        """
        smiles = self.smiles_list[idx]
        label = self.labels[idx]

        # Data augmentation: use random SMILES variant
        if self.augment:
            from src.utils.smiles_tokenizer import smiles_enumeration
            variants = smiles_enumeration(smiles, self.augmentation_factor)
            smiles = np.random.choice(variants)

        # Tokenize the SMILES
        input_ids, attention_mask = self.tokenizer.encode(smiles)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long),
            'smiles': smiles
        }


def download_bbbp_dataset(data_dir: str = "./data") -> str:
    """
    Download BBBP dataset from MoleculeNet.
    
    Returns:
        Path to downloaded CSV file
    """
    os.makedirs(data_dir, exist_ok=True)

    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
    output_path = os.path.join(data_dir, "BBBP.csv")

    # Already downloaded?
    if os.path.exists(output_path):
        print(f"✓ Dataset already exists at {output_path}")
        return output_path

    print(f"⬇ Downloading BBBP dataset...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            f.write(response.content)

        print(f"✓ Downloaded successfully to {output_path}")
        return output_path

    except Exception as e:
        print(f"✗ Error downloading: {e}")
        print(f"Manual download: https://moleculenet.org/datasets-1")
        raise


def load_bbbp_data(
    data_path: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and split BBBP dataset into train/val/test.
    
    Returns:
        train_df, val_df, test_df (pandas DataFrames)
    """
    print(f"\n📖 Loading dataset from {data_path}...")
    
    # Read CSV
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} molecules")

    # Find SMILES column (might be named differently)
    if 'smiles' not in df.columns:
        for col in ['SMILES', 'Smiles', 'mol']:
            if col in df.columns:
                df['smiles'] = df[col]
                break

    # Find label column
    if 'p_np' not in df.columns:
        for col in ['label', 'target', 'y', 'BBB']:
            if col in df.columns:
                df['p_np'] = df[col]
                break

    # Validate we have required columns
    assert 'smiles' in df.columns, "Could not find SMILES column"
    assert 'p_np' in df.columns, "Could not find label column"

    # Clean data
    df = df.dropna(subset=['smiles', 'p_np'])
    df['p_np'] = df['p_np'].astype(int)

    # Show class distribution
    class_counts = df['p_np'].value_counts().sort_index()
    print(f"\n📊 Class distribution:")
    print(f"   BBB- (0): {class_counts.get(0, 0)} molecules ({class_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"   BBB+ (1): {class_counts.get(1, 0)} molecules ({class_counts.get(1, 0)/len(df)*100:.1f}%)")

    # Split into train+val and test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=df['p_np']
    )

    # Split train+val into train and val
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        random_state=random_seed,
        stratify=train_val_df['p_np']
    )

    print(f"\n✂️  Split sizes:")
    print(f"   Training: {len(train_df)} molecules")
    print(f"   Validation: {len(val_df)} molecules")
    print(f"   Test: {len(test_df)} molecules")

    return train_df, val_df, test_df


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    batch_size: int = 64,
    num_workers: int = 0,
    augment_train: bool = True,
    augmentation_factor: int = 3
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print(f"\n🔧 Creating DataLoaders...")
    
    # Create datasets
    train_dataset = BBBDataset(
        train_df['smiles'].tolist(),
        train_df['p_np'].tolist(),
        tokenizer,
        augment=augment_train,
        augmentation_factor=augmentation_factor
    )

    val_dataset = BBBDataset(
        val_df['smiles'].tolist(),
        val_df['p_np'].tolist(),
        tokenizer,
        augment=False
    )

    test_dataset = BBBDataset(
        test_df['smiles'].tolist(),
        test_df['p_np'].tolist(),
        tokenizer,
        augment=False
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    print(f"✓ DataLoaders created")
    return train_loader, val_loader, test_loader


# ---- Test code ----
if __name__ == "__main__":
    from src.utils.smiles_tokenizer import SMILESTokenizer

    print("=" * 70)
    print("BBBP Dataset Loader Test")
    print("=" * 70)

    # Download data
    data_path = download_bbbp_dataset("./data")

    # Load and split
    train_df, val_df, test_df = load_bbbp_data(data_path)

    # Create tokenizer
    tokenizer = SMILESTokenizer(max_length=120)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df,
        tokenizer,
        batch_size=32,
        num_workers=0,
        augment_train=False
    )

    # Test one batch
    print(f"\n🧪 Testing batch loading...")
    batch = next(iter(train_loader))

    print(f"\n📦 Batch contents:")
    print(f"   input_ids shape: {batch['input_ids'].shape}")
    print(f"   attention_mask shape: {batch['attention_mask'].shape}")
    print(f"   labels shape: {batch['labels'].shape}")
    print(f"\n📝 Example molecule:")
    print(f"   SMILES: {batch['smiles'][0]}")
    print(f"   Label: {batch['labels'][0].item()} {'(BBB+)' if batch['labels'][0].item() == 1 else '(BBB-)'}")
    print(f"   Input IDs: {batch['input_ids'][0][:30]}...")
    print(f"   Mask: {batch['attention_mask'][0][:30]}...")

    print("\n" + "=" * 70)
    print("✓ Test complete!")
    print("=" * 70)