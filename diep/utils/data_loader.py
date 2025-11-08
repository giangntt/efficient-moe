"""
Data loading utilities for DiEP training
Supports various datasets and preprocessing pipelines
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer
from datasets import load_dataset, Dataset as HFDataset
from typing import Optional, List, Dict, Union
import random
from tqdm import tqdm


class CalibrationDataset(Dataset):
    """
    Dataset wrapper for calibration data.
    Handles tokenization and preprocessing.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True
    ):
        """
        Args:
            texts: List of text strings
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


class StreamingCalibrationDataset(Dataset):
    """
    Streaming dataset for large calibration datasets.
    Tokenizes on-the-fly to save memory.
    """
    
    def __init__(
        self,
        dataset: HFDataset,
        tokenizer: PreTrainedTokenizer,
        text_column: str = "text",
        max_length: int = 512,
        num_samples: Optional[int] = None
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.max_length = max_length
        
        # Limit dataset size if specified
        if num_samples and len(dataset) > num_samples:
            self.dataset = dataset.select(range(num_samples))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        text = self.dataset[idx][self.text_column]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


def load_wikitext_dataset(
    tokenizer: PreTrainedTokenizer,
    split: str = "train",
    max_length: int = 512,
    num_samples: Optional[int] = None,
    config: str = "wikitext-2-raw-v1"
) -> Dataset:
    """
    Load WikiText dataset for calibration/training.
    
    Args:
        tokenizer: Tokenizer instance
        split: Dataset split (train/validation/test)
        max_length: Maximum sequence length
        num_samples: Number of samples to load (None = all)
        config: WikiText configuration
        
    Returns:
        Dataset instance
    """
    print(f"Loading WikiText dataset ({config}, {split})...")
    
    # Load dataset
    dataset = load_dataset("wikitext", config, split=split)
    
    # Filter empty texts
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 0)
    
    # Limit samples if specified
    if num_samples and len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))
    
    # Extract texts
    texts = [sample['text'] for sample in dataset]
    
    print(f"Loaded {len(texts)} samples")
    
    return CalibrationDataset(texts, tokenizer, max_length=max_length)


def load_c4_dataset(
    tokenizer: PreTrainedTokenizer,
    split: str = "train",
    max_length: int = 512,
    num_samples: int = 1000,
    streaming: bool = True
) -> Dataset:
    """
    Load C4 dataset for calibration/training.
    
    Args:
        tokenizer: Tokenizer instance
        split: Dataset split
        max_length: Maximum sequence length
        num_samples: Number of samples to load
        streaming: Use streaming mode
        
    Returns:
        Dataset instance
    """
    print(f"Loading C4 dataset ({split}, {num_samples} samples)...")
    
    # Load dataset (streaming mode for large dataset)
    if streaming:
        dataset = load_dataset("allenai/c4", "en", split=split, streaming=True)
        # Take first num_samples
        texts = []
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
            if len(sample['text'].strip()) > 0:
                texts.append(sample['text'])
        
        print(f"Loaded {len(texts)} samples")
        return CalibrationDataset(texts, tokenizer, max_length=max_length)
    else:
        dataset = load_dataset("allenai/c4", "en", split=split)
        dataset = dataset.filter(lambda x: len(x['text'].strip()) > 0)
        return StreamingCalibrationDataset(
            dataset, tokenizer, text_column='text', 
            max_length=max_length, num_samples=num_samples
        )


def load_custom_text_dataset(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512
) -> Dataset:
    """
    Load custom text dataset.
    
    Args:
        texts: List of text strings
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        
    Returns:
        Dataset instance
    """
    print(f"Creating custom dataset with {len(texts)} samples...")
    return CalibrationDataset(texts, tokenizer, max_length=max_length)


def load_json_dataset(
    file_path: str,
    tokenizer: PreTrainedTokenizer,
    text_field: str = "text",
    max_length: int = 512,
    num_samples: Optional[int] = None
) -> Dataset:
    """
    Load dataset from JSON file.
    
    Args:
        file_path: Path to JSON file
        tokenizer: Tokenizer instance
        text_field: Field name containing text
        max_length: Maximum sequence length
        num_samples: Number of samples to load
        
    Returns:
        Dataset instance
    """
    print(f"Loading JSON dataset from {file_path}...")
    
    import json
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract texts
    if isinstance(data, list):
        texts = [item[text_field] for item in data if text_field in item]
    else:
        texts = [data[text_field]] if text_field in data else []
    
    # Limit samples
    if num_samples and len(texts) > num_samples:
        texts = texts[:num_samples]
    
    print(f"Loaded {len(texts)} samples")
    return CalibrationDataset(texts, tokenizer, max_length=max_length)


def create_mixed_dataset(
    tokenizer: PreTrainedTokenizer,
    datasets: List[str],
    num_samples_per_dataset: int = 500,
    max_length: int = 512
) -> Dataset:
    """
    Create mixed dataset from multiple sources.
    Useful for diverse calibration data.
    
    Args:
        tokenizer: Tokenizer instance
        datasets: List of dataset names
        num_samples_per_dataset: Samples per dataset
        max_length: Maximum sequence length
        
    Returns:
        Combined dataset
    """
    print(f"Creating mixed dataset from: {datasets}")
    
    all_texts = []
    
    for dataset_name in datasets:
        if dataset_name == "wikitext":
            dataset = load_wikitext_dataset(
                tokenizer, num_samples=num_samples_per_dataset, max_length=max_length
            )
            all_texts.extend(dataset.texts)
        
        elif dataset_name == "c4":
            dataset = load_c4_dataset(
                tokenizer, num_samples=num_samples_per_dataset, max_length=max_length
            )
            all_texts.extend(dataset.texts)
        
        else:
            print(f"Warning: Unknown dataset '{dataset_name}', skipping...")
    
    # Shuffle combined texts
    random.shuffle(all_texts)
    
    print(f"Created mixed dataset with {len(all_texts)} total samples")
    return CalibrationDataset(all_texts, tokenizer, max_length=max_length)


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create DataLoader from dataset.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches
    )


def prepare_calibration_dataloaders(
    tokenizer: PreTrainedTokenizer,
    dataset_name: str = "wikitext",
    num_train_samples: int = 5000,
    num_val_samples: int = 1000,
    batch_size: int = 4,
    max_length: int = 512,
    num_workers: int = 2
) -> tuple:
    """
    Prepare train and validation dataloaders for calibration.
    
    Args:
        tokenizer: Tokenizer instance
        dataset_name: Dataset name (wikitext, c4, or path to JSON)
        num_train_samples: Number of training samples
        num_val_samples: Number of validation samples
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of workers
        
    Returns:
        (train_dataloader, val_dataloader)
    """
    print("\n" + "=" * 60)
    print("Preparing Calibration Dataloaders")
    print("=" * 60)
    
    # Load training dataset
    if dataset_name == "wikitext":
        train_dataset = load_wikitext_dataset(
            tokenizer, split="train", max_length=max_length, 
            num_samples=num_train_samples
        )
        val_dataset = load_wikitext_dataset(
            tokenizer, split="validation", max_length=max_length,
            num_samples=num_val_samples
        )
    
    elif dataset_name == "c4":
        train_dataset = load_c4_dataset(
            tokenizer, split="train", max_length=max_length,
            num_samples=num_train_samples
        )
        val_dataset = load_c4_dataset(
            tokenizer, split="validation", max_length=max_length,
            num_samples=num_val_samples
        )
    
    elif dataset_name.endswith('.json'):
        # Load from JSON file
        all_dataset = load_json_dataset(
            dataset_name, tokenizer, max_length=max_length
        )
        # Split into train/val
        train_size = min(num_train_samples, int(0.8 * len(all_dataset)))
        val_size = min(num_val_samples, len(all_dataset) - train_size)
        
        train_texts = all_dataset.texts[:train_size]
        val_texts = all_dataset.texts[train_size:train_size + val_size]
        
        train_dataset = CalibrationDataset(train_texts, tokenizer, max_length)
        val_dataset = CalibrationDataset(val_texts, tokenizer, max_length)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers
    )
    
    val_dataloader = create_dataloader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    
    print(f"\nDataloaders created:")
    print(f"  Train batches: {len(train_dataloader)}")
    print(f"  Val batches: {len(val_dataloader)}")
    print(f"  Batch size: {batch_size}")
    
    return train_dataloader, val_dataloader


class DataCollatorForLanguageModeling:
    """
    Data collator for language modeling.
    Handles padding and label creation.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        mlm: bool = False,
        mlm_probability: float = 0.15
    ):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability
    
    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Stack examples
        batch = {
            'input_ids': torch.stack([ex['input_ids'] for ex in examples]),
            'attention_mask': torch.stack([ex['attention_mask'] for ex in examples])
        }
        
        # Create labels for causal LM (shifted input_ids)
        batch['labels'] = batch['input_ids'].clone()
        
        return batch