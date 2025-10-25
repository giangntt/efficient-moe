from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch
import random

class PackedTextDataset(Dataset):
    """
    Packs multiple short texts into fixed-length chunks for efficient training.
    This is useful for language models to reduce padding and improve GPU utilization.
    """
    def __init__(self, texts, tokenizer, max_length=1024, separator_token_id=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Use provided separator or default to EOS token
        self.sep_token_id = separator_token_id if separator_token_id is not None else tokenizer.eos_token_id

        # Clean and filter texts
        texts = [t.strip() for t in texts if t.strip()]
        tokenized_texts = [tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]

        # Pack all tokens together
        all_tokens = []
        for seq in tokenized_texts:
            if len(seq) >= max_length:
                seq = seq[:max_length - 1]  # leave space for separator
            all_tokens.extend(seq + [self.sep_token_id])

        # Split into fixed-length chunks
        self.chunks = []
        for i in range(0, len(all_tokens), max_length):
            chunk = all_tokens[i:i + max_length]
            if len(chunk) == max_length:  # keep only full-length chunks
                self.chunks.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return {"input_ids": chunk, "attention_mask": torch.ones_like(chunk)}

def create_packed_dataloader(tokenizer, dataset_name, split="train", max_length=1024, 
                            batch_size=8, sample_size=None, separator_token_id=None, 
                            shuffle=False, num_workers=4, seed=42):
    """
    Create a DataLoader with packed text data.
    """
    dataset = load_dataset(dataset_name, split=split)
    texts = [x["text"] for x in dataset if len(x["text"].strip()) > 0]
    
    random.seed(seed)
    if sample_size is not None and sample_size < len(texts):
        texts = random.sample(texts, sample_size)

    packed_dataset = PackedTextDataset(texts, tokenizer, max_length=max_length, 
                                      separator_token_id=separator_token_id)

    return DataLoader(
        packed_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
