from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
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


# MMLU dataset categories
MMLU_CATEGORIES = {
    "humanities": [
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "prehistory",
        "professional_law",
        "world_religions",
    ],
    "other": [
        "business_ethics",
        "clinical_knowledge",
        "college_medicine",
        "global_facts",
        "human_aging",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "nutrition",
        "professional_accounting",
        "professional_medicine",
        "virology",
    ],
    "social_sciences": [
        "econometrics",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "human_sexuality",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    ],
    "stem": [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "electrical_engineering",
        "elementary_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
    ],
}


def format_mmlu_sample(sample):
    """
    Format an MMLU sample into a prompt string.
    
    Args:
        sample: Dictionary with keys 'question', 'choices', 'answer'
    
    Returns:
        str: Formatted prompt string
    """
    question = sample.get("question", "")
    choices = sample.get("choices", [])
    answer_idx = sample.get("answer", 0)
    
    # Format choices as A, B, C, D
    choice_labels = ["A", "B", "C", "D"]
    choices_text = "\n".join([f"{label}. {choice}" for label, choice in zip(choice_labels, choices)])
    
    # Format the prompt
    prompt = f"{question}\n\n{choices_text}\n\nAnswer:"
    return prompt


def prepare_mmlu_prompts(topic="social_sciences", max_samples_per_subject=5, seed=42):
    """
    Prepare prompts from MMLU dataset for a specific topic.
    
    Args:
        topic: One of "humanities", "other", "social_sciences", "stem"
        max_samples_per_subject: Maximum number of samples to take per subject
        seed: Random seed for reproducibility
    
    Returns:
        list: List of formatted prompt strings
    """
    if topic not in MMLU_CATEGORIES:
        raise ValueError(f"Unknown topic '{topic}'. Must be one of: {list(MMLU_CATEGORIES.keys())}")
    
    # Load MMLU dataset
    all_samples = load_dataset("cais/mmlu", "all")
    samples = concatenate_datasets([all_samples['dev'], all_samples['validation']])
    
    # Get subjects for the topic
    subjects = set(MMLU_CATEGORIES[topic])
    subject_samples = defaultdict(list)
    
    # Collect samples per subject
    for s in samples:
        subj = s.get("subject")
        if subj in subjects and len(subject_samples[subj]) < max_samples_per_subject:
            subject_samples[subj].append(s)
    
    # Flatten and format
    all_samples_list = [s for subs in subject_samples.values() for s in subs]
    prompts = [format_mmlu_sample(sample) for sample in all_samples_list]
    
    return prompts


def prepare_gsm8k_prompts(sample_size=50, seed=42):
    """
    Prepare prompts from GSM8K dataset.

    Args:
        sample_size: Number of examples to randomly sample
        seed: Random seed for reproducibility

    Returns:
        list: List of formatted prompt strings
    """
    # Load GSM8K dataset
    gsm8k = load_dataset("openai/gsm8k", "main")["train"]

    # Randomly sample examples
    random.seed(seed)
    sample_indices = random.sample(range(len(gsm8k)), min(sample_size, len(gsm8k)))
    gsm8k_sample = gsm8k.select(sample_indices)

    # Format prompts
    prompts = [
        f"Question: {ex['question']}\nAnswer:"
        for ex in gsm8k_sample
    ]

    return prompts


def prepare_aime25_prompts(sample_size=None, seed=42):
    """
    Prepare prompts from AIME 2025 dataset (MathArena/aime_2025).

    Args:
        sample_size: Number of examples to randomly sample (None = use all)
        seed: Random seed for reproducibility

    Returns:
        list: List of formatted prompt strings
    """
    aime = load_dataset("MathArena/aime_2025", split="train")

    if sample_size is not None and sample_size < len(aime):
        random.seed(seed)
        indices = random.sample(range(len(aime)), sample_size)
        aime = aime.select(indices)

    prompts = [
        f"Problem: {ex['problem']}\nAnswer:"
        for ex in aime
    ]

    return prompts


def prepare_humaneval_prompts(sample_size=None, seed=42):
    """
    Prepare prompts from HumanEval dataset (openai/openai_humaneval).

    Uses only the `prompt` field (function signature + docstring).

    Args:
        sample_size: Number of examples to randomly sample (None = use all)
        seed: Random seed for reproducibility

    Returns:
        list: List of prompt strings
    """
    humaneval = load_dataset("openai/openai_humaneval", split="test")

    if sample_size is not None and sample_size < len(humaneval):
        random.seed(seed)
        indices = random.sample(range(len(humaneval)), sample_size)
        humaneval = humaneval.select(indices)

    prompts = [ex["prompt"] for ex in humaneval]

    return prompts
