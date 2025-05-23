import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import pickle
from transformers import ASTFeatureExtractor, ASTConfig
import random

from representations.Embedder import Embedder

class EmbeddingsDataset(Dataset):
    """
    Base class for datasets using AST embeddings
    """
    def __init__(self, 
                 root_dir: str, 
                 dataset_name: str = "GTZAN",
                 split: str = "train",
                 sample_rate: int = 16000, 
                 embeddings_file: str = None,
                 embedding_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
                 cache_dir: str = "./embeddings_cache",
                 samples_per_class: int = None):
        """
        Base dataset class for audio embeddings datasets.

        Args:
            root_dir (str): Path to the dataset directory.
            dataset_name (str): Name of the dataset ("GTZAN" or "MoodsMIREX").
            split (str): Dataset split. Options: "train", "val", "all", or "subset".
            sample_rate (int): Sampling rate for audio files.
            embeddings_file (str): Optional path to a file with pre-computed embeddings.
            embedding_model (str): Name of the model to use for embeddings.
            cache_dir (str): Directory to cache embeddings.
            samples_per_class (int): Number of samples per class when split="subset".
        """
        self.root_dir = os.path.join(root_dir, dataset_name)
        self.dataset_name = dataset_name
        self.split = split
        self.sample_rate = sample_rate
        self.embedding_model = embedding_model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.samples_per_class = samples_per_class
        
        # These attributes will be set by child classes
        self.embeddings = None
        self.all_labels = None
        self.indices = None
        
        # Setup embeddings
        self._setup_embeddings(embeddings_file)
    
    def _setup_embeddings(self, embeddings_file):
        """Set up the dataset for embeddings"""
        if self.dataset_name not in ["GTZAN", "MoodsMIREX", "AUDIO_MNIST"]:
            raise ValueError("dataset_name must be either 'GTZAN' or 'MoodsMIREX'")
        
        if embeddings_file is None:
            # Default embeddings file based on model
            model_name = self.embedding_model.replace('/', '_')
            embeddings_file = self.cache_dir / f"{self.dataset_name.lower()}_embeddings_{model_name}.pkl"
        else:
            embeddings_file = Path(embeddings_file)
        
        # Check if embeddings file exists
        if embeddings_file.exists():
            print(f"Loading embeddings from {embeddings_file}")
            with open(embeddings_file, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data['embeddings']
                self.all_labels = data['labels']
                self.label_to_idx = data.get('label_to_idx', {label: label for label in np.unique(self.all_labels)})
        else:
            print(f"Embeddings file not found. Generating embeddings...")
            # Create embedder and extract embeddings
            embedder = Embedder(model_name=self.embedding_model, cache_dir=self.cache_dir)
            if self.dataset_name == "GTZAN":
                self.embeddings, self.all_labels, self.label_to_idx = embedder.get_embeddings_GTZAN(
                    base_path=self.root_dir, force_recalculate=True
                )
            elif self.dataset_name == "MoodsMIREX":
                self.embeddings, self.all_labels, self.label_to_idx = embedder.get_embeddings_MoodsMIREX(
                    base_path=self.root_dir, force_recalculate=True
                )
            elif self.dataset_name == "AUDIO_MNIST":
                self.embeddings, self.all_labels = embedder.get_embeddings_AUDIO_MNIST(
                    base_path=self.root_dir, force_recalculate=True
                )

        self.class_names = list(self.label_to_idx.keys())
        # Create indices based on split type
        self._create_indices()

    def _create_indices(self):
        """
        Create indices based on the specified split
        """
        if self.split == "all":
            # Use all data
            self.indices = torch.arange(len(self.all_labels))
        elif self.split == "subset":
            # Create a balanced subset with samples_per_class samples per class
            if self.samples_per_class is None:
                raise ValueError("samples_per_class must be specified when split='subset'")
            self._create_subset_indices()
        else:
            # Create train/val split
            self._split_indices()
    
    def _create_subset_indices(self):
        """
        Create indices for a balanced subset with samples_per_class samples per class,
        but only selecting from the training portion of the data.
        """
        # First, create train/val split by class
        class_indices = {}
        train_indices_by_class = {}
        
        # Get indices for each class
        for class_name in self.class_names:
            class_idx = self.label_to_idx[class_name]
            class_indices[class_name] = torch.where(self.all_labels == class_idx)[0]
            
            # Split into train/val (80/20)
            split_idx = int(0.8 * len(class_indices[class_name]))
            # Only store the training indices for sampling
            train_indices_by_class[class_name] = class_indices[class_name][:split_idx].tolist()
        
        # Now sample from training indices only
        subset_indices = []
        for class_name, indices in train_indices_by_class.items():
            if len(indices) < self.samples_per_class:
                print(f"Warning: Class {class_name} has only {len(indices)} training samples, "
                    f"which is less than the requested {self.samples_per_class}. "
                    f"Using all available training samples for this class.")
                subset_indices.extend(indices)
            else:
                # Sample randomly without replacement from training set
                sampled_indices = random.sample(indices, self.samples_per_class)
                subset_indices.extend(sampled_indices)
        
        # Convert to tensor
        self.indices = torch.tensor(subset_indices)
        
        print(f"Created subset dataset with {len(self.indices)} samples "
            f"({len(class_indices)} classes with ~{self.samples_per_class} samples each), "
            f"sampled only from training data")

    def _split_indices(self):
        """
        Create train/val split indices ensuring class balance
        """
        # Create indices for splits by class
        class_indices = {}
        for class_name in self.class_names:
            class_idx = self.label_to_idx[class_name]
            class_indices[class_name] = torch.where(self.all_labels == class_idx)[0]
        
        # Split by class to ensure balance
        train_indices = []
        val_indices = []
        for class_name, indices in class_indices.items():
            split_idx = int(0.8 * len(indices))
            train_indices.append(indices[:split_idx])
            val_indices.append(indices[split_idx:])
        
        # Combine indices
        train_indices = torch.cat(train_indices)
        val_indices = torch.cat(val_indices)
        
        # Select appropriate indices based on split
        if self.split == "train":
            self.indices = train_indices
        else:  # "val"
            self.indices = val_indices
 
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        # Get embedding and label directly
        index = self.indices[idx]
        embedding = self.embeddings[index]
        label = self.all_labels[index]
        return embedding, label


if __name__ == "__main__":
    # Example usage for GTZAN
    dataset = EmbeddingsDataset(root_dir="data", dataset_name="GTZAN", split="all")
    print(f"Number of samples: {len(dataset)}")
    # print(f"Sample data: {dataset[0]}")
    
    # Example usage with subset
    dataset_subset = EmbeddingsDataset(
        root_dir="data", 
        dataset_name="GTZAN", 
        split="subset", 
        samples_per_class=5
    )
    print(f"Number of samples in subset: {len(dataset_subset)}")

    dataset_subset = EmbeddingsDataset(
        root_dir="data", 
        dataset_name="AUDIO_MNIST", 
        split="train", 
        # embeddings_file = "embeddings_cache/mnist_audio_embeddings_MIT_ast-finetuned-audioset-10-10-0.4593.pkl"
        # embedding_model="MIT/ast-finetuned-audioset-10-10-0.4593"
    )
    breakpoint()
    print(f"Number of samples in AUDIO_MNIST: {len(dataset_subset)}")