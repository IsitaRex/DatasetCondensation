import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from transformers import ASTFeatureExtractor, ASTModel, ASTConfig
from pathlib import Path
import pickle

AST_SAMPLE_RATE = 16000
class Embedder:
    """
    Class for extracting and managing audio embeddings using the AST model
    """
    def __init__(self, model_name="MIT/ast-finetuned-audioset-10-10-0.4593", cache_dir="./embeddings_cache"):
        """
        Initialize the Embedder with an AST model
        
        Args:
            model_name: Name of the pre-trained Hugging Face model
            cache_dir: Directory to save processed embeddings
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Parameters
        self.sample_rate = AST_SAMPLE_RATE
        
        # Load feature extractor and model
        print(f"Loading model {model_name}...")
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
        self.model = ASTModel.from_pretrained(model_name)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Move model to GPU/MPS if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                  "mps" if torch.backends.mps.is_available() else 
                                  "cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)
    
    def load_audio(self, file_path):
        """
        Load and prepare an audio file
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            waveform: Normalized waveform
            sample_rate: Sample rate
        """
        waveform, sample_rate = torchaudio.load(file_path, normalize=True)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)
            
        return waveform.flatten(), self.sample_rate
    
    def extract_embedding(self, audio_array, sr):
        """
        Extract embedding from an audio array
        
        Args:
            audio_array: Audio array (waveform)
            sr: Sample rate
            
        Returns:
            embedding: Audio embedding (pooler_output)
        """
        # Resample if necessary
        if sr != self.sample_rate:
            audio_array = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(audio_array)
            
        # Extract features
        inputs = self.feature_extractor(audio_array, sampling_rate=self.sample_rate, return_tensors="pt")
        
        # Move inputs to the model's device
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
            
        # Get embeddings from the model
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Move result to CPU
        embedding = outputs.pooler_output.cpu()
        
        return embedding
    
    def get_embeddings_GTZAN(self, base_path="data/GTZAN", force_recalculate=False):
        """
        Process all audio files in the GTZAN folder and return embeddings and labels
        
        Args:
            base_path: Path to the GTZAN data folder
            force_recalculate: If True, recalculate embeddings even if cache file exists
            
        Returns:
            embeddings: Tensor with all embeddings (n_samples, embedding_dim)
            labels: Tensor with all labels (n_samples)
        """
        # Create cache filename based on model
        cache_file = self.cache_dir / f"gtzan_embeddings_{self.model_name.replace('/', '_')}.pkl"
        
        # Check if cache exists and load it if recalculation is not forced
        if cache_file.exists() and not force_recalculate:
            print(f"Loading embeddings from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                return cached_data['embeddings'], cached_data['labels']
        
        # If cache doesn't exist or recalculation is forced, process files
        print(f"Processing audio files in {base_path}...")
        
        embeddings_list = []
        labels_list = []
        genres = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        label_to_idx = {genre: idx for idx, genre in enumerate(sorted(genres))}
        
        # Process each genre
        for genre in sorted(genres):
            genre_path = os.path.join(base_path, genre)
            print(f"Processing genre: {genre}")
            
            # Get all WAV files
            audio_files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
            
            # Process each file
            for audio_file in tqdm(audio_files):
                file_path = os.path.join(genre_path, audio_file)
                
                try:
                    # Load and extract embedding
                    audio_array, sr = self.load_audio(file_path)
                    embedding = self.extract_embedding(audio_array, sr)
                    
                    # Save embedding and label
                    embeddings_list.append(embedding)
                    labels_list.append(label_to_idx[genre])
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        # Convert lists to tensors
        embeddings = torch.cat(embeddings_list, dim=0)
        labels = torch.tensor(labels_list, dtype=torch.long)
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'embeddings': embeddings,
                'labels': labels,
                'label_to_idx': label_to_idx
            }, f)
        
        print(f"GTZAN embeddings saved to cache: {cache_file}")
        print(f"Embeddings dimensions: {embeddings.shape}")
        print(f"Processed genres: {label_to_idx}")
        
        return embeddings, labels, label_to_idx
    
    def get_embeddings_MoodsMIREX(self, base_path="data/MoodsMIREX", force_recalculate=False):
      """
      Process audio files in the MoodsMIREX dataset and return embeddings and labels
      
      Args:
          base_path: Path to the MoodsMIREX data folder (containing Audio subfolder and clusters.txt)
          force_recalculate: If True, recalculate embeddings even if cache file exists
          
      Returns:
          embeddings: Tensor with all embeddings (n_samples, embedding_dim)
          labels: Tensor with all labels (n_samples)
          cluster_names: List of cluster names/labels
      """
      # Create cache filename based on model
      cache_file = self.cache_dir / f"moodsmirex_embeddings_{self.model_name.replace('/', '_')}.pkl"
      
      # Check if cache exists and load it if recalculation is not forced
      if cache_file.exists() and not force_recalculate:
          print(f"Loading MoodsMIREX embeddings from cache: {cache_file}")
          with open(cache_file, 'rb') as f:
              cached_data = pickle.load(f)
              return cached_data['embeddings'], cached_data['labels']
      
      # If cache doesn't exist or recalculation is forced, process files
      print(f"Processing MoodsMIREX audio files in {base_path}...")
      
      # Read clusters.txt to get mood labels
      clusters_file = os.path.join(base_path, "clusters.txt")
      
      # Read cluster labels from file
      try:
          with open(clusters_file, 'r') as f:
              cluster_names = [line.strip() for line in f.readlines() if line.strip()]
      except Exception as e:
          raise Exception(f"Error reading clusters file: {e}")
      
      # Process audio files
      audio_dir = os.path.join(base_path, "Audio")
      if not os.path.exists(audio_dir):
          raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
      
      # Get all audio files (supporting multiple formats)
      audio_files = [f for f in os.listdir(audio_dir) ]
      
      # Sort files to ensure consistent order with labels
      audio_files.sort()
      
      if not audio_files:
          raise FileNotFoundError(f"No audio files found in {audio_dir}")
      
      print(f"Found {len(audio_files)} audio files, processing...")
      
      if len(audio_files) != len(cluster_names):
          print(f"Warning: Number of audio files ({len(audio_files)}) doesn't match number of labels ({len(cluster_names)})")
          print(f"Will process only the first {min(len(audio_files), len(cluster_names))} files")
      
      # Create a mapping from cluster names to indices
      unique_clusters = sorted(set(cluster_names))
      label_to_idx = {cluster: idx for idx, cluster in enumerate(unique_clusters)}
      
      # Process each file
      embeddings_list = []
      labels_list = []
      filenames_processed = []
      
      num_files_to_process = min(len(audio_files), len(cluster_names))
      
      for i in tqdm(range(num_files_to_process), desc="Extracting embeddings"):
          audio_file = audio_files[i]
          cluster_label = cluster_names[i]
          file_path = os.path.join(audio_dir, audio_file)
          
          try:
              # Load and extract embedding
              audio_array, sr = self.load_audio(file_path)
              embedding = self.extract_embedding(audio_array, sr)
              
              # Get label index
              label_idx = label_to_idx[cluster_label]
              
              # Save embedding and label
              embeddings_list.append(embedding)
              labels_list.append(label_idx)
              filenames_processed.append(audio_file)
              
          except Exception as e:
              print(f"Error processing {file_path}: {e}")
      
      # Convert lists to tensors
      if not embeddings_list:
          raise ValueError("No embeddings were successfully extracted")
          
      embeddings = torch.cat(embeddings_list, dim=0)
      labels = torch.tensor(labels_list, dtype=torch.long)
      
      # Save to cache
      with open(cache_file, 'wb') as f:
          pickle.dump({
              'embeddings': embeddings,
              'labels': labels,
              'label_to_idx': label_to_idx
          }, f)
    

      print(f"MoodsMIREX embeddings saved to cache: {cache_file}")
      print(f"Embeddings dimensions: {embeddings.shape}")
      print(f"Processed clusters: {len(label_to_idx)}")
      
      return embeddings, labels
    
    def get_embeddings_AUDIO_MNIST(self, base_path="data/MNIST_AUDIO", force_recalculate=False):
        """
        Process all audio files in the MNIST_AUDIO dataset and return embeddings and labels.

        Args:
            base_path: Path to the MNIST_AUDIO data folder (containing subfolders for each person).
            force_recalculate: If True, recalculate embeddings even if cache file exists.

        Returns:
            embeddings: Tensor with all embeddings (n_samples, embedding_dim).
            labels: Tensor with all labels (n_samples).
            person_ids: Tensor with all person IDs (n_samples).
        """
        # Create cache filename based on model
        cache_file = self.cache_dir / f"mnist_audio_embeddings_{self.model_name.replace('/', '_')}.pkl"

        # Check if cache exists and load it if recalculation is not forced
        if cache_file.exists() and not force_recalculate:
            print(f"Loading MNIST_AUDIO embeddings from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                return cached_data['embeddings'], cached_data['labels'], cached_data['person_ids']

        # If cache doesn't exist or recalculation is forced, process files
        print(f"Processing MNIST_AUDIO audio files in {base_path}...")

        embeddings_list = []
        labels_list = []
        person_ids_list = []

        # Process each person's folder
        persons = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        persons.sort()  # Ensure consistent order

        for person in persons:
            person_path = os.path.join(base_path, person)
            print(f"Processing person: {person}")

            # Get all WAV files in the person's folder
            audio_files = [f for f in os.listdir(person_path) if f.endswith('.wav')]

            for audio_file in tqdm(audio_files, desc=f"Processing files for person {person}"):
                file_path = os.path.join(person_path, audio_file)

                try:
                    # Extract label from the filename (e.g., "0_03_1.wav" -> label = 0)
                    label = int(audio_file.split('_')[0])

                    # Load and extract embedding
                    audio_array, sr = self.load_audio(file_path)
                    embedding = self.extract_embedding(audio_array, sr)

                    # Save embedding, label, and person ID
                    embeddings_list.append(embedding)
                    labels_list.append(label)
                    person_ids_list.append(int(person))

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

        # Convert lists to tensors
        embeddings = torch.cat(embeddings_list, dim=0)
        labels = torch.tensor(labels_list, dtype=torch.long)
        person_ids = torch.tensor(person_ids_list, dtype=torch.long)

        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'embeddings': embeddings,
                'labels': labels,
                'person_ids': person_ids
            }, f)

        print(f"MNIST_AUDIO embeddings saved to cache: {cache_file}")
        print(f"Embeddings dimensions: {embeddings.shape}")
        print(f"Processed persons: {len(persons)}")

        return embeddings, labels
    
    def get_embedding_for_file(self, file_path):
        """
        Get embedding for an individual audio file
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            embedding: Audio embedding
        """
        audio_array, sr = self.load_audio(file_path)
        embedding = self.extract_embedding(audio_array, sr)
        return embedding
    

if __name__ == "__main__":
    # Example usage
    embedder = Embedder()
    
    # Get embeddings for GTZAN dataset
    # embeddings, labels, label_to_idx= embedder.get_embeddings_GTZAN(base_path="data/GTZAN", force_recalculate=True)
    # embeddings, labels, label_to_idx = embedder.get_embeddings_GTZAN(force_recalculate= False)
    # breakpoint()
    
    # Get embedding for a single file
    file_path = "data/GTZAN/blues/blues.00000.wav"  # Replace with your audio file path
    embedding = embedder.get_embedding_for_file(file_path)
    
    print(f"Embedding shape for {file_path}: {embedding.shape}")

    # Get embeddings AUDIO MNIST dataset
    embeddings, labels = embedder.get_embeddings_MNIST_AUDIO(base_path="data/AUDIO_MNIST", force_recalculate=False)