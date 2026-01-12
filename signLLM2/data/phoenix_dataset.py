"""
Phoenix-2014T I3D Features Dataset Loader - Fixed padding
"""
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random
from torch.nn.utils.rnn import pad_sequence

class PhoenixFeaturesDataset(Dataset):
    def __init__(self, data_root, split='train', config=None, max_samples=None):
        """
        Args:
            data_root: Root directory of dataset
            split: Dataset split ('train', 'val', 'test')
            config: Configuration object
            max_samples: Maximum number of samples to load
        """
        self.split = split
        self.config = config
        self.max_samples = max_samples
        self.max_seq_length = 100  # Maximum sequence length for padding
        
        # Set feature directory
        self.features_dir = os.path.join(config.features_dir, split)
        
        # Load annotations if available
        self.annotations = self._load_annotations()
        
        # Load features
        self.features, self.texts, self.filenames, self.original_lengths = self._load_features()
        
        print(f"✅ Phoenix-2014T Features {split}: {len(self)} samples loaded")
        print(f"   Max sequence length: {self.max_seq_length}")
        print(f"   Feature dimension: {self.config.feature_dim}")
    
    def _load_annotations(self):
        """Load annotations from TSV files"""
        annotation_files = {
            'train': 'cvpr23.fairseq.i3d.train.how2sign.tsv',
            'val': 'cvpr23.fairseq.i3d.val.how2sign.tsv',
            'test': 'cvpr23.fairseq.i3d.test.how2sign.tsv'
        }
        
        if self.split in annotation_files:
            anno_path = os.path.join(self.config.annotations_dir, annotation_files[self.split])
            
            if os.path.exists(anno_path):
                try:
                    print(f"Loading annotations from: {anno_path}")
                    # TSV files with no header, tab-separated
                    df = pd.read_csv(anno_path, sep='\t', header=None, on_bad_lines='skip')
                    
                    # Assuming format: video_id \t text
                    if len(df.columns) >= 2:
                        df = df.iloc[:, :2]  # Take first two columns
                        df.columns = ['video_id', 'text']
                        print(f"Loaded {len(df)} annotations")
                        return df
                except Exception as e:
                    print(f"Error loading annotations: {e}")
        
        return None
    
    def _load_features(self):
        """Load pre-extracted I3D features with padding"""
        if not os.path.exists(self.features_dir):
            print(f"Error: Features directory not found: {self.features_dir}")
            return [], [], [], []
        
        # Get all .npy files
        all_files = [f for f in os.listdir(self.features_dir) if f.endswith('.npy')]
        
        if not all_files:
            print(f"Warning: No .npy files found in {self.features_dir}")
            return [], [], [], []
        
        # Sort for reproducibility
        all_files.sort()
        
        # Limit samples if specified
        if self.max_samples and len(all_files) > self.max_samples:
            indices = list(range(len(all_files)))
            random.seed(42)
            selected_indices = random.sample(indices, self.max_samples)
            all_files = [all_files[i] for i in selected_indices]
        
        print(f"Found {len(all_files)} feature files in {self.split} split")
        
        features_list = []
        texts_list = []
        filenames_list = []
        lengths_list = []
        
        # Find max length for padding
        max_len = 0
        sample_lengths = []
        
        for i, filename in enumerate(all_files):
            if i % 50 == 0:
                print(f"  Scanning file {i+1}/{len(all_files)}...")
            
            file_path = os.path.join(self.features_dir, filename)
            
            try:
                # Load numpy array
                feature_array = np.load(file_path, allow_pickle=True)
                
                if isinstance(feature_array, np.ndarray):
                    seq_length = feature_array.shape[0]
                    sample_lengths.append(seq_length)
                    max_len = max(max_len, seq_length)
                    
            except Exception as e:
                continue
        
        # Calculate statistics
        if sample_lengths:
            print(f"Sequence length statistics:")
            print(f"  Min: {min(sample_lengths)}")
            print(f"  Max: {max(sample_lengths)}")
            print(f"  Mean: {np.mean(sample_lengths):.1f}")
            print(f"  Median: {np.median(sample_lengths)}")
        
        # Set max sequence length (clip at 200 for memory)
        self.max_seq_length = min(max_len, 200)
        print(f"Using max sequence length: {self.max_seq_length}")
        
        # Now load and pad features
        for i, filename in enumerate(all_files):
            if i % 50 == 0:
                print(f"  Loading file {i+1}/{len(all_files)}...")
            
            file_path = os.path.join(self.features_dir, filename)
            
            try:
                # Load numpy array
                feature_array = np.load(file_path, allow_pickle=True)
                
                # Convert to tensor
                if isinstance(feature_array, np.ndarray):
                    feature_tensor = torch.from_numpy(feature_array).float()
                    
                    # Ensure feature dimension matches config
                    if feature_tensor.shape[-1] != self.config.feature_dim:
                        print(f"Warning: Feature {filename} has dimension {feature_tensor.shape[-1]}, "
                              f"expected {self.config.feature_dim}")
                        continue
                    
                    # Pad or truncate sequence
                    seq_length = feature_tensor.shape[0]
                    
                    if seq_length > self.max_seq_length:
                        # Truncate: take middle part
                        start = (seq_length - self.max_seq_length) // 2
                        feature_tensor = feature_tensor[start:start + self.max_seq_length]
                    elif seq_length < self.max_seq_length:
                        # Pad with zeros
                        padding = self.max_seq_length - seq_length
                        feature_tensor = torch.cat([
                            feature_tensor,
                            torch.zeros(padding, self.config.feature_dim)
                        ], dim=0)
                    
                    # Get corresponding text
                    text = self._get_text_for_file(filename)
                    
                    features_list.append(feature_tensor)
                    texts_list.append(text)
                    filenames_list.append(filename)
                    lengths_list.append(min(seq_length, self.max_seq_length))
                    
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
        
        return features_list, texts_list, filenames_list, lengths_list
    
    def _get_text_for_file(self, filename):
        """Get text annotation for a feature file"""
        if self.annotations is None:
            # Use filename as text
            base_name = os.path.splitext(filename)[0]
            return f"Sign: {base_name}"
        
        # Remove .npy extension
        video_id = os.path.splitext(filename)[0]
        
        # Try to find in annotations
        if 'video_id' in self.annotations.columns:
            matches = self.annotations[self.annotations['video_id'] == video_id]
            if not matches.empty:
                return matches.iloc[0]['text']
        
        # Try partial match
        for _, row in self.annotations.iterrows():
            if 'video_id' in row and isinstance(row['video_id'], str):
                if video_id in row['video_id'] or row['video_id'] in video_id:
                    return row['text']
        
        # Fallback
        return f"Sign: {video_id}"
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        text = self.texts[idx]
        filename = self.filenames[idx]
        seq_length = self.original_lengths[idx]
        
        # Feature should already be padded to max_seq_length
        # shape: (max_seq_length, feature_dim)
        
        return {
            'feature': feature,
            'text': text,
            'filename': filename,
            'seq_length': seq_length
        }

# Custom collate function for variable length sequences
def features_collate_fn(batch):
    """Custom collate function for features"""
    features = torch.stack([item['feature'] for item in batch])
    texts = [item['text'] for item in batch]
    filenames = [item['filename'] for item in batch]
    seq_lengths = torch.tensor([item['seq_length'] for item in batch])
    
    return {
        'feature': features,
        'text': texts,
        'filename': filenames,
        'seq_length': seq_lengths
    }