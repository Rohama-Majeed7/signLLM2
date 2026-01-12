"""
Phoenix-2014T I3D Features Dataset Loader
Loads pre-extracted I3D features instead of raw videos
"""
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random

class PhoenixFeaturesDataset(Dataset):
    def __init__(self, data_root, split='train', config=None, max_samples=None):
        self.split = split
        self.config = config
        self.max_samples = max_samples
        
        # Set feature paths based on config
        if config.use_i3d_features:
            self.features_dir = os.path.join(config.i3d_features_dir, split)
            self.feature_dim = config.i3d_feature_dim
            print(f"Using I3D features from: {self.features_dir}")
        else:
            self.features_dir = os.path.join(config.mediapipe_features_dir, split)
            self.feature_dim = config.mediapipe_feature_dim
            print(f"Using MediaPipe features from: {self.features_dir}")
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Load features and texts
        self.features, self.texts, self.feature_files = self._load_features()
        
        print(f"✅ Phoenix-2014T Features {split} dataset loaded: {len(self)} samples")
    
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
                    # TSV files are tab-separated
                    df = pd.read_csv(anno_path, sep='\t', header=None)
                    
                    # Assuming format: video_id \t text
                    if len(df.columns) >= 2:
                        df.columns = ['video_id', 'text']
                    elif len(df.columns) == 1:
                        df.columns = ['text']
                        # Extract video_id from text or create one
                        df['video_id'] = df.index.astype(str)
                    
                    print(f"Loaded {len(df)} annotations")
                    return df
                    
                except Exception as e:
                    print(f"Error loading annotations: {e}")
        
        return None
    
    def _load_features(self):
        """Load pre-extracted features"""
        if not os.path.exists(self.features_dir):
            print(f"Error: Features directory not found: {self.features_dir}")
            return [], [], []
        
        # Get all .npy files
        feature_files = []
        for file in os.listdir(self.features_dir):
            if file.endswith('.npy'):
                feature_files.append(file)
        
        # Sort for reproducibility
        feature_files.sort()
        
        print(f"Found {len(feature_files)} feature files in {self.features_dir}")
        
        # Limit samples if specified
        if self.max_samples and len(feature_files) > self.max_samples:
            indices = list(range(len(feature_files)))
            random.seed(42)
            selected_indices = random.sample(indices, self.max_samples)
            feature_files = [feature_files[i] for i in selected_indices]
        
        # Load features and find corresponding texts
        features_list = []
        texts_list = []
        files_list = []
        
        for feature_file in feature_files:
            feature_path = os.path.join(self.features_dir, feature_file)
            
            try:
                # Load numpy array
                feature_array = np.load(feature_path)
                
                # Convert to tensor
                if isinstance(feature_array, np.ndarray):
                    # Handle different feature shapes
                    if feature_array.ndim == 1:
                        # Single feature vector
                        feature_tensor = torch.from_numpy(feature_array).float()
                    elif feature_array.ndim == 2:
                        # Temporal sequence of features
                        # Take mean or use all frames
                        feature_tensor = torch.from_numpy(feature_array).float()
                    else:
                        print(f"Unexpected feature shape in {feature_file}: {feature_array.shape}")
                        continue
                else:
                    print(f"Unexpected data type in {feature_file}")
                    continue
                
                # Get text from annotations
                text = self._get_text_from_annotations(feature_file)
                
                features_list.append(feature_tensor)
                texts_list.append(text)
                files_list.append(feature_file)
                
            except Exception as e:
                print(f"Error loading feature file {feature_file}: {e}")
                continue
        
        return features_list, texts_list, files_list
    
    def _get_text_from_annotations(self, feature_file):
        """Get text for feature file from annotations"""
        if self.annotations is None:
            # Use filename as text
            base_name = os.path.splitext(feature_file)[0]
            return f"Sign language: {base_name}"
        
        # Remove .npy extension
        video_id = os.path.splitext(feature_file)[0]
        
        # Try exact match with video_id column
        if 'video_id' in self.annotations.columns:
            matches = self.annotations[self.annotations['video_id'] == video_id]
            if not matches.empty:
                return matches.iloc[0]['text']
        
        # Try partial match
        for _, row in self.annotations.iterrows():
            # Check if video_id is in text or vice versa
            if 'video_id' in row and isinstance(row['video_id'], str):
                if video_id in row['video_id'] or row['video_id'] in video_id:
                    return row['text']
            elif 'text' in row:
                if video_id in row['text']:
                    return row['text']
        
        # Use filename as fallback
        return f"Sign language: {video_id}"
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        text = self.texts[idx]
        feature_file = self.feature_files[idx]
        
        # Ensure feature has correct shape for model
        # Model expects: (batch, channels, temporal, height, width) for videos
        # But we have features, so we need to reshape
        
        if feature.dim() == 1:
            # Single vector: reshape to (1, 1, feature_dim)
            feature = feature.unsqueeze(0).unsqueeze(0)
        elif feature.dim() == 2:
            # Temporal features: reshape to (1, temporal, feature_dim)
            # Model expects (C, T, H, W), so we need to add channel dimension
            feature = feature.permute(1, 0)  # (feature_dim, temporal)
            feature = feature.unsqueeze(0)  # Add channel dimension
        
        return {
            'feature': feature,
            'text': text,
            'feature_file': feature_file
        }