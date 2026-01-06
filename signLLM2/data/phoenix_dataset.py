"""
Phoenix-2014T Dataset Loader for Kaggle
"""
import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class PhoenixDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, config=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.config = config
        
        # Load annotations
        anno_file = os.path.join(data_dir, f'annotations/{split}.corpus.csv')
        self.annotations = self._load_annotations(anno_file)
        
        # Get video paths
        self.video_paths = []
        self.texts = []
        
        for idx, row in self.annotations.iterrows():
            video_path = os.path.join(data_dir, 'features/fullFrame-210x260px', 
                                     f'{split}/{row["video"]}.mp4')
            if os.path.exists(video_path):
                self.video_paths.append(video_path)
                self.texts.append(row['translation'])
    
    def _load_annotations(self, anno_file):
        """Load Phoenix dataset annotations"""
        import pandas as pd
        df = pd.read_csv(anno_file, delimiter='|')
        df.columns = [col.strip() for col in df.columns]
        return df
    
    def _load_video(self, video_path):
        """Load and preprocess video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize and convert to RGB
            frame = cv2.resize(frame, self.config.img_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        # Convert to tensor and normalize
        frames = np.array(frames)
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)
        
        # Pad or trim to fixed length
        if len(frames) < self.config.video_frames:
            padding = self.config.video_frames - len(frames)
            frames = torch.cat([frames, frames[-1:].repeat(padding, 1, 1, 1)], dim=0)
        else:
            frames = frames[:self.config.video_frames]
        
        return frames
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        text = self.texts[idx]
        
        # Load video
        video = self._load_video(video_path)
        
        if self.transform:
            video = self.transform(video)
        
        return {
            'video': video,  # (T, C, H, W)
            'text': text,
            'video_path': video_path
        }