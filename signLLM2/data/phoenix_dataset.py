"""
Phoenix-2014T Dataset Loader for the specific dataset structure
"""
import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd
import gzip

class PhoenixDataset(Dataset):
    def __init__(self, data_root, split='train', transform=None, config=None):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.config = config
        
        # Paths
        self.videos_dir = os.path.join(config.videos_dir, split) if config else \
                         os.path.join(data_root, "videos_phoenix/videos", split)
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Get video files
        self.video_files = []
        self.texts = []
        
        # Load from annotations
        if self.annotations is not None:
            print(f"Loading {len(self.annotations)} samples from annotations...")
            for idx, row in self.annotations.iterrows():
                video_file = f"{row['name']}.mp4"
                video_path = os.path.join(self.videos_dir, video_file)
                
                if os.path.exists(video_path):
                    self.video_files.append(video_path)
                    self.texts.append(row['translation'])
                else:
                    # Try alternative naming
                    alt_files = [f for f in os.listdir(self.videos_dir) 
                                if row['name'] in f and f.endswith('.mp4')]
                    if alt_files:
                        video_path = os.path.join(self.videos_dir, alt_files[0])
                        self.video_files.append(video_path)
                        self.texts.append(row['translation'])
        
        # If no annotations loaded, use all videos in directory
        if len(self.video_files) == 0:
            print("Loading videos directly from directory...")
            for video_file in os.listdir(self.videos_dir):
                if video_file.endswith('.mp4'):
                    self.video_files.append(os.path.join(self.videos_dir, video_file))
                    self.texts.append(f"Video: {video_file}")
        
        print(f"✅ Phoenix-2014T {split} dataset loaded: {len(self)} videos")
    
    def _load_annotations(self):
        """Load annotations from gzip files"""
        annotation_files = {
            'train': 'phoenix14t.pami0.train.annotations_only.gzip',
            'dev': 'phoenix14t.pami0.dev.annotations_only.gzip',
            'test': 'phoenix14t.pami0.test.annotations_only.gzip'
        }
        
        if self.split in annotation_files:
            anno_path = os.path.join(self.config.annotations_dir, annotation_files[self.split])
            
            if os.path.exists(anno_path):
                try:
                    print(f"Loading annotations from: {anno_path}")
                    
                    # Read gzip file
                    with gzip.open(anno_path, 'rt', encoding='utf-8') as f:
                        # Phoenix annotations are tab-separated
                        lines = f.readlines()
                        
                    # Parse annotations
                    data = []
                    for line in lines:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            name = parts[0]
                            translation = parts[1]
                            data.append({'name': name, 'translation': translation})
                    
                    return pd.DataFrame(data)
                    
                except Exception as e:
                    print(f"Error loading annotations: {e}")
        
        return None
    
    def _load_video(self, video_path):
        """Load and preprocess video"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video {video_path}")
                return self._create_dummy_video()
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                print(f"Warning: Video {video_path} has 0 frames")
                cap.release()
                return self._create_dummy_video()
            
            # Calculate frame sampling
            target_frames = self.config.video_frames if self.config else 32
            frame_interval = max(1, total_frames // target_frames)
            
            frames = []
            frame_count = 0
            
            while len(frames) < target_frames and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Resize
                    frame = cv2.resize(frame, self.config.img_size)
                    # Convert to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Normalize
                    frame = frame.astype(np.float32) / 255.0
                    frames.append(frame)
                
                frame_count += 1
            
            cap.release()
            
            # Handle insufficient frames
            if len(frames) < target_frames:
                # Repeat last frame
                last_frame = frames[-1] if frames else np.zeros((*self.config.img_size, 3), dtype=np.float32)
                while len(frames) < target_frames:
                    frames.append(last_frame.copy())
            elif len(frames) > target_frames:
                frames = frames[:target_frames]
            
            # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
            video_tensor = torch.from_numpy(np.array(frames)).permute(3, 0, 1, 2).float()
            
            return video_tensor
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return self._create_dummy_video()
    
    def _create_dummy_video(self):
        """Create dummy video for testing"""
        if self.config:
            C, T, H, W = 3, self.config.video_frames, self.config.img_size[0], self.config.img_size[1]
        else:
            C, T, H, W = 3, 32, 224, 224
        
        return torch.randn(C, T, H, W)
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        text = self.texts[idx]
        
        # Load video
        video = self._load_video(video_path)
        
        if self.transform:
            video = self.transform(video)
        
        return {
            'video': video,
            'text': text,
            'video_path': os.path.basename(video_path)
        }