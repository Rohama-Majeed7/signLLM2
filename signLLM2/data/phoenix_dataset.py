"""
Phoenix-2014T Dataset Loader - Fixed annotation loading
"""
import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd
import gzip

class PhoenixDataset(Dataset):
    def __init__(self, data_root, split='train', transform=None, config=None, max_samples=None):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.config = config
        
        # Paths
        self.videos_dir = os.path.join(config.videos_dir, split) if config else \
                         os.path.join(data_root, "videos_phoenix/videos", split)
        
        # Load video-text pairs
        self.video_files, self.texts = self._load_data(max_samples)
        
        print(f"✅ Phoenix-2014T {split} dataset loaded: {len(self)} videos")
    
    def _load_annotations_binary(self, anno_path):
        """Load annotations from binary gzip file"""
        try:
            # Read binary file
            with gzip.open(anno_path, 'rb') as f:
                content = f.read()
            
            # Try to decode with different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    text = content.decode(encoding)
                    lines = text.strip().split('\n')
                    
                    data = []
                    for line in lines:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            name = parts[0]
                            translation = parts[1]
                            data.append({'name': name, 'translation': translation})
                    
                    print(f"Successfully decoded with {encoding}")
                    return pd.DataFrame(data)
                    
                except UnicodeDecodeError:
                    continue
            
            print(f"Could not decode with any encoding, trying raw parsing...")
            
            # If all encodings fail, try raw parsing
            lines = content.split(b'\n')
            data = []
            for line in lines:
                if line:
                    parts = line.split(b'\t')
                    if len(parts) >= 2:
                        try:
                            name = parts[0].decode('utf-8', errors='ignore')
                            translation = parts[1].decode('utf-8', errors='ignore')
                            data.append({'name': name, 'translation': translation})
                        except:
                            continue
            
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"Error loading annotations from {anno_path}: {e}")
            return None
    
    def _load_data(self, max_samples=None):
        """Load video files and texts"""
        video_files = []
        texts = []
        
        # Try to load annotations first
        annotation_files = {
            'train': 'phoenix14t.pami0.train.annotations_only.gzip',
            'dev': 'phoenix14t.pami0.dev.annotations_only.gzip',
            'test': 'phoenix14t.pami0.test.annotations_only.gzip'
        }
        
        if self.split in annotation_files:
            anno_path = os.path.join(self.config.data_root, annotation_files[self.split])
            
            if os.path.exists(anno_path):
                print(f"Loading annotations from: {anno_path}")
                annotations = self._load_annotations_binary(anno_path)
                
                if annotations is not None and not annotations.empty:
                    print(f"Loaded {len(annotations)} annotations")
                    
                    for idx, row in annotations.iterrows():
                        if max_samples and len(video_files) >= max_samples:
                            break
                        
                        video_name = row['name']
                        # Try different video extensions and naming patterns
                        video_patterns = [
                            f"{video_name}.mp4",
                            f"{video_name}.avi",
                            video_name  # Sometimes the name includes extension
                        ]
                        
                        for pattern in video_patterns:
                            video_path = os.path.join(self.videos_dir, pattern)
                            if os.path.exists(video_path):
                                video_files.append(video_path)
                                texts.append(row['translation'])
                                break
                        else:
                            # Try to find any file containing the video name
                            if os.path.exists(self.videos_dir):
                                for file in os.listdir(self.videos_dir):
                                    if video_name in file and file.endswith('.mp4'):
                                        video_path = os.path.join(self.videos_dir, file)
                                        video_files.append(video_path)
                                        texts.append(row['translation'])
                                        break
        
        # If no annotations loaded or not enough videos, load directly
        if len(video_files) == 0:
            print("Loading videos directly from directory...")
            if os.path.exists(self.videos_dir):
                all_videos = []
                for file in os.listdir(self.videos_dir):
                    if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        all_videos.append(file)
                
                # Sort for reproducibility
                all_videos.sort()
                
                # Limit samples if specified
                if max_samples:
                    all_videos = all_videos[:max_samples]
                
                for video_file in all_videos:
                    video_path = os.path.join(self.videos_dir, video_file)
                    video_files.append(video_path)
                    # Create descriptive text from filename
                    text = f"Sign language video: {os.path.splitext(video_file)[0]}"
                    texts.append(text)
        
        return video_files, texts
    
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
            
            if total_frames == 0 or fps == 0:
                print(f"Warning: Video {video_path} has 0 frames or 0 fps")
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
                    if self.config:
                        frame = cv2.resize(frame, self.config.img_size)
                    else:
                        frame = cv2.resize(frame, (112, 112))
                    
                    # Convert to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Normalize
                    frame = frame.astype(np.float32) / 255.0
                    frames.append(frame)
                
                frame_count += 1
            
            cap.release()
            
            # Handle insufficient frames
            if len(frames) < target_frames:
                if frames:  # Pad with last frame
                    last_frame = frames[-1]
                    while len(frames) < target_frames:
                        frames.append(last_frame.copy())
                else:  # Create dummy frames
                    return self._create_dummy_video()
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
            C, T, H, W = 3, 32, 112, 112
        
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