"""
Configuration for Phoenix-2014T I3D Features Dataset
"""
import torch
import os

class Config:
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset paths
    data_root = "/kaggle/input/rwth-phoenix-2014t-i3d-features-mediapipe-features"
    
    # Features paths
    i3d_features_dir = os.path.join(data_root, "i3d_features_rwth phoenix 2014t/i3d_features_rwth phoenix 2014t")
    mediapipe_features_dir = os.path.join(data_root, "mediapipe_features_rwth phoenix 2014t/mediapipe_features")
    annotations_dir = os.path.join(data_root, "tsv files_rwth phoenix 2014t/tsv files")
    
    # Which features to use
    use_i3d_features = True
    use_mediapipe_features = False  # Set to True if you want to use both
    
    # Feature dimensions
    i3d_feature_dim = 1024  # I3D features are typically 1024-dimensional
    mediapipe_feature_dim = 1404  # MediaPipe features dimension
    
    # Model parameters
    codebook_size = 256
    codebook_dim = 512
    
    # Training
    batch_size = 16  # Can be larger since features are smaller than videos
    learning_rate = 0.001
    num_epochs = 5
    lambda_mmd = 0.5
    lambda_sim = 1.0
    
    # Dataset splits
    train_split = 'train'
    val_split = 'val'
    test_split = 'test'
    
    # Training mode
    training_mode = "quick"  # "quick", "full", or "test"
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Set parameters based on training mode
        if self.training_mode == "quick":
            self.num_epochs = 3
            self.max_train_samples = 100
            self.max_val_samples = 50
            self.max_test_samples = 50
        elif self.training_mode == "test":
            self.num_epochs = 1
            self.max_train_samples = 20
            self.max_val_samples = 10
            self.max_test_samples = 10
        else:  # "full"
            self.num_epochs = 20
            self.max_train_samples = None  # Use all
            self.max_val_samples = None
            self.max_test_samples = None
        
        # Auto-detect feature dimension
        self.feature_dim = self.i3d_feature_dim if self.use_i3d_features else self.mediapipe_feature_dim
        
        # Verify paths
        self._verify_paths()
    
    def _verify_paths(self):
        """Verify all dataset paths exist"""
        paths_to_check = [
            (self.i3d_features_dir, "I3D features"),
            (self.annotations_dir, "Annotations")
        ]
        
        print("📁 Verifying dataset paths:")
        for path, name in paths_to_check:
            if os.path.exists(path):
                print(f"  ✅ {name}: {path}")
            else:
                print(f"  ❌ {name} not found: {path}")