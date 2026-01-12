"""
Complete Configuration for Phoenix-2014T I3D Features
"""
import torch
import os

class Config:
    # ==================== DEVICE ====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ==================== DATASET PATHS ====================
    data_root = "/kaggle/input/rwth-phoenix-2014t-i3d-features-mediapipe-features"
    
    # I3D Features
    i3d_features_dir = os.path.join(data_root, "i3d_features_rwth phoenix 2014t/i3d_features_rwth phoenix 2014t")
    
    # MediaPipe Features (optional)
    mediapipe_features_dir = os.path.join(data_root, "mediapipe_features_rwth phoenix 2014t/mediapipe_features")
    
    # Annotations
    annotations_dir = os.path.join(data_root, "tsv files_rwth phoenix 2014t/tsv files")
    
    # ==================== FEATURE SETTINGS ====================
    use_i3d_features = True  # Set to False to use MediaPipe features
    i3d_feature_dim = 1024   # I3D features dimension
    mediapipe_feature_dim = 1404  # MediaPipe features dimension
    
    # Fixed sequence length (for padding/truncation)
    fixed_sequence_length = 150
    
    # ==================== MODEL PARAMETERS ====================
    codebook_size = 256      # Size of character-level codebook
    codebook_dim = 512       # Dimension of codebook embeddings
    
    # ==================== TRAINING PARAMETERS ====================
    batch_size = 8          # Batch size
    learning_rate = 0.001   # Learning rate
    num_epochs = 5         # Number of epochs
    lambda_mmd = 0.5        # Weight for MMD loss
    lambda_sim = 1.0        # Weight for similarity loss
    
    # ==================== DATASET SPLITS ====================
    train_split = 'train'
    val_split = 'val'
    test_split = 'test'
    
    # ==================== TRAINING MODE ====================
    training_mode = "quick"  # Options: "test", "quick", "full"
    
    # ==================== INITIALIZATION ====================
    def __init__(self, **kwargs):
        # Override defaults with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # ===== DERIVED ATTRIBUTES =====
        # Feature settings
        if self.use_i3d_features:
            self.feature_dim = self.i3d_feature_dim
            self.features_dir = self.i3d_features_dir
        else:
            self.feature_dim = self.mediapipe_feature_dim
            self.features_dir = self.mediapipe_features_dir
        
        # ===== TRAINING MODE SETTINGS =====
        if self.training_mode == "test":
            self.num_epochs = 1
            self.max_train_samples = 20
            self.max_val_samples = 10
            self.max_test_samples = 10
        elif self.training_mode == "quick":
            self.num_epochs = 3
            self.max_train_samples = 100
            self.max_val_samples = 50
            self.max_test_samples = 50
        else:  # "full"
            self.num_epochs = 20
            self.max_train_samples = None  # Use all
            self.max_val_samples = None
            self.max_test_samples = None
        
        # ===== CPU OPTIMIZATION =====
        if self.device.type == 'cpu':
            print("⚠️ Training on CPU - adjusting parameters")
            self.batch_size = 4
            self.fixed_sequence_length = 100
            if self.training_mode == "full":
                self.training_mode = "quick"
        
        # ===== VALIDATION =====
        self._validate_paths()
        
        # ===== PRINT CONFIG =====
        self.print_config()
    
    def _validate_paths(self):
        """Validate that all required paths exist"""
        print("\n📁 Validating dataset paths:")
        
        paths_to_check = [
            (self.features_dir, "Features directory"),
            (os.path.join(self.features_dir, self.train_split), f"Train split ({self.train_split})"),
            (os.path.join(self.features_dir, self.val_split), f"Val split ({self.val_split})"),
            (self.annotations_dir, "Annotations directory")
        ]
        
        for path, description in paths_to_check:
            if os.path.exists(path):
                print(f"  ✅ {description}: {path}")
            else:
                print(f"  ⚠️  {description} not found: {path}")
    
    def print_config(self):
        """Print configuration summary"""
        print("\n" + "=" * 60)
        print("CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"  Device: {self.device}")
        print(f"  Training mode: {self.training_mode}")
        print(f"  Features: {'I3D' if self.use_i3d_features else 'MediaPipe'}")
        print(f"  Feature dimension: {self.feature_dim}")
        print(f"  Fixed sequence length: {self.fixed_sequence_length}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Max train samples: {self.max_train_samples or 'All'}")
        print("=" * 60)

# Create default config instance
config = Config()