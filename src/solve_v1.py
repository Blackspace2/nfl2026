# copy form https://www.kaggle.com/code/muhammadqasimshabbir/nfl-big-data-bowl-2026-prediction/notebook

import numpy as np
import pandas as pd
import warnings
import gc  # 导入垃圾回收模块
from pathlib import Path
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter1d

# Machine Learning
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# ================================================================================
# GPU REQUIREMENTS CHECK
# ================================================================================

def check_gpu_requirements():
    """Check if all required GPU libraries are available"""
    gpu_available = torch.cuda.is_available()
    
    if gpu_available:
        print("✅ CUDA is available")
        try:
            import xgboost
            print("✅ XGBoost GPU support available")
        except ImportError:
            print("⚠️  XGBoost not found - install with: pip install xgboost")
        
        try:
            import lightgbm
            print("✅ LightGBM GPU support available")
        except ImportError:
            print("⚠️  LightGBM not found - install with: pip install lightgbm")
        
        try:
            import catboost
            print("✅ CatBoost GPU support available")
        except ImportError:
            print("⚠️  CatBoost not found - install with: pip install catboost")
    else:
        print("⚠️  CUDA not available - will use CPU for all models")
    
    return gpu_available

# ================================================================================
# CONFIGURATION
# ================================================================================

class Config:
    DATA_DIR = Path("../dataset/nfl-big-data-bowl-2026-prediction/")
    SEEDS = [42, 123, 2024]  # Multiple seeds for ensemble
    FIELD_X_MIN, FIELD_X_MAX = 0.0, 120.0
    FIELD_Y_MIN, FIELD_Y_MAX = 0.0, 53.3
    MAX_SPEED = 12.0
    N_FOLDS = 5
    NN_BATCH_SIZE = 2048
    NN_EPOCHS = 30
    NN_LEARNING_RATE = 0.001
    
    # GPU Configuration
    USE_GPU = True
    GPU_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    @classmethod
    def print_gpu_info(cls):
        """Print GPU information and availability"""
        print("="*60)
        print("GPU CONFIGURATION")
        print("="*60)
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"GPU Device: {cls.GPU_DEVICE}")
        print(f"GPU Count: {cls.GPU_COUNT}")
        
        if torch.cuda.is_available():
            for i in range(cls.GPU_COUNT):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("No GPU available - using CPU")
        print("="*60)
    
    @classmethod
    def cleanup_gpu_memory(cls):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

# ================================================================================
# ENHANCED NFL PLAYER MOVEMENT PREDICTOR
# ================================================================================

class EnhancedNFLPlayerMovementPredictor:
    """Enhanced NFL Player Movement Prediction with advanced temporal features"""
    
    def __init__(self, data_dir, seed=42):
        self.data_dir = Path(data_dir)
        self.seed = seed
        self.weeks = list(range(1, 18))
        self.models_dx = {} 
        self.models_dy = {}
        self.scalers = {} 
        self.label_encoders = {}
        self.nn_models_dx = []
        self.nn_models_dy = []
        
    def load_and_combine_datasets(self):
        """Load and combine weekly training data with progress tracking"""
        print("Loading datasets...")
        
        input_paths = [self.data_dir / f"train/input_2023_w{w:02d}.csv" for w in self.weeks]
        output_paths = [self.data_dir / f"train/output_2023_w{w:02d}.csv" for w in self.weeks]
        
        # Filter existing files
        input_paths = [p for p in input_paths if p.exists()]
        output_paths = [p for p in output_paths if p.exists()]
        
        print(f"Found {len(input_paths)} weeks of training data")
        
        train_input = self._load_multiple_csv_files(input_paths)
        train_output = self._load_multiple_csv_files(output_paths)
        
        test_input = pd.read_csv(self.data_dir / "test_input.csv")
        test_template = pd.read_csv(self.data_dir / "test.csv")
        
        print(f"Loaded {len(train_input):,} input records, {len(train_output):,} output records")
        
        return train_input, train_output, test_input, test_template
    
    def _load_multiple_csv_files(self, file_paths):
        """Load and concatenate multiple CSV files with progress tracking"""
        data_frames = []
        for p in tqdm(file_paths, desc="Loading files"):
            data_frames.append(pd.read_csv(p))
        return pd.concat(data_frames, ignore_index=True)
    
    def _convert_height_to_inches(self, height_str):
        """Convert height from 'ft-in' format to total inches"""
        if not isinstance(height_str, str) or '-' not in height_str:
            return 70  # Default height
        try:
            feet, inches = map(int, height_str.split('-'))
            return feet * 12 + inches
        except (ValueError, AttributeError):
            return 70
    
    def _extract_temporal_features(self, tracking_data):
        """Extract comprehensive temporal features from tracking data"""
        print("Extracting temporal features...")
        
        # Get last frame before throw
        last_frame = tracking_data.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id']) \
                                 .groupby(['game_id', 'play_id', 'nfl_id'], as_index=False).last()
        last_frame = last_frame.rename(columns={'x': 'final_pre_throw_x', 'y': 'final_pre_throw_y'})
        
        # Calculate temporal statistics from all frames before throw
        temporal_stats = tracking_data.groupby(['game_id', 'play_id', 'nfl_id']).agg({
            'x': ['mean', 'std', 'min', 'max'],
            'y': ['mean', 'std', 'min', 'max'],
            's': ['mean', 'std', 'max', 'min'],
            'a': ['mean', 'std', 'max', 'min'],
            'dir': lambda x: np.std(np.diff(x)) if len(x) > 1 else 0,
            'o': lambda x: np.std(np.diff(x)) if len(x) > 1 else 0,
        }).reset_index()
        
        # Flatten column names
        temporal_stats.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                  for col in temporal_stats.columns.values]
        temporal_stats = temporal_stats.rename(columns={
            'dir_<lambda>': 'dir_change_rate',
            'o_<lambda>': 'orientation_change_rate'
        })
        
        # Get movement patterns from last N frames
        last_n_frames = 5
        recent_frames = tracking_data.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id']) \
                                    .groupby(['game_id', 'play_id', 'nfl_id']).tail(last_n_frames)
        
        # Calculate trajectory features from recent frames
        trajectory_features = recent_frames.groupby(['game_id', 'play_id', 'nfl_id']).agg({
            'x': lambda x: (x.iloc[-1] - x.iloc[0]) if len(x) > 1 else 0,
            'y': lambda x: (x.iloc[-1] - x.iloc[0]) if len(x) > 1 else 0,
            's': lambda x: x.diff().mean() if len(x) > 1 else 0,
        }).reset_index()
        trajectory_features.columns = ['game_id', 'play_id', 'nfl_id', 
                                      'recent_displacement_x', 'recent_displacement_y', 'acceleration_trend']
        
        # Merge temporal features
        last_frame = last_frame.merge(temporal_stats, on=['game_id', 'play_id', 'nfl_id'], how='left')
        last_frame = last_frame.merge(trajectory_features, on=['game_id', 'play_id', 'nfl_id'], how='left')
        
        # Convert height if available
        if 'player_height' in last_frame.columns:
            last_frame['height_inches'] = last_frame['player_height'].apply(self._convert_height_to_inches)
        
        return last_frame
    
    def _incorporate_target_receiver_data(self, player_data):
        """Add target receiver position data to all players in the same play"""
        if 'player_role' not in player_data.columns:
            print("Warning: 'player_role' column not found. Skipping target receiver incorporation.")
            player_data['target_receiver_x'] = np.nan
            player_data['target_receiver_y'] = np.nan
            return player_data
        
        target_receivers = player_data[player_data['player_role'] == "Targeted Receiver"][
            ['game_id', 'play_id', 'final_pre_throw_x', 'final_pre_throw_y']
        ].rename(columns={
            'final_pre_throw_x': 'target_receiver_x', 
            'final_pre_throw_y': 'target_receiver_y'
        })
        
        # Remove duplicates if any
        target_receivers = target_receivers.drop_duplicates(['game_id', 'play_id'])
        
        return player_data.merge(target_receivers, on=['game_id', 'play_id'], how='left')
    
    def _calculate_advanced_features(self, data_frame, training_mode=False):
        """Create comprehensive feature set with advanced temporal and physics features"""
        df = data_frame.copy()
        
        print("Calculating advanced features...")
        
        # ===== TEMPORAL FEATURES =====
        if 'frame_id' in df.columns:
            df['time_seconds'] = df['frame_id'] / 10.0  # 10 FPS
            df['time_normalized'] = df['frame_id'] / df.groupby(['game_id', 'play_id', 'nfl_id'])['frame_id'].transform('max')
            
            # Polynomial time features
            df['time_squared'] = df['time_seconds'] ** 2
            df['time_cubed'] = df['time_seconds'] ** 3
            df['sqrt_time'] = np.sqrt(df['time_seconds'])
            df['log_time'] = np.log1p(df['time_seconds'])
            
            # Fourier features for cyclical patterns
            df['time_sin'] = np.sin(2 * np.pi * df['time_normalized'])
            df['time_cos'] = np.cos(2 * np.pi * df['time_normalized'])
            df['time_sin_2'] = np.sin(4 * np.pi * df['time_normalized'])
            df['time_cos_2'] = np.cos(4 * np.pi * df['time_normalized'])
            
            # Phase-based features
            df['is_early_play'] = (df['time_normalized'] < 0.33).astype(int)
            df['is_mid_play'] = ((df['time_normalized'] >= 0.33) & (df['time_normalized'] < 0.67)).astype(int)
            df['is_late_play'] = (df['time_normalized'] >= 0.67).astype(int)
        
        # ===== VELOCITY AND PHYSICS FEATURES =====
        if all(col in df.columns for col in ['s', 'dir']):
            direction_radians = np.deg2rad(df['dir'].fillna(0))
            df['velocity_x'] = df['s'] * np.sin(direction_radians)
            df['velocity_y'] = df['s'] * np.cos(direction_radians)
            
            # Momentum features
            if 'player_weight' in df.columns:
                df['momentum_magnitude'] = df['player_weight'] * df['s']
            
            # Expected positions based on physics
            if 'time_seconds' in df.columns:
                df['expected_x_constant_v'] = df['final_pre_throw_x'] + df['velocity_x'] * df['time_seconds']
                df['expected_y_constant_v'] = df['final_pre_throw_y'] + df['velocity_y'] * df['time_seconds']
                
                if 'a' in df.columns:
                    df['expected_x_with_accel'] = df['final_pre_throw_x'] + df['velocity_x'] * df['time_seconds'] + \
                                                  0.5 * df['a'] * np.sin(direction_radians) * df['time_squared']
                    df['expected_y_with_accel'] = df['final_pre_throw_y'] + df['velocity_y'] * df['time_seconds'] + \
                                                  0.5 * df['a'] * np.cos(direction_radians) * df['time_squared']
        
        # ===== MOVEMENT CONSISTENCY FEATURES =====
        if 's_mean' in df.columns:
            df['speed_consistency'] = df['s'] / (df['s_mean'] + 0.1)
            df['speed_deviation'] = np.abs(df['s'] - df['s_mean'])
            
        if 'a_mean' in df.columns:
            df['acceleration_consistency'] = df['a'] / (df['a_mean'] + 0.1)
            df['acceleration_deviation'] = np.abs(df['a'] - df['a_mean'])
        
        # ===== TEMPORAL INTERACTION FEATURES =====
        if 'time_seconds' in df.columns:
            df['time_x_speed'] = df['time_seconds'] * df['s']
            df['time_x_acceleration'] = df['time_seconds'] * df['a']
            if 'time_squared' in df.columns:
                df['time_squared_x_speed'] = df['time_squared'] * df['s']
        
        # ===== BALL TRAJECTORY FEATURES =====
        if all(col in df.columns for col in ['ball_land_x', 'ball_land_y', 'final_pre_throw_x', 'final_pre_throw_y']):
            ball_dx = df['ball_land_x'] - df['final_pre_throw_x']
            ball_dy = df['ball_land_y'] - df['final_pre_throw_y']
            df['distance_to_ball_landing'] = np.sqrt(ball_dx**2 + ball_dy**2)
            df['angle_to_ball_landing'] = np.arctan2(ball_dy, ball_dx)
            
            # Ball direction unit vectors
            df['ball_direction_x'] = ball_dx / (df['distance_to_ball_landing'] + 1e-6)
            df['ball_direction_y'] = ball_dy / (df['distance_to_ball_landing'] + 1e-6)
            
            # Time until ball arrival
            estimated_ball_speed = 20.0
            df['estimated_time_to_ball'] = df['distance_to_ball_landing'] / estimated_ball_speed
            if 'time_seconds' in df.columns:
                df['time_ratio_to_ball'] = df['time_seconds'] / (df['estimated_time_to_ball'] + 0.1)
            
            # Closing speed
            if 'velocity_x' in df.columns:
                ball_unit_x = ball_dx / (df['distance_to_ball_landing'] + 1e-6)
                ball_unit_y = ball_dy / (df['distance_to_ball_landing'] + 1e-6)
                df['closing_speed'] = df['velocity_x'] * ball_unit_x + df['velocity_y'] * ball_unit_y
                
                df['projected_time_to_ball'] = df['distance_to_ball_landing'] / (np.abs(df['closing_speed']) + 0.1)
                if 'time_seconds' in df.columns:
                    df['time_urgency'] = df['time_seconds'] / (df['projected_time_to_ball'] + 0.1)
            
            # Temporal ball distance features
            if 'time_seconds' in df.columns:
                df['distance_to_ball_x_time'] = df['distance_to_ball_landing'] * df['time_seconds']
                if 'time_squared' in df.columns:
                    df['distance_to_ball_x_time_squared'] = df['distance_to_ball_landing'] * df['time_squared']
        
        # ===== TARGET RECEIVER FEATURES =====
        if all(col in df.columns for col in ['target_receiver_x', 'target_receiver_y', 'final_pre_throw_x', 'final_pre_throw_y']):
            target_dx = df['target_receiver_x'] - df['final_pre_throw_x']
            target_dy = df['target_receiver_y'] - df['final_pre_throw_y']
            df['distance_to_target'] = np.sqrt(target_dx**2 + target_dy**2)
            df['angle_to_target'] = np.arctan2(target_dy, target_dx)
            
            if 'time_seconds' in df.columns:
                df['distance_to_target_x_time'] = df['distance_to_target'] * df['time_seconds']
        
        # ===== TARGET INDICATOR =====
        if 'player_role' in df.columns:
            df['is_target_receiver'] = (df['player_role'] == "Targeted Receiver").astype(int)
        else:
            df['is_target_receiver'] = 0
        
        # ===== FIELD POSITION FEATURES =====
        if 'final_pre_throw_x' in df.columns:
            df['normalized_x'] = df['final_pre_throw_x'] / Config.FIELD_X_MAX
            df['field_region_x'] = pd.cut(df['final_pre_throw_x'], bins=6, labels=False)
            df['distance_from_endzone'] = np.minimum(df['final_pre_throw_x'], Config.FIELD_X_MAX - df['final_pre_throw_x'])
        
        if 'final_pre_throw_y' in df.columns:
            df['normalized_y'] = df['final_pre_throw_y'] / Config.FIELD_Y_MAX
            df['field_region_y'] = pd.cut(df['final_pre_throw_y'], bins=4, labels=False)
            df['distance_from_sideline'] = np.minimum(df['final_pre_throw_y'], Config.FIELD_Y_MAX - df['final_pre_throw_y'])
        
        # ===== GAME CONTEXT FEATURES =====
        if 'absolute_yardline_number' in df.columns:
            df['yards_to_endzone'] = df['absolute_yardline_number']
            df['is_redzone'] = (df['absolute_yardline_number'] <= 20).astype(int)
        
        # ===== TEAM INDICATOR =====
        if 'player_side' in df.columns:
            df['is_offense'] = (df['player_side'] == 'Offense').astype(int)
            df['is_passer'] = (df['player_role'] == 'Passer').astype(int)
            df['is_coverage'] = (df['player_role'] == 'Defensive Coverage').astype(int)
        else:
            df['is_offense'] = 0
            df['is_passer'] = 0
            df['is_coverage'] = 0
        
        # ===== PLAYER PHYSICAL ATTRIBUTES =====
        if all(col in df.columns for col in ['player_weight', 'height_inches']):
            valid_height = df['height_inches'] > 0
            df['bmi'] = np.nan
            df.loc[valid_height, 'bmi'] = (df.loc[valid_height, 'player_weight'] * 0.453592) / (
                (df.loc[valid_height, 'height_inches'] * 0.0254) ** 2)
        
        # ===== MOTION ANALYSIS FEATURES =====
        if all(col in df.columns for col in ['dir', 'o']):
            df['speed_orientation_discrepancy'] = np.abs(df['dir'] - df['o'])
        
        # ===== INTERACTION FEATURES =====
        if all(col in df.columns for col in ['s', 'a']):
            df['speed_times_acceleration'] = df['s'] * df['a']
        
        if all(col in df.columns for col in ['distance_to_ball_landing', 's']):
            df['distance_speed_ratio'] = df['distance_to_ball_landing'] / (df['s'] + 1.0)
            df['distance_ball_x_speed'] = df['distance_to_ball_landing'] * df['s']
        
        # ===== ADVANCED INTERACTION FEATURES =====
        if 'is_target_receiver' in df.columns and 'time_seconds' in df.columns:
            df['is_target_x_time'] = df['is_target_receiver'] * df['time_seconds']
            if 'time_squared' in df.columns:
                df['is_target_x_time_squared'] = df['is_target_receiver'] * df['time_squared']
        
        if 'is_offense' in df.columns:
            if 'is_early_play' in df.columns:
                df['is_offense_x_early_play'] = df['is_offense'] * df['is_early_play']
            if 'is_late_play' in df.columns:
                df['is_offense_x_late_play'] = df['is_offense'] * df['is_late_play']
        
        if 'is_target_receiver' in df.columns and 'is_late_play' in df.columns:
            df['is_target_x_late_play'] = df['is_target_receiver'] * df['is_late_play']
        
        # ===== TRAINING TARGETS =====
        if training_mode and all(col in df.columns for col in ['x', 'final_pre_throw_x', 'y', 'final_pre_throw_y']):
            df['displacement_x'] = df['x'] - df['final_pre_throw_x']
            df['displacement_y'] = df['y'] - df['final_pre_throw_y']
        
        return df
    
    def _encode_categorical_features(self, data_frame, categorical_columns):
        """Encode categorical variables with label encoding"""
        encoded_df = data_frame.copy()
        
        for col in categorical_columns:
            if col in encoded_df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    # Handle NaN values
                    encoded_df[col] = encoded_df[col].fillna('Unknown')
                    encoded_df[col] = self.label_encoders[col].fit_transform(encoded_df[col])
                else:
                    encoded_df[col] = encoded_df[col].fillna('Unknown')
                    # Handle unseen categories
                    unique_vals = set(encoded_df[col].unique())
                    trained_vals = set(self.label_encoders[col].classes_)
                    if not unique_vals.issubset(trained_vals):
                        # For unseen categories, use 'Unknown'
                        encoded_df[col] = encoded_df[col].apply(
                            lambda x: x if x in trained_vals else 'Unknown'
                        )
                    encoded_df[col] = self.label_encoders[col].transform(encoded_df[col])
            else:
                print(f"Warning: Categorical column '{col}' not found in data. Skipping.")
                # Add as constant if missing
                encoded_df[col] = 0
        
        return encoded_df
    
    def prepare_features(self, input_data, output_data, training_mode=False):
        """Complete feature engineering pipeline with temporal features"""
        print("Extracting temporal features...")
        temporal_features = self._extract_temporal_features(input_data)
        print("Incorporating target receiver data...")
        temporal_features = self._incorporate_target_receiver_data(temporal_features)
        
        # Identify available columns for merging
        available_columns = temporal_features.columns.tolist()
        merge_columns = ['game_id', 'play_id', 'nfl_id']
        
        # Add other columns if they exist
        optional_columns = [
            'final_pre_throw_x', 'final_pre_throw_y', 's', 'a', 'o', 'dir',
            'player_role', 'player_side', 'num_frames_output', 'ball_land_x', 
            'ball_land_y', 'target_receiver_x', 'target_receiver_y',
            'play_direction', 'absolute_yardline_number', 'height_inches', 'player_weight',
            # Temporal features
            'x_mean', 'x_std', 'x_min', 'x_max',
            'y_mean', 'y_std', 'y_min', 'y_max',
            's_mean', 's_std', 's_max', 's_min',
            'a_mean', 'a_std', 'a_max', 'a_min',
            'dir_change_rate', 'orientation_change_rate',
            'recent_displacement_x', 'recent_displacement_y', 'acceleration_trend'
        ]
        
        for col in optional_columns:
            if col in available_columns:
                merge_columns.append(col)
        
        print(f"Merging with columns: {len(merge_columns)} columns")
        
        # Merge with output data
        merged_data = output_data.merge(
            temporal_features[merge_columns],
            on=['game_id', 'play_id', 'nfl_id'],
            how='left'
        )
        
        print("Calculating advanced features...")
        return self._calculate_advanced_features(merged_data, training_mode=training_mode)
    
    def train_models(self):
        """Train ensemble models with cross-validation and neural networks"""
        # Print GPU information
        Config.print_gpu_info()
        
        # Load data
        print("Loading datasets...")
        train_input, train_output, test_input, test_template = self.load_and_combine_datasets()
        
        # Prepare features
        print("Preparing training features...")
        self.train_data = self.prepare_features(train_input, train_output, training_mode=True)
        
        print("Preparing test features...")
        self.test_data = self.prepare_features(test_input, test_template, training_mode=False)
        
        # Define feature sets based on available columns
        available_columns = self.train_data.columns.tolist()
        print(f"Available columns in training data: {len(available_columns)}")
        
        # Define comprehensive feature list
        potential_numerical_features = [
            # Position and movement
            'final_pre_throw_x', 'final_pre_throw_y', 's', 'a', 'o', 'dir',
            
            # Time features
            'time_seconds', 'time_normalized', 'time_squared', 'time_cubed', 
            'sqrt_time', 'log_time', 'time_sin', 'time_cos', 'time_sin_2', 'time_cos_2',
            'is_early_play', 'is_mid_play', 'is_late_play',
            
            # Historical statistics
            'x_mean', 'x_std', 'x_min', 'x_max',
            'y_mean', 'y_std', 'y_min', 'y_max',
            's_mean', 's_std', 's_max', 's_min',
            'a_mean', 'a_std', 'a_max', 'a_min',
            'dir_change_rate', 'orientation_change_rate',
            'recent_displacement_x', 'recent_displacement_y', 'acceleration_trend',
            
            # Movement consistency
            'speed_consistency', 'speed_deviation',
            'acceleration_consistency', 'acceleration_deviation',
            
            # Velocity and physics
            'velocity_x', 'velocity_y', 'momentum_magnitude',
            'expected_x_constant_v', 'expected_y_constant_v',
            'expected_x_with_accel', 'expected_y_with_accel',
            
            # Ball features
            'distance_to_ball_landing', 'angle_to_ball_landing', 'closing_speed',
            'ball_direction_x', 'ball_direction_y',
            'estimated_time_to_ball', 'time_ratio_to_ball',
            'projected_time_to_ball', 'time_urgency',
            'distance_to_ball_x_time', 'distance_to_ball_x_time_squared',
            
            # Target features
            'distance_to_target', 'is_target_receiver', 'angle_to_target',
            'distance_to_target_x_time', 'is_target_x_time_squared',
            
            # Field position
            'normalized_x', 'normalized_y', 'field_region_x', 'field_region_y',
            'distance_from_sideline', 'distance_from_endzone',
            
            # Game context
            'yards_to_endzone', 'is_offense', 'is_passer', 'is_coverage', 'is_redzone',
            
            # Player attributes
            'height_inches', 'player_weight', 'bmi',
            
            # Motion analysis
            'speed_orientation_discrepancy', 'motion_consistency',
            'proximity_to_ball_ratio', 'lateral_position_importance', 'downfield_progress',
            
            # Interactions
            'speed_times_acceleration', 'distance_speed_ratio', 'distance_ball_x_speed',
            'time_x_speed', 'time_x_acceleration', 'time_squared_x_speed',
            'is_target_x_time', 'is_offense_x_early_play', 'is_offense_x_late_play', 'is_target_x_late_play'
        ]
        
        potential_categorical_features = ['player_role', 'player_side', 'play_direction']
        
        # Select only features that exist in the data
        self.numerical_features = [f for f in potential_numerical_features if f in available_columns]
        self.categorical_features = [f for f in potential_categorical_features if f in available_columns]
        
        print(f"Using {len(self.numerical_features)} numerical features")
        print(f"Using {len(self.categorical_features)} categorical features")
        
        # Check if we have target variables for training
        if not all(col in self.train_data.columns for col in ['displacement_x', 'displacement_y']):
            raise KeyError("Target variables (displacement_x, displacement_y) not found in training data")
        
        # Prepare training data
        print("Preparing training matrix...")
        X_train = self.train_data[self.numerical_features + self.categorical_features].copy()
        X_train = self._encode_categorical_features(X_train, self.categorical_features)
        
        # Handle missing values
        X_train = X_train.fillna(0)
        
        # Scale numerical features
        self.scalers['numerical'] = StandardScaler()
        X_train[self.numerical_features] = self.scalers['numerical'].fit_transform(
            X_train[self.numerical_features]
        )
        
        y_dx = self.train_data['displacement_x'].values
        y_dy = self.train_data['displacement_y'].values
        
        print(f"Training data shape: {X_train.shape}")
        
        # Train with cross-validation
        self._train_with_cv(X_train, y_dx, y_dy)
        
        print("Model training completed!")
        return self
    
    def _train_with_cv(self, X_train, y_dx, y_dy):
        """Train models with cross-validation"""
        print("Training models with cross-validation...")
        
        # Cross-validation setup
        groups = self.train_data['game_id'].values
        gkf = GroupKFold(n_splits=Config.N_FOLDS)
        
        # Initialize model lists
        self.models_dx = {'xgb': [], 'lgb': [], 'cat': []}
        self.models_dy = {'xgb': [], 'lgb': [], 'cat': []}
        self.nn_models_dx = []
        self.nn_models_dy = []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, groups=groups)):
            print(f"  Fold {fold + 1}/{Config.N_FOLDS}")
            
            X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_dx, y_val_dx = y_dx[train_idx], y_dx[val_idx]
            y_train_dy, y_val_dy = y_dy[train_idx], y_dy[val_idx]
            
            # XGBoost with GPU support
            xgb_params = {
                'n_estimators': 2000,
                'learning_rate': 0.05,
                'max_depth': 8,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.seed + fold,
                'verbosity': 0,
                'objective': 'reg:squarederror',
            }
            
            # Add GPU support if available
            if Config.USE_GPU and torch.cuda.is_available():
                xgb_params.update({
                    'tree_method': 'gpu_hist',
                    'predictor': 'gpu_predictor',
                    'gpu_id': 0
                })
                print(f"    Training XGBoost on GPU for fold {fold + 1}")
            else:
                xgb_params['tree_method'] = 'hist'
                print(f"    Training XGBoost on CPU for fold {fold + 1}")
            
            xgb_dx = XGBRegressor(**xgb_params)
            xgb_dx.fit(X_train_fold, y_train_dx)
            self.models_dx['xgb'].append(xgb_dx)
            
            xgb_dy = XGBRegressor(**xgb_params)
            xgb_dy.fit(X_train_fold, y_train_dy)
            self.models_dy['xgb'].append(xgb_dy)
            
            # LightGBM with GPU support
            lgb_params = {
                'n_estimators': 2000,
                'learning_rate': 0.05,
                'max_depth': 8,
                'num_leaves': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.seed + fold,
                'verbosity': -1,
                'objective': 'regression',
            }
            
            # Add GPU support if available
            if Config.USE_GPU and torch.cuda.is_available():
                lgb_params.update({
                    'device': 'gpu',
                    'gpu_platform_id': 0,
                    'gpu_device_id': 0
                })
                print(f"    Training LightGBM on GPU for fold {fold + 1}")
            else:
                print(f"    Training LightGBM on CPU for fold {fold + 1}")
            
            lgb_dx = LGBMRegressor(**lgb_params)
            lgb_dx.fit(X_train_fold, y_train_dx)
            self.models_dx['lgb'].append(lgb_dx)
            
            lgb_dy = LGBMRegressor(**lgb_params)
            lgb_dy.fit(X_train_fold, y_train_dy)
            self.models_dy['lgb'].append(lgb_dy)
            
            # CatBoost with GPU support
            cat_params = {
                'iterations': 2000,
                'learning_rate': 0.05,
                'depth': 8,
                'random_seed': self.seed + fold,
                'verbose': False,
                'loss_function': 'RMSE',
                'eval_metric': 'RMSE',
            }
            
            # Add GPU support if available
            if Config.USE_GPU and torch.cuda.is_available():
                cat_params.update({
                    'task_type': 'GPU',
                    'devices': '0'
                })
                print(f"    Training CatBoost on GPU for fold {fold + 1}")
            else:
                print(f"    Training CatBoost on CPU for fold {fold + 1}")
            
            cat_dx = CatBoostRegressor(**cat_params)
            cat_dx.fit(X_train_fold, y_train_dx)
            self.models_dx['cat'].append(cat_dx)
            
            cat_dy = CatBoostRegressor(**cat_params)
            cat_dy.fit(X_train_fold, y_train_dy)
            self.models_dy['cat'].append(cat_dy)
            
            # Neural Network
            nn_dx = self._train_neural_network(X_train_fold.values, y_train_dx, X_val_fold.values, y_val_dx, self.seed + fold)
            self.nn_models_dx.append(nn_dx)
            
            nn_dy = self._train_neural_network(X_train_fold.values, y_train_dy, X_val_fold.values, y_val_dy, self.seed + fold + 100)
            self.nn_models_dy.append(nn_dy)
            
            # Clean up GPU memory after each fold
            Config.cleanup_gpu_memory()
    
    def _train_neural_network(self, X_train, y_train, X_val, y_val, seed=42):
        """Train a neural network model with GPU support"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        device = torch.device(Config.GPU_DEVICE)
        print(f"    Training Neural Network on {device} for seed {seed}")
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.FloatTensor(y_train.reshape(-1, 1))
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.FloatTensor(y_val.reshape(-1, 1))
        )
        
        train_loader = DataLoader(train_dataset, batch_size=Config.NN_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.NN_BATCH_SIZE)
        
        # Create model
        model = SimpleNN(X_train.shape[1]).to(device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.NN_LEARNING_RATE)
        
        best_val_loss = float('inf')
        best_model_state = model.state_dict()
        patience_counter = 0
        
        for epoch in range(Config.NN_EPOCHS):
            # Training
            model.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        return model
    
    def generate_predictions(self):
        """Generate ensemble predictions for test data"""
        print("Preparing test features...")
        X_test = self.test_data[self.numerical_features + self.categorical_features].copy()
        X_test = self._encode_categorical_features(X_test, self.categorical_features)
        X_test = X_test.fillna(0)
        
        # Scale numerical features
        X_test[self.numerical_features] = self.scalers['numerical'].transform(
            X_test[self.numerical_features]
        )
        
        # Generate ensemble predictions
        print("Generating predictions...")
        pred_dx = self._ensemble_prediction(X_test.values, self.models_dx, self.nn_models_dx)
        pred_dy = self._ensemble_prediction(X_test.values, self.models_dy, self.nn_models_dy)
        
        # Calculate final positions
        self.test_data['predicted_x'] = self.test_data['final_pre_throw_x'] + pred_dx
        self.test_data['predicted_y'] = self.test_data['final_pre_throw_y'] + pred_dy
        
        # Apply physics constraints
        self.test_data = self._apply_constraints(self.test_data)
        
        # Smooth trajectories
        self.test_data = self._smooth_trajectories(self.test_data)
        
        return self.test_data
    
    def _ensemble_prediction(self, X, tree_models, nn_models):
        """Generate weighted ensemble predictions from tree models and neural networks"""
        predictions = []
        
        # Tree model predictions
        weights = {'xgb': 0.3, 'lgb': 0.3, 'cat': 0.2}
        
        for model_name, models in tree_models.items():
            if model_name in weights:
                fold_preds = []
                for model in models:
                    fold_preds.append(model.predict(X))
                avg_pred = np.mean(fold_preds, axis=0)
                predictions.append(avg_pred * weights[model_name])
        
        # Neural network predictions
        device = torch.device(Config.GPU_DEVICE)
        X_tensor = torch.FloatTensor(X).to(device)
        
        nn_preds = []
        for model in nn_models:
            model.eval()
            with torch.no_grad():
                pred = model(X_tensor).cpu().numpy().squeeze()
            nn_preds.append(pred)
        
        nn_avg_pred = np.mean(nn_preds, axis=0)
        predictions.append(nn_avg_pred * 0.2)  # 20% weight for NN
        
        return np.sum(predictions, axis=0)
    
    def _apply_constraints(self, test_data):
        """Apply physics constraints to predictions"""
        print("Applying physics constraints...")
        
        dx = test_data['predicted_x'] - test_data['final_pre_throw_x']
        dy = test_data['predicted_y'] - test_data['final_pre_throw_y']
        displacement = np.sqrt(dx**2 + dy**2)
        
        if 'time_seconds' in test_data.columns:
            max_displacement = Config.MAX_SPEED * test_data['time_seconds']
            
            # Scale down impossible movements
            mask = displacement > max_displacement
            if np.any(mask):
                scale = max_displacement[mask] / (displacement[mask] + 1e-6)
                dx[mask] *= scale
                dy[mask] *= scale
                test_data.loc[mask, 'predicted_x'] = test_data.loc[mask, 'final_pre_throw_x'] + dx[mask]
                test_data.loc[mask, 'predicted_y'] = test_data.loc[mask, 'final_pre_throw_y'] + dy[mask]
        
        # Clip to field boundaries
        test_data['predicted_x'] = test_data['predicted_x'].clip(Config.FIELD_X_MIN, Config.FIELD_X_MAX)
        test_data['predicted_y'] = test_data['predicted_y'].clip(Config.FIELD_Y_MIN, Config.FIELD_Y_MAX)
        
        return test_data
    
    def _smooth_trajectories(self, test_data):
        """Smooth trajectories using Gaussian filtering"""
        print("Smoothing trajectories...")
        
        for (game_id, play_id, nfl_id), group in test_data.groupby(['game_id', 'play_id', 'nfl_id']):
            if len(group) > 3:
                idx = group.index
                test_data.loc[idx, 'predicted_x'] = gaussian_filter1d(group['predicted_x'].values, sigma=0.5)
                test_data.loc[idx, 'predicted_y'] = gaussian_filter1d(group['predicted_y'].values, sigma=0.5)
        
        return test_data
    
    def create_submission_file(self, output_path="submission.csv"):
        """Create submission file in required format"""
        # Create ID column
        self.test_data['unique_id'] = (
            self.test_data['game_id'].astype(str) + "_" +
            self.test_data['play_id'].astype(str) + "_" +
            self.test_data['nfl_id'].astype(str) + "_" +
            self.test_data['frame_id'].astype(str)
        )
        
        submission_df = self.test_data[['unique_id', 'predicted_x', 'predicted_y']].rename(
            columns={'predicted_x': 'x', 'predicted_y': 'y', 'unique_id': 'id'}
        )
        
        submission_df.to_csv(output_path, index=False)
        print(f"Submission file saved to {output_path}")
        print(f"Submission shape: {submission_df.shape}")
        return submission_df

# ================================================================================
# NEURAL NETWORK MODEL
# ================================================================================

class SimpleNN(nn.Module):
    """Simple neural network for regression"""
    
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.layers(x)

# ================================================================================
# MAIN EXECUTION
# ================================================================================

if __name__ == "__main__":
    try:
        # Check GPU requirements
        print("🔍 Checking GPU requirements...")
        check_gpu_requirements()
        
        # Print initial GPU information
        Config.print_gpu_info()
        
        # Initialize enhanced predictor
        predictor = EnhancedNFLPlayerMovementPredictor(
            data_dir="dataset/nfl-big-data-bowl-2026-prediction/",
            # data_dir="dataset/example-data/",
            seed=42
        )
        
        # Train models
        print("\n🚀 Training enhanced models with GPU acceleration...")
        predictor.train_models()
        
        # Generate predictions
        print("\n📊 Generating predictions...")
        predictions = predictor.generate_predictions()
        
        # Create submission
        print("\n💾 Creating submission file...")
        submission = predictor.create_submission_file("../run/submission.csv")
        
        print("\n✅ Enhanced pipeline completed successfully!")
        print(f"📈 Final submission shape: {submission.shape}")
        print("\nFirst 5 predictions:")
        print(submission.head())
        
        # Final GPU memory cleanup
        Config.cleanup_gpu_memory()
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        Config.cleanup_gpu_memory()