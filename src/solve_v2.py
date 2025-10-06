import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor, Pool as CatBoostPool
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

from multiprocessing import Pool as MultiprocessingPool, cpu_count  # 导入多进程库
from tqdm.auto import tqdm
import pickle

ITERATIONS = 30000

basedir = 'dataset/nfl-big-data-bowl-2026-prediction'

def load_weekly_data(week_num):
    input_df = pd.read_csv(f'{basedir}/train/input_2023_w{week_num:02d}.csv')
    output_df = pd.read_csv(f'{basedir}/train/output_2023_w{week_num:02d}.csv')
    return input_df, output_df

def load_all_train_data():
    print("Loading training data...")
    with MultiprocessingPool(min(cpu_count(), 18)) as pool:
        results = list(tqdm(pool.imap(load_weekly_data, range(1, 19)), total=18))
    
    input_dfs = [r[0] for r in results]
    output_dfs = [r[1] for r in results]
    
    input_data = pd.concat(input_dfs, ignore_index=True)
    output_data = pd.concat(output_dfs, ignore_index=True)
    
    print(f"Input data shape: {input_data.shape}")
    print(f"Output data shape: {output_data.shape}")
    
    return input_data, output_data

def engineer_advanced_features(df):
    """Advanced feature engineering with sequence and interaction features"""
    df = df.copy()
    
    df['velocity_x'] = df['s'] * np.cos(np.radians(df['dir']))
    df['velocity_y'] = df['s'] * np.sin(np.radians(df['dir']))
    
    df['dist_to_ball'] = np.sqrt(
        (df['x'] - df['ball_land_x'])**2 + 
        (df['y'] - df['ball_land_y'])**2
    )
    
    df['angle_to_ball'] = np.arctan2(
        df['ball_land_y'] - df['y'],
        df['ball_land_x'] - df['x']
    )
    
    df['velocity_toward_ball'] = (
        df['velocity_x'] * np.cos(df['angle_to_ball']) + 
        df['velocity_y'] * np.sin(df['angle_to_ball'])
    )
    
    df['time_to_ball'] = df['num_frames_output'] / 10.0
    
    df['orientation_diff'] = np.abs(df['o'] - df['dir'])
    df['orientation_diff'] = np.minimum(df['orientation_diff'], 360 - df['orientation_diff'])
    
    df['role_targeted_receiver'] = (df['player_role'] == 'Targeted Receiver').astype(int)
    df['role_defensive_coverage'] = (df['player_role'] == 'Defensive Coverage').astype(int)
    df['role_passer'] = (df['player_role'] == 'Passer').astype(int)
    df['side_offense'] = (df['player_side'] == 'Offense').astype(int)
    
    height_parts = df['player_height'].str.split('-', expand=True)
    df['height_inches'] = height_parts[0].astype(float) * 12 + height_parts[1].astype(float)
    df['bmi'] = (df['player_weight'] / (df['height_inches']**2)) * 703
    
    df['acceleration_x'] = df['a'] * np.cos(np.radians(df['dir']))
    df['acceleration_y'] = df['a'] * np.sin(np.radians(df['dir']))
    
    df['distance_to_target_x'] = df['ball_land_x'] - df['x']
    df['distance_to_target_y'] = df['ball_land_y'] - df['y']
    
    df['speed_squared'] = df['s'] ** 2
    df['accel_magnitude'] = np.sqrt(df['acceleration_x']**2 + df['acceleration_y']**2)
    
    df['velocity_alignment'] = np.cos(df['angle_to_ball'] - np.radians(df['dir']))
    
    df['expected_x_at_ball'] = df['x'] + df['velocity_x'] * df['time_to_ball']
    df['expected_y_at_ball'] = df['y'] + df['velocity_y'] * df['time_to_ball']
    
    df['error_from_ball_x'] = df['expected_x_at_ball'] - df['ball_land_x']
    df['error_from_ball_y'] = df['expected_y_at_ball'] - df['ball_land_y']
    df['error_from_ball'] = np.sqrt(df['error_from_ball_x']**2 + df['error_from_ball_y']**2)
    
    df['momentum_x'] = df['player_weight'] * df['velocity_x']
    df['momentum_y'] = df['player_weight'] * df['velocity_y']
    
    df['kinetic_energy'] = 0.5 * df['player_weight'] * df['speed_squared']
    
    df['angle_diff'] = np.abs(df['o'] - np.degrees(df['angle_to_ball']))
    df['angle_diff'] = np.minimum(df['angle_diff'], 360 - df['angle_diff'])
    
    df['time_squared'] = df['time_to_ball'] ** 2
    df['dist_squared'] = df['dist_to_ball'] ** 2
    
    df['weighted_dist_by_time'] = df['dist_to_ball'] / (df['time_to_ball'] + 0.1)
    
    return df

def add_sequence_features(df):
    """Add temporal lag and rolling features"""
    df = df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
    
    group_cols = ['game_id', 'play_id', 'nfl_id']
    
    for lag in [1, 2, 3, 4, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's', 'a']:
            if col in df.columns:
                df[f'{col}_lag{lag}'] = df.groupby(group_cols)[col].shift(lag)
    
    for window in [3, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's']:
            if col in df.columns:
                df[f'{col}_rolling_mean_{window}'] = df.groupby(group_cols)[col].rolling(window, min_periods=1).mean().reset_index(level=[0,1,2], drop=True)
                df[f'{col}_rolling_std_{window}'] = df.groupby(group_cols)[col].rolling(window, min_periods=1).std().reset_index(level=[0,1,2], drop=True)
    
    for col in ['velocity_x', 'velocity_y']:
        if col in df.columns:
            df[f'{col}_delta'] = df.groupby(group_cols)[col].diff()
    
    return df

def create_training_dataset(input_df, output_df):
    output_df = output_df.copy()
    output_df['id'] = (output_df['game_id'].astype(str) + '_' + 
                       output_df['play_id'].astype(str) + '_' + 
                       output_df['nfl_id'].astype(str) + '_' + 
                       output_df['frame_id'].astype(str))
    
    output_df = output_df.rename(columns={'x': 'target_x', 'y': 'target_y'})
    
    input_agg = input_df.groupby(['game_id', 'play_id', 'nfl_id']).last().reset_index()
    
    if 'frame_id' in input_agg.columns:
        input_agg = input_agg.drop('frame_id', axis=1)
    
    merged = output_df.merge(
        input_agg,
        on=['game_id', 'play_id', 'nfl_id'],
        how='left',
        suffixes=('', '_input')
    )
    
    return merged

def physics_baseline_prediction(x, y, velocity_x, velocity_y, frame_id):
    time_delta = frame_id / 10.0
    pred_x = x + velocity_x * time_delta
    pred_y = y + velocity_y * time_delta
    pred_x = np.clip(pred_x, 0, 120)
    pred_y = np.clip(pred_y, 0, 53.3)
    return pred_x, pred_y

def main():
    print(f"CPU cores: {cpu_count()}")
    
    input_data, output_data = load_all_train_data()
    
    print("\n=== Advanced Feature Engineering ===")
    print("Step 1: Engineering advanced physics features...")
    input_features = engineer_advanced_features(input_data)
    
    print("Step 2: Adding sequence and rolling features...")
    input_features = add_sequence_features(input_features)
    
    print(f"Feature engineered data shape: {input_features.shape}")
    print(f"Total features: {input_features.shape[1]}")
    
    print("\nStep 3: Creating training dataset...")
    train_df = create_training_dataset(input_features, output_data)
    print(f"Training dataset shape: {train_df.shape}")
    
    feature_cols = [
        'x', 'y', 's', 'a', 'o', 'dir',
        'velocity_x', 'velocity_y', 'dist_to_ball', 'angle_to_ball',
        'velocity_toward_ball', 'time_to_ball', 'orientation_diff',
        'role_targeted_receiver', 'role_defensive_coverage', 'role_passer',
        'side_offense', 'height_inches', 'player_weight', 'bmi',
        'ball_land_x', 'ball_land_y', 'num_frames_output', 'frame_id',
        'acceleration_x', 'acceleration_y', 'distance_to_target_x', 'distance_to_target_y',
        'speed_squared', 'accel_magnitude', 'velocity_alignment',
        'expected_x_at_ball', 'expected_y_at_ball',
        'error_from_ball_x', 'error_from_ball_y', 'error_from_ball',
        'momentum_x', 'momentum_y', 'kinetic_energy',
        'angle_diff', 'time_squared', 'dist_squared', 'weighted_dist_by_time'
    ]
    
    for lag in [1, 2, 3, 4, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's', 'a']:
            feature_cols.append(f'{col}_lag{lag}')
    
    for window in [3, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's']:
            feature_cols.append(f'{col}_rolling_mean_{window}')
            feature_cols.append(f'{col}_rolling_std_{window}')
    
    feature_cols.extend(['velocity_x_delta', 'velocity_y_delta'])
    
    available_features = [col for col in feature_cols if col in train_df.columns]
    print(f"Available features: {len(available_features)}")
    
    train_df = train_df.dropna(subset=available_features + ['target_x', 'target_y'])
    print(f"Training data after removing NaNs: {train_df.shape}")
    
    print("\n=== Physics Baseline ===")
    baseline_x, baseline_y = physics_baseline_prediction(
        train_df['x'].values,
        train_df['y'].values,
        train_df['velocity_x'].values,
        train_df['velocity_y'].values,
        train_df['frame_id'].values
    )
    
    baseline_rmse = np.sqrt(
        0.5 * (mean_squared_error(train_df['target_x'], baseline_x) +
               mean_squared_error(train_df['target_y'], baseline_y))
    )
    print(f"Physics Baseline RMSE: {baseline_rmse:.4f}")
    
    # Prepare data for CatBoost
    X = train_df[available_features].values
    y_x = train_df['target_x'].values
    y_y = train_df['target_y'].values
    
    # Initialize 5-fold cross-validation
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    models_x = []
    models_y = []
    val_rmse_scores = []
    
    print(f"\n=== Training Ultra-Optimized CatBoost with 5-Fold CV ===")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\nFold {fold}/{n_folds}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_x_train, y_x_val = y_x[train_idx], y_x[val_idx]
        y_y_train, y_y_val = y_y[train_idx], y_y[val_idx]
        
        print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
        
        # Define CatBoost Pools
        train_pool_x = CatBoostPool(X_train, y_x_train)
        val_pool_x = CatBoostPool(X_val, y_x_val)
        train_pool_y = CatBoostPool(X_train, y_y_train)
        val_pool_y = CatBoostPool(X_val, y_y_val)
        
        # Train X-coordinate model
        print(f"Training X coordinate model for fold {fold}...")
        model_x = CatBoostRegressor(
            iterations=ITERATIONS,
            learning_rate=0.05,
            depth=10,
            l2_leaf_reg=3.0,
            random_seed=42, 
            task_type='GPU',
            devices='0',
            early_stopping_rounds=500,
            verbose=200,
            loss_function='RMSE'
        )
        
        model_x.fit(
            train_pool_x,
            eval_set=val_pool_x,
            verbose=200
        )
        models_x.append(model_x)
        
        # Train Y-coordinate model
        print(f"Training Y coordinate model for fold {fold}...")
        model_y = CatBoostRegressor(
            iterations=ITERATIONS,
            learning_rate=0.05,
            depth=10,
            l2_leaf_reg=3.0,
            random_seed=42, 
            task_type='GPU',
            devices='0',
            early_stopping_rounds=500,
            verbose=200,
            loss_function='RMSE'
        )
        
        model_y.fit(
            train_pool_y,
            eval_set=val_pool_y,
            verbose=200
        )
        models_y.append(model_y)
        
        # Validation predictions
        pred_x = model_x.predict(X_val)
        pred_y = model_y.predict(X_val)
        
        pred_x = np.clip(pred_x, 0, 120)
        pred_y = np.clip(pred_y, 0, 53.3)
        
        # Compute RMSE for this fold
        fold_rmse = np.sqrt(
            0.5 * (mean_squared_error(y_x_val, pred_x) +
                   mean_squared_error(y_y_val, pred_y))
        )
        val_rmse_scores.append(fold_rmse)
        print(f"Fold {fold} RMSE: {fold_rmse:.4f}")
    
    # Average RMSE across folds
    catboost_rmse = np.mean(val_rmse_scores)
    print(f"\nAverage CatBoost RMSE across {n_folds} folds: {catboost_rmse:.4f}")
    print(f"Standard Deviation of RMSE: {np.std(val_rmse_scores):.4f}")
    
    print(f"\n{'='*60}")
    print(f"ULTRA-OPTIMIZED CATBOOST PERFORMANCE (5-FOLD CV)")
    print(f"{'='*60}")
    print(f"Physics Baseline RMSE:     {baseline_rmse:.4f}")
    print(f"Ultra-Optimized CatBoost:  {catboost_rmse:.4f}")
    print(f"Improvement:               {((baseline_rmse - catboost_rmse) / baseline_rmse * 100):.2f}%")
    print(f"Target RMSE:               0.9000")
    target_met = 'YES - TARGET ACHIEVED!' if catboost_rmse < 0.9 else 'NO - Continuing optimization...'
    print(f"Target Met:                {target_met}")
    print(f"{'='*60}")
    
    # Feature importance (averaged across folds)
    importance_x = np.mean([model.get_feature_importance() for model in models_x], axis=0)
    importance_y = np.mean([model.get_feature_importance() for model in models_y], axis=0)
    
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance_x': importance_x,
        'importance_y': importance_y
    })
    feature_importance['importance_avg'] = (feature_importance['importance_x'] + 
                                            feature_importance['importance_y']) / 2
    feature_importance = feature_importance.sort_values('importance_avg', ascending=False)
    
    print("\nTop 30 Most Important Features (Averaged across folds):")
    print(feature_importance.head(30).to_string())
    
    # Save models
    with open('catboost_5fold_models.pkl', 'wb') as f:
        pickle.dump({
            'models_x': models_x,
            'models_y': models_y,
            'features': available_features,
            'rmse': catboost_rmse
        }, f)
    print("\nModels saved to catboost_5fold_models.pkl")
    
    print("\n=== Generating Submission ===")
    test_input = pd.read_csv(f'{basedir}/test_input.csv')
    test_data = pd.read_csv(f'{basedir}/test.csv')
    
    print("Engineering features for test data...")
    test_features = engineer_advanced_features(test_input)
    test_features = add_sequence_features(test_features)
    
    test_agg = test_features.groupby(['game_id', 'play_id', 'nfl_id']).last().reset_index()
    
    if 'frame_id' in test_agg.columns:
        test_agg = test_agg.drop('frame_id', axis=1)
    
    test_merged = test_data.merge(
        test_agg,
        on=['game_id', 'play_id', 'nfl_id'],
        how='left'
    )
    
    test_merged['id'] = (test_merged['game_id'].astype(str) + '_' + 
                         test_merged['play_id'].astype(str) + '_' + 
                         test_merged['nfl_id'].astype(str) + '_' + 
                         test_merged['frame_id'].astype(str))
    
    for col in available_features:
        if col not in test_merged.columns:
            test_merged[col] = 0
    
    X_test = test_merged[available_features].fillna(0).values
    
    # Ensemble predictions across all folds
    pred_x_test = np.mean([model.predict(X_test) for model in models_x], axis=0)
    pred_y_test = np.mean([model.predict(X_test) for model in models_y], axis=0)
    
    pred_x_test = np.clip(pred_x_test, 0, 120)
    pred_y_test = np.clip(pred_y_test, 0, 53.3)
    
    submission = pd.DataFrame({
        'id': test_merged['id'],
        'x': pred_x_test,
        'y': pred_y_test
    })
    
    submission.to_csv('run/submission_v2.csv', index=False)
    print(f"\n[SUCCESS] Submission saved: run/submission_v2.csv")
    print(f"Shape: {submission.shape}")
    
    print("\n=== Submission Validation ===")
    print(f"No NaN values: {submission.isnull().sum().sum() == 0}")
    print(f"X range: [{submission['x'].min():.2f}, {submission['x'].max():.2f}]")
    print(f"Y range: [{submission['y'].min():.2f}, {submission['y'].max():.2f}]")
    print(f"Unique IDs: {submission['id'].nunique()}")
    
    return catboost_rmse

if __name__ == "__main__":
    final_rmse = main()
    print(f"\n[FINAL] Validation RMSE: {final_rmse:.4f}")
    achievement = 'ACHIEVED!' if final_rmse < 0.9 else 'Not yet - need further optimization'
    print(f"[FINAL] Target RMSE < 0.9: {achievement}")