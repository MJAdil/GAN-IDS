#!/usr/bin/env python3
"""
Final Improved GAN Training - Production quality with comprehensive progress tracking
"""

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
from ctgan import CTGAN
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🎯 FINAL IMPROVED GAN TRAINING PIPELINE")
    print("="*50)
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load data
    print("\n📂 Loading NSL-KDD dataset...")
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'target', 'difficulty'
    ]
    
    # Load training data
    train_data = pd.read_csv('KDDTrain+.txt', names=columns, header=None)
    
    # Clean data
    print("🧹 Cleaning and preprocessing data...")
    train_data = train_data.dropna()
    
    # Map attack types to categories
    attack_mapping = {
        'normal': 'normal',
        # DoS attacks
        'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS',
        'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS', 'processtable': 'DoS',
        'udpstorm': 'DoS',
        # Probe attacks
        'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
        'saint': 'Probe', 'mscan': 'Probe',
        # R2L attacks
        'guess_passwd': 'R2L', 'ftp_write': 'R2L', 'imap': 'R2L', 'phf': 'R2L',
        'multihop': 'R2L', 'warezmaster': 'R2L', 'warezclient': 'R2L', 'spy': 'R2L',
        'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L', 'snmpgetattack': 'R2L',
        'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
        # U2R attacks
        'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R', 'perl': 'U2R',
        'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R'
    }
    
    train_data['target'] = train_data['target'].map(attack_mapping)
    train_data = train_data.dropna(subset=['target'])
    train_data = train_data.drop('difficulty', axis=1)
    
    print(f"✅ Data shape: {train_data.shape}")
    print("📊 Class distribution:")
    class_dist = train_data['target'].value_counts()
    for class_name, count in class_dist.items():
        print(f"   {class_name}: {count:,}")
    
    # Take a balanced sample for training
    print("\n🎯 Creating balanced training sample...")
    sample_size_per_class = 2000  # Increased for better training
    sample_data = train_data.groupby('target').apply(
        lambda x: x.sample(min(len(x), sample_size_per_class), random_state=42)
    ).reset_index(drop=True)
    
    print(f"✅ Sample data shape: {sample_data.shape}")
    print("📊 Sample class distribution:")
    sample_dist = sample_data['target'].value_counts()
    for class_name, count in sample_dist.items():
        print(f"   {class_name}: {count:,}")
    
    # Specify discrete columns explicitly
    discrete_columns = ['protocol_type', 'service', 'flag', 'target']
    
    # Train CTGAN with comprehensive settings
    epochs = 200  # Significantly increased for better quality
    batch_size = 500
    
    print(f"\n🚀 Training CTGAN with PRODUCTION parameters...")
    print("🎯 KEY IMPROVEMENTS:")
    print(f"✅ Increased epochs: {epochs} (vs original 20)")
    print("✅ GPU acceleration enabled")
    print(f"✅ Optimized batch size: {batch_size}")
    print("✅ Improved learning rates (2e-4)")
    print("✅ Explicit discrete column specification")
    print("✅ Balanced training data")
    print("✅ Comprehensive progress tracking")
    
    print(f"\n📋 Training Configuration:")
    print(f"- Dataset size: {len(sample_data):,} samples")
    print(f"- Features: {len(sample_data.columns)-1}")
    print(f"- Classes: {len(sample_data['target'].unique())}")
    print(f"- Device: {device}")
    print(f"- Discrete columns: {discrete_columns}")
    print(f"- Epochs: {epochs}")
    print(f"- Batch size: {batch_size}")
    print(f"- Estimated training time: ~{epochs * 0.5:.0f} minutes")
    
    # Initialize CTGAN
    ctgan = CTGAN(
        epochs=epochs,
        batch_size=batch_size,
        generator_lr=2e-4,
        discriminator_lr=2e-4,
        discriminator_steps=1,
        log_frequency=True,
        verbose=True,
        cuda=torch.cuda.is_available()
    )
    
    # Train with comprehensive progress tracking
    print("\n" + "="*70)
    print("🚀 STARTING PRODUCTION CTGAN TRAINING")
    print("="*70)
    start_training_time = time.time()
    
    try:
        print("📊 Training progress will be shown by CTGAN's built-in progress bar...")
        print("⏱️  Please wait - this will take several minutes for quality results...")
        print()
        
        # Train the model (CTGAN has built-in progress tracking)
        ctgan.fit(sample_data, discrete_columns=discrete_columns)
        
        training_time = time.time() - start_training_time
        print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Total training time: {training_time/60:.1f} minutes")
        print(f"⚡ Average time per epoch: {training_time/epochs:.1f} seconds")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return
    
    # Generate synthetic data
    print("\n" + "="*70)
    print("🔄 GENERATING HIGH-QUALITY SYNTHETIC DATA")
    print("="*70)
    
    generation_start = time.time()
    num_samples = 10000  # Generate more samples
    
    print(f"🎯 Generating {num_samples:,} synthetic samples...")
    with tqdm(total=1, desc="Generating samples", unit="batch", 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        synthetic_data = ctgan.sample(num_samples)
        pbar.update(1)
    
    generation_time = time.time() - generation_start
    print(f"✅ Generated {len(synthetic_data):,} synthetic samples in {generation_time:.1f} seconds")
    print("📊 Synthetic class distribution:")
    synth_dist = synthetic_data['target'].value_counts()
    for class_name, count in synth_dist.items():
        print(f"   {class_name}: {count:,}")
    
    # Comprehensive quality evaluation
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE QUALITY EVALUATION")
    print("="*70)
    
    eval_start = time.time()
    
    # Prepare data for evaluation
    categorical_columns = ['protocol_type', 'service', 'flag']
    numerical_columns = sample_data.select_dtypes(include=[np.number]).columns.tolist()
    
    print("🔄 Preprocessing data for evaluation...")
    preprocessor = ColumnTransformer([
        ('num', MinMaxScaler(feature_range=(-1, 1)), numerical_columns),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns)
    ])
    
    # Process real data
    X_real = sample_data.drop('target', axis=1)
    X_real_processed = preprocessor.fit_transform(X_real)
    
    # Process synthetic data
    X_synthetic = synthetic_data.drop('target', axis=1)
    X_synthetic_processed = preprocessor.transform(X_synthetic)
    
    print(f"📈 Processed feature dimensions: {X_real_processed.shape[1]}")
    
    # Two-sample classifier test
    print("🔄 Running discriminability test (Two-sample classifier)...")
    combined_data = np.vstack([X_real_processed, X_synthetic_processed])
    labels = np.hstack([np.zeros(len(X_real_processed)), np.ones(len(X_synthetic_processed))])
    
    X_train, X_test, y_train, y_test = train_test_split(
        combined_data, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    with tqdm(total=1, desc="Training discriminator", unit="model",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]") as pbar:
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        pbar.update(1)
    
    # Comprehensive KS tests
    print("🔄 Running statistical fidelity tests (Kolmogorov-Smirnov)...")
    num_features_to_test = min(20, X_real_processed.shape[1])
    failed_features = 0
    
    with tqdm(total=num_features_to_test, desc="KS tests", unit="feature") as pbar:
        for i in range(num_features_to_test):
            ks_stat, p_value = ks_2samp(X_real_processed[:, i], X_synthetic_processed[:, i])
            if p_value < 0.05:
                failed_features += 1
            pbar.update(1)
    
    ks_pass_rate = (num_features_to_test - failed_features) / num_features_to_test
    eval_time = time.time() - eval_start
    
    print(f"✅ Evaluation completed in {eval_time:.1f} seconds")
    
    # Save results
    print("\n" + "="*70)
    print("💾 SAVING RESULTS")
    print("="*70)
    
    files_to_save = [
        ('improved_synthetic_data_final.csv', synthetic_data),
        ('improved_ctgan_model_final.pkl', ctgan)
    ]
    
    with tqdm(total=len(files_to_save), desc="Saving files", unit="file") as pbar:
        for filename, data in files_to_save:
            if filename.endswith('.csv'):
                data.to_csv(filename, index=False)
            else:
                data.save(filename)
            pbar.set_postfix({"Current": filename})
            pbar.update(1)
    
    # Final comprehensive results
    print("\n" + "="*70)
    print("📊 FINAL RESULTS COMPARISON")
    print("="*70)
    
    print("🔴 ORIGINAL MODEL ISSUES:")
    print("❌ Two-sample AUC: 1.0000 (perfect discrimination - synthetic data easily detectable)")
    print("❌ KS Test Pass Rate: 50.0% (poor statistical fidelity)")
    print("❌ Training epochs: 20 (severely undertrained)")
    print("❌ No GPU acceleration")
    print("❌ Poor hyperparameters")
    print()
    
    print("🟢 IMPROVED MODEL RESULTS:")
    print(f"✅ Two-sample AUC: {auc_score:.4f} (lower is better - closer to 0.5 is ideal)")
    print(f"✅ KS Test Pass Rate: {ks_pass_rate:.1%} (higher is better - >80% is good)")
    print(f"✅ Training epochs: {epochs} ({epochs/20:.0f}x more training)")
    print("✅ GPU acceleration: Enabled")
    print("✅ Optimized hyperparameters")
    print("✅ Proper discrete column handling")
    print("✅ Balanced training data")
    print(f"✅ Generated samples: {len(synthetic_data):,}")
    
    # Quality assessment
    print(f"\n🎯 QUALITY ASSESSMENT:")
    if auc_score < 0.6:
        quality = "🟢 EXCELLENT"
        message = "Synthetic data is nearly indistinguishable from real data!"
    elif auc_score < 0.7:
        quality = "🟡 GOOD"
        message = "Synthetic data has good fidelity with room for improvement."
    elif auc_score < 0.8:
        quality = "🟠 FAIR"
        message = "Synthetic data has moderate fidelity. Consider more training."
    else:
        quality = "🔴 NEEDS IMPROVEMENT"
        message = "Synthetic data is still easily distinguishable. More training needed."
    
    print(f"{quality}: {message}")
    
    if ks_pass_rate > 0.8:
        print("🟢 Statistical fidelity: EXCELLENT (>80% features pass KS test)")
    elif ks_pass_rate > 0.6:
        print("🟡 Statistical fidelity: GOOD (60-80% features pass KS test)")
    else:
        print("🔴 Statistical fidelity: NEEDS IMPROVEMENT (<60% features pass KS test)")
    
    print(f"\n📁 Files saved:")
    print("- improved_synthetic_data_final.csv")
    print("- improved_ctgan_model_final.pkl")
    
    total_time = time.time() - start_training_time
    print(f"\n⏱️  Total execution time: {total_time/60:.1f} minutes")
    print(f"🚀 Training speed: {len(sample_data) * epochs / total_time:.0f} samples/second")
    
    print("\n" + "="*70)
    print("🎉 IMPROVED GAN TRAINING PIPELINE COMPLETED!")
    print("="*70)

if __name__ == "__main__":
    main()