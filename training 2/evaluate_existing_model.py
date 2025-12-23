#!/usr/bin/env python3
"""
Evaluate the existing CTGAN model to understand current issues
"""

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_existing_data():
    """Load existing preprocessed data"""
    print("Loading existing data files...")
    
    # Load preprocessed data
    real_data = np.load('real_preprocessed.npy')
    synthetic_data = np.load('synthetic_preprocessed.npy')
    
    print(f"Real data shape: {real_data.shape}")
    print(f"Synthetic data shape: {synthetic_data.shape}")
    
    return real_data, synthetic_data

def evaluate_quality(real_data, synthetic_data):
    """Evaluate the quality of existing synthetic data"""
    print("\n" + "="*50)
    print("EVALUATING EXISTING MODEL QUALITY")
    print("="*50)
    
    # 1. Two-sample classifier test
    print("\n1. Two-sample Classifier Test:")
    
    # Combine real and synthetic data
    combined_data = np.vstack([real_data, synthetic_data])
    labels = np.hstack([np.zeros(len(real_data)), np.ones(len(synthetic_data))])
    
    # Train classifier
    X_train, X_test, y_train, y_test = train_test_split(
        combined_data, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC Score: {auc_score:.4f}")
    
    if auc_score < 0.6:
        print("✅ EXCELLENT: Synthetic data is indistinguishable from real data")
    elif auc_score < 0.7:
        print("✅ GOOD: Synthetic data has good fidelity")
    elif auc_score < 0.8:
        print("⚠️  FAIR: Synthetic data has moderate fidelity")
    else:
        print("❌ POOR: Synthetic data is easily distinguishable from real data")
    
    # 2. Basic statistics comparison
    print("\n2. Statistical Comparison:")
    
    real_mean = np.mean(real_data, axis=0)
    synthetic_mean = np.mean(synthetic_data, axis=0)
    real_std = np.std(real_data, axis=0)
    synthetic_std = np.std(synthetic_data, axis=0)
    
    mean_diff = np.mean(np.abs(real_mean - synthetic_mean))
    std_diff = np.mean(np.abs(real_std - synthetic_std))
    
    print(f"Average mean difference: {mean_diff:.4f}")
    print(f"Average std difference: {std_diff:.4f}")
    
    # 3. Feature-wise KS tests (sample of features)
    print("\n3. Kolmogorov-Smirnov Tests (first 20 features):")
    failed_features = 0
    total_features = min(20, real_data.shape[1])
    
    for i in range(total_features):
        ks_stat, p_value = ks_2samp(real_data[:, i], synthetic_data[:, i])
        if p_value < 0.05:
            failed_features += 1
    
    pass_rate = (total_features - failed_features) / total_features
    print(f"Features failing KS test: {failed_features}/{total_features}")
    print(f"Pass rate: {pass_rate:.1%}")
    
    return {
        'auc_score': auc_score,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'ks_pass_rate': pass_rate
    }

def check_gpu_availability():
    """Check GPU availability and specs"""
    print("GPU Information:")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("No GPU available")

def main():
    """Main evaluation function"""
    print("EVALUATING EXISTING GAN MODEL")
    print("="*50)
    
    # Check GPU
    check_gpu_availability()
    
    try:
        # Load existing data
        real_data, synthetic_data = load_existing_data()
        
        # Evaluate quality
        metrics = evaluate_quality(real_data, synthetic_data)
        
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Two-sample AUC: {metrics['auc_score']:.4f} (should be ~0.5)")
        print(f"KS Test Pass Rate: {metrics['ks_pass_rate']:.1%} (should be >80%)")
        print(f"Mean Difference: {metrics['mean_diff']:.4f} (should be <0.1)")
        print(f"Std Difference: {metrics['std_diff']:.4f} (should be <0.1)")
        
        print("\n🔍 ISSUES IDENTIFIED:")
        if metrics['auc_score'] > 0.8:
            print("❌ High discriminability - synthetic data easily detectable")
        if metrics['ks_pass_rate'] < 0.8:
            print("❌ Poor statistical fidelity - distributions don't match")
        if metrics['mean_diff'] > 0.1:
            print("❌ Large mean differences between real and synthetic")
        if metrics['std_diff'] > 0.1:
            print("❌ Large variance differences between real and synthetic")
            
        print("\n💡 RECOMMENDATIONS:")
        print("✅ Increase training epochs from 20 to 400+")
        print("✅ Use GPU acceleration for faster training")
        print("✅ Implement conditional generation for better class balance")
        print("✅ Add regularization to improve statistical fidelity")
        print("✅ Use improved hyperparameters (learning rates, batch size)")
        
    except FileNotFoundError as e:
        print(f"❌ Error loading data files: {e}")
        print("Make sure the following files exist:")
        print("- real_preprocessed.npy")
        print("- synthetic_preprocessed.npy")

if __name__ == "__main__":
    main()