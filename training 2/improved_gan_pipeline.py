#!/usr/bin/env python3
"""
Improved GAN Training Pipeline for IDS Data Generation
Addresses the issues found in the original implementation:
- Insufficient training epochs (20 -> 400+)
- Poor statistical fidelity
- High discriminability
- Missing GPU optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from scipy.stats import ks_2samp
from tqdm import tqdm
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

class ImprovedGANPipeline:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.device = device
        self.scaler = None
        self.encoder = None
        self.preprocessor = None
        self.feature_names = None
        self.categorical_columns = ['protocol_type', 'service', 'flag']
        self.target_column = 'target'
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
    
    def load_and_preprocess_data(self, train_file='KDDTrain+.txt', test_file='KDDTest+.txt'):
        """Load and preprocess NSL-KDD dataset"""
        print("Loading NSL-KDD dataset...")
        
        # Column names for NSL-KDD dataset
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
        train_data = pd.read_csv(train_file, names=columns, header=None)
        print(f"Training data shape: {train_data.shape}")
        
        # Map detailed attack types to high-level categories
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
        
        # Apply mapping
        train_data['target'] = train_data['target'].map(attack_mapping)
        train_data = train_data.dropna(subset=['target'])  # Remove unmapped attacks
        
        # Remove difficulty column
        train_data = train_data.drop('difficulty', axis=1)
        
        print("Class distribution:")
        print(train_data['target'].value_counts())
        
        return train_data
    
    def prepare_features(self, data):
        """Prepare features for training"""
        print("Preparing features...")
        
        # Separate features and target
        X = data.drop(self.target_column, axis=1)
        y = data[self.target_column]
        
        # Identify numerical columns
        numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"Numerical columns: {len(numerical_columns)}")
        print(f"Categorical columns: {len(self.categorical_columns)}")
        
        # Create preprocessor
        self.preprocessor = ColumnTransformer([
            ('num', MinMaxScaler(feature_range=(-1, 1)), numerical_columns),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), self.categorical_columns)
        ])
        
        # Fit and transform
        X_processed = self.preprocessor.fit_transform(X)
        
        # Get feature names
        num_features = numerical_columns
        cat_features = self.preprocessor.named_transformers_['cat'].get_feature_names_out(self.categorical_columns)
        self.feature_names = num_features + list(cat_features)
        
        print(f"Total features after preprocessing: {X_processed.shape[1]}")
        
        return X_processed, y
    
    def analyze_data_quality(self, real_data, synthetic_data, feature_names):
        """Comprehensive data quality analysis"""
        print("\n" + "="*50)
        print("DATA QUALITY ANALYSIS")
        print("="*50)
        
        # Convert to DataFrames for easier analysis
        real_df = pd.DataFrame(real_data, columns=feature_names)
        synthetic_df = pd.DataFrame(synthetic_data, columns=feature_names)
        
        # 1. Statistical Tests (KS test for continuous features)
        print("\n1. Kolmogorov-Smirnov Tests:")
        ks_results = []
        failed_features = []
        
        for i, feature in enumerate(feature_names):
            if i < len(feature_names) - len([f for f in feature_names if '_' in f]):  # Numerical features
                ks_stat, p_value = ks_2samp(real_df.iloc[:, i], synthetic_df.iloc[:, i])
                ks_results.append((feature, ks_stat, p_value))
                if p_value < 0.05:
                    failed_features.append(feature)
        
        print(f"Features failing KS test (p < 0.05): {len(failed_features)}/{len(ks_results)}")
        print(f"Pass rate: {(len(ks_results) - len(failed_features))/len(ks_results)*100:.1f}%")
        
        # 2. Two-sample classifier test
        print("\n2. Two-sample Classifier Test:")
        
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
        
        # 3. Basic statistics comparison
        print("\n3. Statistical Moments Comparison:")
        real_stats = real_df.describe()
        synthetic_stats = synthetic_df.describe()
        
        mean_diff = np.abs(real_stats.loc['mean'] - synthetic_stats.loc['mean']).mean()
        std_diff = np.abs(real_stats.loc['std'] - synthetic_stats.loc['std']).mean()
        
        print(f"Average mean difference: {mean_diff:.4f}")
        print(f"Average std difference: {std_diff:.4f}")
        
        return {
            'ks_pass_rate': (len(ks_results) - len(failed_features))/len(ks_results),
            'auc_score': auc_score,
            'mean_diff': mean_diff,
            'std_diff': std_diff
        }
    
    def train_improved_ctgan(self, X, y, epochs=500, batch_size=500, save_path='improved_ctgan_model.pkl'):
        """Train CTGAN with improved parameters"""
        print(f"\nTraining CTGAN with {epochs} epochs...")
        
        # Combine features and target for CTGAN
        data_df = pd.DataFrame(X, columns=self.feature_names)
        data_df[self.target_column] = y
        
        # Initialize CTGAN with improved parameters
        from ctgan import CTGAN
        
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
        
        print("Starting CTGAN training...")
        print(f"Data shape: {data_df.shape}")
        print(f"Using GPU: {torch.cuda.is_available()}")
        
        # Train the model
        ctgan.fit(data_df)
        
        # Save the model
        print(f"Saving model to {save_path}")
        ctgan.save(save_path)
        
        return ctgan
    
    def generate_balanced_synthetic_data(self, ctgan, original_data, target_samples_per_class=10000):
        """Generate balanced synthetic data"""
        print(f"\nGenerating balanced synthetic data...")
        
        # Get class distribution
        class_counts = original_data[self.target_column].value_counts()
        print("Original class distribution:")
        print(class_counts)
        
        synthetic_samples = []
        
        for class_name in class_counts.index:
            print(f"Generating {target_samples_per_class} samples for class '{class_name}'...")
            
            # Generate samples for this class
            class_samples = ctgan.sample(
                target_samples_per_class, 
                conditions={self.target_column: class_name}
            )
            
            synthetic_samples.append(class_samples)
        
        # Combine all synthetic samples
        synthetic_data = pd.concat(synthetic_samples, ignore_index=True)
        
        print(f"\nGenerated synthetic data shape: {synthetic_data.shape}")
        print("Synthetic class distribution:")
        print(synthetic_data[self.target_column].value_counts())
        
        return synthetic_data
    
    def save_results(self, synthetic_data, quality_metrics, save_dir='results'):
        """Save results and generate reports"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save synthetic data
        synthetic_path = os.path.join(save_dir, 'improved_synthetic_data.csv')
        synthetic_data.to_csv(synthetic_path, index=False)
        print(f"Synthetic data saved to: {synthetic_path}")
        
        # Save quality metrics
        metrics_path = os.path.join(save_dir, 'quality_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write("IMPROVED GAN QUALITY METRICS\n")
            f.write("="*40 + "\n\n")
            f.write(f"KS Test Pass Rate: {quality_metrics['ks_pass_rate']:.3f}\n")
            f.write(f"Two-sample AUC: {quality_metrics['auc_score']:.4f}\n")
            f.write(f"Mean Difference: {quality_metrics['mean_diff']:.4f}\n")
            f.write(f"Std Difference: {quality_metrics['std_diff']:.4f}\n")
        
        print(f"Quality metrics saved to: {metrics_path}")
        
        # Generate comparison plots
        self.generate_comparison_plots(synthetic_data, save_dir)
    
    def generate_comparison_plots(self, synthetic_data, save_dir):
        """Generate comparison plots"""
        print("Generating comparison plots...")
        
        # Class distribution comparison
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        synthetic_data[self.target_column].value_counts().plot(kind='bar')
        plt.title('Synthetic Data Class Distribution')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        # Load original for comparison
        original_data = self.load_and_preprocess_data()
        original_data[self.target_column].value_counts().plot(kind='bar')
        plt.title('Original Data Class Distribution')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'class_distribution_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to: {save_dir}")

def main():
    """Main execution function"""
    print("IMPROVED GAN TRAINING PIPELINE FOR IDS")
    print("="*50)
    
    # Initialize pipeline
    pipeline = ImprovedGANPipeline(random_state=42)
    
    # Load and preprocess data
    data = pipeline.load_and_preprocess_data()
    X, y = pipeline.prepare_features(data)
    
    # Train improved CTGAN
    ctgan = pipeline.train_improved_ctgan(X, y, epochs=500, batch_size=500)
    
    # Generate balanced synthetic data
    synthetic_data = pipeline.generate_balanced_synthetic_data(ctgan, data, target_samples_per_class=5000)
    
    # Separate features and target from synthetic data
    synthetic_X = synthetic_data.drop(pipeline.target_column, axis=1).values
    
    # Analyze quality
    quality_metrics = pipeline.analyze_data_quality(X, synthetic_X, pipeline.feature_names)
    
    # Save results
    pipeline.save_results(synthetic_data, quality_metrics)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"✅ Generated {len(synthetic_data)} synthetic samples")
    print(f"✅ KS Test Pass Rate: {quality_metrics['ks_pass_rate']:.1%}")
    print(f"✅ Two-sample AUC: {quality_metrics['auc_score']:.4f}")
    print("✅ Results saved to 'results/' directory")

if __name__ == "__main__":
    main()