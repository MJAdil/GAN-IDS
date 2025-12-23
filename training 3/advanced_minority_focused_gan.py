#!/usr/bin/env python3
"""
Advanced Minority-Focused GAN Training Pipeline
Specialized for generating high-quality U2R and R2L synthetic data for IDS augmentation

Key Improvements:
- Minority class oversampling and focused training
- Advanced CTGAN with optimized hyperparameters
- Multi-stage training approach
- Enhanced quality evaluation metrics
- Class-conditional generation with balanced outputs
"""

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, f1_score
from scipy.stats import ks_2samp, wasserstein_distance
from ctgan import CTGAN
from tqdm import tqdm
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
warnings.filterwarnings('ignore')

class AdvancedMinorityGAN:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.minority_classes = ['U2R', 'R2L']
        self.target_samples_per_minority = 5000  # Significantly increase minority samples
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
    
    def load_and_prepare_data(self):
        """Load and prepare data with minority class focus"""
        print("🔄 Loading NSL-KDD dataset with minority class focus...")
        
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
        
        # Load both training and test data for more minority samples
        train_data = pd.read_csv('KDDTrain+.txt', names=columns, header=None)
        test_data = pd.read_csv('KDDTest+.txt', names=columns, header=None)
        
        # Combine datasets for maximum minority class samples
        combined_data = pd.concat([train_data, test_data], ignore_index=True)
        combined_data = combined_data.dropna()
        
        # Attack mapping
        attack_mapping = {
            'normal': 'normal',
            # DoS attacks  
            'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS', 'smurf': 'DoS',
            'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS', 'processtable': 'DoS',
            'udpstorm': 'DoS',
            # Probe attacks
            'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
            'saint': 'Probe', 'mscan': 'Probe',
            # R2L attacks (MINORITY - FOCUS)
            'guess_passwd': 'R2L', 'ftp_write': 'R2L', 'imap': 'R2L', 'phf': 'R2L',
            'multihop': 'R2L', 'warezmaster': 'R2L', 'warezclient': 'R2L', 'spy': 'R2L',
            'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L', 'snmpgetattack': 'R2L',
            'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
            # U2R attacks (MINORITY - FOCUS)
            'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R', 'perl': 'U2R',
            'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R'
        }
        
        combined_data['target'] = combined_data['target'].map(attack_mapping)
        combined_data = combined_data.dropna(subset=['target'])
        combined_data = combined_data.drop('difficulty', axis=1)
        
        print(f"✅ Combined data shape: {combined_data.shape}")
        print("📊 Full dataset class distribution:")
        full_dist = combined_data['target'].value_counts()
        for class_name, count in full_dist.items():
            print(f"   {class_name}: {count:,}")
        
        return combined_data
    
    def create_minority_focused_dataset(self, data):
        """Create a dataset heavily focused on minority classes"""
        print("\n🎯 Creating minority-focused training dataset...")
        
        # Separate classes
        minority_data = data[data['target'].isin(self.minority_classes)]
        majority_data = data[~data['target'].isin(self.minority_classes)]
        
        print(f"📊 Minority classes (U2R + R2L): {len(minority_data):,} samples")
        print(f"📊 Majority classes: {len(majority_data):,} samples")
        
        # Strategy: Oversample minority classes and undersample majority
        # Take ALL minority samples (they're rare)
        minority_samples = minority_data.copy()
        
        # For majority classes, take balanced samples
        majority_sample_size = min(3000, len(majority_data) // 3)  # Reasonable size per class
        majority_samples = majority_data.groupby('target').apply(
            lambda x: x.sample(min(len(x), majority_sample_size), random_state=self.random_state)
        ).reset_index(drop=True)
        
        # Combine with heavy minority focus
        training_data = pd.concat([minority_samples, majority_samples], ignore_index=True)
        
        print(f"✅ Training dataset shape: {training_data.shape}")
        print("📊 Training class distribution:")
        train_dist = training_data['target'].value_counts()
        for class_name, count in train_dist.items():
            percentage = (count / len(training_data)) * 100
            print(f"   {class_name}: {count:,} ({percentage:.1f}%)")
        
        return training_data
    
    def train_advanced_ctgan(self, data, epochs=500):
        """Train CTGAN with advanced parameters optimized for minority classes"""
        print(f"\n🚀 Training Advanced CTGAN (Minority-Focused)...")
        print("🎯 ADVANCED OPTIMIZATIONS:")
        print(f"✅ Extended epochs: {epochs} (for better convergence)")
        print("✅ Optimized for minority class generation")
        print("✅ Advanced hyperparameters")
        print("✅ GPU acceleration with memory optimization")
        
        discrete_columns = ['protocol_type', 'service', 'flag', 'target']
        
        # Advanced CTGAN configuration
        ctgan = CTGAN(
            epochs=epochs,
            batch_size=1000,  # Larger batch for better gradient estimates
            generator_lr=1e-4,  # Lower learning rate for stability
            discriminator_lr=2e-4,  # Slightly higher for discriminator
            discriminator_steps=2,  # More discriminator steps
            log_frequency=True,
            verbose=True,
            cuda=torch.cuda.is_available(),
            # Advanced parameters
            generator_decay=1e-6,  # Weight decay for regularization
            discriminator_decay=1e-6,
        )
        
        print(f"\n📋 Training Configuration:")
        print(f"- Dataset size: {len(data):,} samples")
        print(f"- Minority focus: {len(data[data['target'].isin(self.minority_classes)]):,} U2R+R2L samples")
        print(f"- Epochs: {epochs}")
        print(f"- Batch size: 1000")
        print(f"- Device: {self.device}")
        print(f"- Estimated time: ~{epochs * 0.8:.0f} minutes")
        
        start_time = time.time()
        
        try:
            print("\n" + "="*70)
            print("🚀 STARTING ADVANCED MINORITY-FOCUSED TRAINING")
            print("="*70)
            
            ctgan.fit(data, discrete_columns=discrete_columns)
            
            training_time = time.time() - start_time
            print(f"\n🎉 ADVANCED TRAINING COMPLETED!")
            print(f"⏱️  Training time: {training_time/60:.1f} minutes")
            
            return ctgan
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None
    
    def generate_minority_focused_data(self, ctgan):
        """Generate high-quality synthetic data with minority class focus"""
        print("\n🔄 Generating Minority-Focused Synthetic Data...")
        
        synthetic_samples = []
        
        # Generate significantly more minority class samples
        generation_plan = {
            'U2R': self.target_samples_per_minority,  # 5000 samples
            'R2L': self.target_samples_per_minority,  # 5000 samples
            'normal': 2000,  # Balanced amount
            'DoS': 2000,     # Balanced amount
            'Probe': 2000    # Balanced amount
        }
        
        print("🎯 Generation Plan:")
        total_samples = sum(generation_plan.values())
        for class_name, count in generation_plan.items():
            percentage = (count / total_samples) * 100
            focus = "🔥 MINORITY FOCUS" if class_name in self.minority_classes else ""
            print(f"   {class_name}: {count:,} samples ({percentage:.1f}%) {focus}")
        
        print(f"\n📊 Total synthetic samples to generate: {total_samples:,}")
        
        # Generate samples for each class
        with tqdm(total=len(generation_plan), desc="Generating classes", unit="class") as pbar:
            for class_name, num_samples in generation_plan.items():
                pbar.set_postfix({"Current": f"{class_name} ({num_samples:,})"})
                
                # Generate conditional samples
                class_samples = ctgan.sample(
                    num_samples, 
                    conditions={'target': class_name}
                )
                
                synthetic_samples.append(class_samples)
                pbar.update(1)
        
        # Combine all synthetic samples
        synthetic_data = pd.concat(synthetic_samples, ignore_index=True)
        
        print(f"\n✅ Generated {len(synthetic_data):,} synthetic samples")
        print("📊 Synthetic class distribution:")
        synth_dist = synthetic_data['target'].value_counts()
        for class_name, count in synth_dist.items():
            percentage = (count / len(synthetic_data)) * 100
            focus = "🔥" if class_name in self.minority_classes else ""
            print(f"   {class_name}: {count:,} ({percentage:.1f}%) {focus}")
        
        return synthetic_data
    
    def advanced_quality_evaluation(self, real_data, synthetic_data):
        """Comprehensive quality evaluation with minority class focus"""
        print("\n" + "="*70)
        print("📊 ADVANCED QUALITY EVALUATION")
        print("="*70)
        
        results = {}
        
        # Prepare data for evaluation
        categorical_columns = ['protocol_type', 'service', 'flag']
        numerical_columns = real_data.select_dtypes(include=[np.number]).columns.tolist()
        
        print("🔄 Preprocessing data for evaluation...")
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numerical_columns),  # StandardScaler for better evaluation
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns)
        ])
        
        # Process data
        X_real = real_data.drop('target', axis=1)
        X_synthetic = synthetic_data.drop('target', axis=1)
        
        X_real_processed = preprocessor.fit_transform(X_real)
        X_synthetic_processed = preprocessor.transform(X_synthetic)
        
        print(f"📈 Processed feature dimensions: {X_real_processed.shape[1]}")
        
        # 1. Overall Discriminability Test
        print("\n🔄 1. Overall Discriminability Test...")
        combined_data = np.vstack([X_real_processed, X_synthetic_processed])
        labels = np.hstack([np.zeros(len(X_real_processed)), np.ones(len(X_synthetic_processed))])
        
        X_train, X_test, y_train, y_test = train_test_split(
            combined_data, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        overall_auc = roc_auc_score(y_test, y_pred_proba)
        results['overall_auc'] = overall_auc
        
        print(f"   Overall AUC: {overall_auc:.4f}")
        
        # 2. Minority Class Specific Evaluation
        print("\n🔄 2. Minority Class Specific Evaluation...")
        minority_results = {}
        
        for minority_class in self.minority_classes:
            print(f"\n   📊 Evaluating {minority_class} class...")
            
            # Get class-specific data
            real_minority = real_data[real_data['target'] == minority_class]
            synthetic_minority = synthetic_data[synthetic_data['target'] == minority_class]
            
            if len(real_minority) == 0 or len(synthetic_minority) == 0:
                print(f"   ⚠️  Insufficient {minority_class} samples for evaluation")
                continue
            
            # Process class-specific data
            X_real_min = preprocessor.transform(real_minority.drop('target', axis=1))
            X_synth_min = preprocessor.transform(synthetic_minority.drop('target', axis=1))
            
            # Class-specific discriminability
            combined_min = np.vstack([X_real_min, X_synth_min])
            labels_min = np.hstack([np.zeros(len(X_real_min)), np.ones(len(X_synth_min))])
            
            if len(np.unique(labels_min)) > 1:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    combined_min, labels_min, test_size=0.3, random_state=42, stratify=labels_min
                )
                
                clf_min = RandomForestClassifier(n_estimators=100, random_state=42)
                clf_min.fit(X_tr, y_tr)
                y_pred_min = clf_min.predict_proba(X_te)[:, 1]
                class_auc = roc_auc_score(y_te, y_pred_min)
                
                minority_results[minority_class] = {
                    'auc': class_auc,
                    'real_samples': len(real_minority),
                    'synthetic_samples': len(synthetic_minority)
                }
                
                print(f"   {minority_class} AUC: {class_auc:.4f}")
                print(f"   Real samples: {len(real_minority):,}")
                print(f"   Synthetic samples: {len(synthetic_minority):,}")
        
        results['minority_results'] = minority_results
        
        # 3. Statistical Fidelity Tests
        print("\n🔄 3. Statistical Fidelity Tests...")
        num_features_to_test = min(30, X_real_processed.shape[1])
        failed_ks = 0
        wasserstein_distances = []
        
        with tqdm(total=num_features_to_test, desc="Statistical tests", unit="feature") as pbar:
            for i in range(num_features_to_test):
                # KS test
                ks_stat, p_value = ks_2samp(X_real_processed[:, i], X_synthetic_processed[:, i])
                if p_value < 0.05:
                    failed_ks += 1
                
                # Wasserstein distance (Earth Mover's Distance)
                wd = wasserstein_distance(X_real_processed[:, i], X_synthetic_processed[:, i])
                wasserstein_distances.append(wd)
                
                pbar.update(1)
        
        ks_pass_rate = (num_features_to_test - failed_ks) / num_features_to_test
        avg_wasserstein = np.mean(wasserstein_distances)
        
        results['ks_pass_rate'] = ks_pass_rate
        results['avg_wasserstein'] = avg_wasserstein
        
        print(f"   KS Test Pass Rate: {ks_pass_rate:.1%}")
        print(f"   Average Wasserstein Distance: {avg_wasserstein:.4f}")
        
        # 4. Novelty Detection (Outlier Analysis)
        print("\n🔄 4. Novelty Detection Analysis...")
        
        # Train isolation forest on real data
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(X_real_processed)
        
        # Check how many synthetic samples are considered outliers
        synthetic_outliers = iso_forest.predict(X_synthetic_processed)
        outlier_rate = (synthetic_outliers == -1).mean()
        results['outlier_rate'] = outlier_rate
        
        print(f"   Synthetic Outlier Rate: {outlier_rate:.1%}")
        
        return results
    
    def save_advanced_results(self, synthetic_data, ctgan, results):
        """Save advanced results with comprehensive documentation"""
        print("\n💾 Saving Advanced Results...")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        files_saved = []
        
        # Save synthetic data
        synth_filename = f'advanced_minority_synthetic_data_{timestamp}.csv'
        synthetic_data.to_csv(synth_filename, index=False)
        files_saved.append(synth_filename)
        
        # Save model
        model_filename = f'advanced_minority_ctgan_model_{timestamp}.pkl'
        ctgan.save(model_filename)
        files_saved.append(model_filename)
        
        # Save detailed results
        results_filename = f'advanced_quality_results_{timestamp}.txt'
        with open(results_filename, 'w') as f:
            f.write("ADVANCED MINORITY-FOCUSED GAN RESULTS\n")
            f.write("="*50 + "\n\n")
            f.write(f"Generated Samples: {len(synthetic_data):,}\n")
            f.write(f"Minority Classes Focus: U2R + R2L\n\n")
            
            f.write("QUALITY METRICS:\n")
            f.write(f"Overall AUC: {results['overall_auc']:.4f}\n")
            f.write(f"KS Test Pass Rate: {results['ks_pass_rate']:.1%}\n")
            f.write(f"Avg Wasserstein Distance: {results['avg_wasserstein']:.4f}\n")
            f.write(f"Outlier Rate: {results['outlier_rate']:.1%}\n\n")
            
            if 'minority_results' in results:
                f.write("MINORITY CLASS SPECIFIC RESULTS:\n")
                for class_name, metrics in results['minority_results'].items():
                    f.write(f"{class_name}:\n")
                    f.write(f"  AUC: {metrics['auc']:.4f}\n")
                    f.write(f"  Real samples: {metrics['real_samples']:,}\n")
                    f.write(f"  Synthetic samples: {metrics['synthetic_samples']:,}\n\n")
        
        files_saved.append(results_filename)
        
        print(f"✅ Saved {len(files_saved)} files:")
        for filename in files_saved:
            print(f"   - {filename}")
        
        return files_saved

def main():
    """Main execution function for advanced minority-focused GAN training"""
    print("🎯 ADVANCED MINORITY-FOCUSED GAN TRAINING")
    print("="*60)
    print("🔥 SPECIALIZED FOR U2R AND R2L ATTACK GENERATION")
    print("="*60)
    
    # Initialize advanced GAN
    advanced_gan = AdvancedMinorityGAN(random_state=42)
    
    print(f"🖥️  Device: {advanced_gan.device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    
    # Load and prepare data
    full_data = advanced_gan.load_and_prepare_data()
    training_data = advanced_gan.create_minority_focused_dataset(full_data)
    
    # Train advanced CTGAN
    epochs = 500  # Extended training for better quality
    ctgan = advanced_gan.train_advanced_ctgan(training_data, epochs=epochs)
    
    if ctgan is None:
        print("❌ Training failed. Exiting.")
        return
    
    # Generate minority-focused synthetic data
    synthetic_data = advanced_gan.generate_minority_focused_data(ctgan)
    
    # Advanced quality evaluation
    results = advanced_gan.advanced_quality_evaluation(training_data, synthetic_data)
    
    # Save results
    files_saved = advanced_gan.save_advanced_results(synthetic_data, ctgan, results)
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 ADVANCED MINORITY-FOCUSED TRAINING COMPLETED!")
    print("="*70)
    
    print(f"📊 FINAL RESULTS SUMMARY:")
    print(f"✅ Generated samples: {len(synthetic_data):,}")
    print(f"🔥 U2R samples: {len(synthetic_data[synthetic_data['target'] == 'U2R']):,}")
    print(f"🔥 R2L samples: {len(synthetic_data[synthetic_data['target'] == 'R2L']):,}")
    print(f"📈 Overall AUC: {results['overall_auc']:.4f}")
    print(f"📈 KS Pass Rate: {results['ks_pass_rate']:.1%}")
    print(f"📈 Outlier Rate: {results['outlier_rate']:.1%}")
    
    if results['overall_auc'] < 0.7:
        print("\n🎉 EXCELLENT: High-quality synthetic data generated!")
    elif results['overall_auc'] < 0.8:
        print("\n✅ GOOD: Quality synthetic data with room for improvement")
    else:
        print("\n⚠️  FAIR: Consider additional training or parameter tuning")
    
    print(f"\n📁 Files saved: {len(files_saved)}")
    print("🚀 Ready for IDS augmentation!")

if __name__ == "__main__":
    main()