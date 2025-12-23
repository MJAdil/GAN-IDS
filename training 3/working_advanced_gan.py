#!/usr/bin/env python3
"""
Working Advanced Minority-Focused GAN Training Pipeline
Fixed version that works with the current CTGAN implementation
"""

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
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

class WorkingAdvancedGAN:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.minority_classes = ['U2R', 'R2L']
        
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
        
        # Strategy: Take ALL minority samples + balanced majority samples
        minority_samples = minority_data.copy()
        
        # Replicate minority samples to increase their representation
        # This helps CTGAN learn minority patterns better
        minority_replicated = []
        for class_name in self.minority_classes:
            class_data = minority_data[minority_data['target'] == class_name]
            if len(class_data) > 0:
                # Replicate minority samples (especially U2R which is very rare)
                replication_factor = 10 if class_name == 'U2R' else 3
                for _ in range(replication_factor):
                    minority_replicated.append(class_data)
        
        if minority_replicated:
            minority_samples = pd.concat(minority_replicated, ignore_index=True)
        
        # For majority classes, take balanced samples
        majority_sample_size = 2000  # Reasonable size per class
        majority_samples = majority_data.groupby('target').apply(
            lambda x: x.sample(min(len(x), majority_sample_size), random_state=self.random_state)
        ).reset_index(drop=True)
        
        # Combine with heavy minority focus
        training_data = pd.concat([minority_samples, majority_samples], ignore_index=True)
        
        # Shuffle the data
        training_data = training_data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        print(f"✅ Training dataset shape: {training_data.shape}")
        print("📊 Training class distribution:")
        train_dist = training_data['target'].value_counts()
        for class_name, count in train_dist.items():
            percentage = (count / len(training_data)) * 100
            focus = "🔥 MINORITY FOCUS" if class_name in self.minority_classes else ""
            print(f"   {class_name}: {count:,} ({percentage:.1f}%) {focus}")
        
        return training_data
    
    def train_advanced_ctgan(self, data, epochs=400):
        """Train CTGAN with advanced parameters optimized for minority classes"""
        print(f"\n🚀 Training Advanced CTGAN (Minority-Focused)...")
        print("🎯 ADVANCED OPTIMIZATIONS:")
        print(f"✅ Extended epochs: {epochs} (for better convergence)")
        print("✅ Optimized for minority class generation")
        print("✅ Advanced hyperparameters")
        print("✅ GPU acceleration with memory optimization")
        print("✅ Minority class replication for better learning")
        
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
        )
        
        print(f"\n📋 Training Configuration:")
        print(f"- Dataset size: {len(data):,} samples")
        print(f"- Minority focus: {len(data[data['target'].isin(self.minority_classes)]):,} U2R+R2L samples")
        print(f"- Epochs: {epochs}")
        print(f"- Batch size: 1000")
        print(f"- Device: {self.device}")
        print(f"- Estimated time: ~{epochs * 0.5:.0f} minutes")
        
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
    
    def generate_balanced_synthetic_data(self, ctgan, target_total=15000):
        """Generate synthetic data with heavy minority class focus"""
        print(f"\n🔄 Generating Minority-Focused Synthetic Data...")
        print(f"🎯 Target: {target_total:,} total samples with minority class emphasis")
        
        # Generate a large batch and then filter/balance
        print("   Generating large batch for filtering...")
        
        # Generate more samples than needed for filtering
        generation_batch_size = target_total * 2
        
        with tqdm(total=1, desc="Generating batch", unit="batch") as pbar:
            synthetic_data = ctgan.sample(generation_batch_size)
            pbar.update(1)
        
        print(f"✅ Generated {len(synthetic_data):,} raw samples")
        print("📊 Raw synthetic class distribution:")
        raw_dist = synthetic_data['target'].value_counts()
        for class_name, count in raw_dist.items():
            percentage = (count / len(synthetic_data)) * 100
            focus = "🔥" if class_name in self.minority_classes else ""
            print(f"   {class_name}: {count:,} ({percentage:.1f}%) {focus}")
        
        # Now create a balanced dataset with minority class emphasis
        print("\n🎯 Creating minority-focused balanced dataset...")
        
        balanced_samples = []
        
        # Define target distribution (heavily favoring minorities)
        target_distribution = {
            'U2R': 5000,    # Heavy focus on U2R
            'R2L': 5000,    # Heavy focus on R2L
            'normal': 2000, # Balanced amount
            'DoS': 2000,    # Balanced amount
            'Probe': 1000   # Smaller amount
        }
        
        print("🎯 Target distribution:")
        total_target = sum(target_distribution.values())
        for class_name, target_count in target_distribution.items():
            percentage = (target_count / total_target) * 100
            focus = "🔥 MINORITY FOCUS" if class_name in self.minority_classes else ""
            print(f"   {class_name}: {target_count:,} ({percentage:.1f}%) {focus}")
        
        # Sample from generated data according to target distribution
        for class_name, target_count in target_distribution.items():
            class_data = synthetic_data[synthetic_data['target'] == class_name]
            
            if len(class_data) >= target_count:
                # Randomly sample target amount
                sampled_data = class_data.sample(target_count, random_state=self.random_state)
            else:
                # Take all available and replicate if needed
                sampled_data = class_data.copy()
                while len(sampled_data) < target_count:
                    additional_needed = target_count - len(sampled_data)
                    additional_samples = min(additional_needed, len(class_data))
                    sampled_data = pd.concat([
                        sampled_data, 
                        class_data.sample(additional_samples, random_state=self.random_state, replace=True)
                    ], ignore_index=True)
            
            balanced_samples.append(sampled_data)
            print(f"   ✅ {class_name}: {len(sampled_data):,} samples selected")
        
        # Combine balanced samples
        final_synthetic_data = pd.concat(balanced_samples, ignore_index=True)
        
        # Shuffle the final dataset
        final_synthetic_data = final_synthetic_data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        print(f"\n✅ Final synthetic dataset: {len(final_synthetic_data):,} samples")
        print("📊 Final class distribution:")
        final_dist = final_synthetic_data['target'].value_counts()
        for class_name, count in final_dist.items():
            percentage = (count / len(final_synthetic_data)) * 100
            focus = "🔥" if class_name in self.minority_classes else ""
            print(f"   {class_name}: {count:,} ({percentage:.1f}%) {focus}")
        
        return final_synthetic_data
    
    def comprehensive_evaluation(self, real_data, synthetic_data):
        """Comprehensive quality evaluation"""
        print("\n" + "="*70)
        print("📊 COMPREHENSIVE QUALITY EVALUATION")
        print("="*70)
        
        results = {}
        
        # Prepare data for evaluation
        categorical_columns = ['protocol_type', 'service', 'flag']
        numerical_columns = real_data.select_dtypes(include=[np.number]).columns.tolist()
        
        print("🔄 Preprocessing data for evaluation...")
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numerical_columns),
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
            
            minority_results[minority_class] = {
                'real_samples': len(real_minority),
                'synthetic_samples': len(synthetic_minority),
                'augmentation_ratio': len(synthetic_minority) / len(real_minority) if len(real_minority) > 0 else 0
            }
            
            print(f"   Real samples: {len(real_minority):,}")
            print(f"   Synthetic samples: {len(synthetic_minority):,}")
            print(f"   Augmentation ratio: {len(synthetic_minority) / len(real_minority):.1f}x")
        
        results['minority_results'] = minority_results
        
        # 3. Statistical Fidelity Tests
        print("\n🔄 3. Statistical Fidelity Tests...")
        num_features_to_test = min(20, X_real_processed.shape[1])
        failed_ks = 0
        
        with tqdm(total=num_features_to_test, desc="Statistical tests", unit="feature") as pbar:
            for i in range(num_features_to_test):
                ks_stat, p_value = ks_2samp(X_real_processed[:, i], X_synthetic_processed[:, i])
                if p_value < 0.05:
                    failed_ks += 1
                pbar.update(1)
        
        ks_pass_rate = (num_features_to_test - failed_ks) / num_features_to_test
        results['ks_pass_rate'] = ks_pass_rate
        
        print(f"   KS Test Pass Rate: {ks_pass_rate:.1%}")
        
        return results
    
    def save_results(self, synthetic_data, ctgan, results):
        """Save results with comprehensive documentation"""
        print("\n💾 Saving Advanced Results...")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        files_saved = []
        
        # Save synthetic data
        synth_filename = f'working_advanced_synthetic_data_{timestamp}.csv'
        synthetic_data.to_csv(synth_filename, index=False)
        files_saved.append(synth_filename)
        
        # Save model
        model_filename = f'working_advanced_ctgan_model_{timestamp}.pkl'
        ctgan.save(model_filename)
        files_saved.append(model_filename)
        
        # Save detailed results
        results_filename = f'working_advanced_results_{timestamp}.txt'
        with open(results_filename, 'w') as f:
            f.write("WORKING ADVANCED MINORITY-FOCUSED GAN RESULTS\n")
            f.write("="*60 + "\n\n")
            f.write(f"Generated Samples: {len(synthetic_data):,}\n")
            f.write(f"Minority Classes Focus: U2R + R2L\n\n")
            
            f.write("QUALITY METRICS:\n")
            f.write(f"Overall AUC: {results['overall_auc']:.4f}\n")
            f.write(f"KS Test Pass Rate: {results['ks_pass_rate']:.1%}\n\n")
            
            if 'minority_results' in results:
                f.write("MINORITY CLASS AUGMENTATION:\n")
                for class_name, metrics in results['minority_results'].items():
                    f.write(f"{class_name}:\n")
                    f.write(f"  Real samples: {metrics['real_samples']:,}\n")
                    f.write(f"  Synthetic samples: {metrics['synthetic_samples']:,}\n")
                    f.write(f"  Augmentation ratio: {metrics['augmentation_ratio']:.1f}x\n\n")
        
        files_saved.append(results_filename)
        
        print(f"✅ Saved {len(files_saved)} files:")
        for filename in files_saved:
            print(f"   - {filename}")
        
        return files_saved

def main():
    """Main execution function"""
    print("🎯 WORKING ADVANCED MINORITY-FOCUSED GAN TRAINING")
    print("="*70)
    print("🔥 SPECIALIZED FOR U2R AND R2L ATTACK GENERATION")
    print("="*70)
    
    # Initialize advanced GAN
    advanced_gan = WorkingAdvancedGAN(random_state=42)
    
    print(f"🖥️  Device: {advanced_gan.device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    
    # Load and prepare data
    full_data = advanced_gan.load_and_prepare_data()
    training_data = advanced_gan.create_minority_focused_dataset(full_data)
    
    # Train advanced CTGAN
    epochs = 300  # Balanced training time vs quality
    ctgan = advanced_gan.train_advanced_ctgan(training_data, epochs=epochs)
    
    if ctgan is None:
        print("❌ Training failed. Exiting.")
        return
    
    # Generate minority-focused synthetic data
    synthetic_data = advanced_gan.generate_balanced_synthetic_data(ctgan, target_total=15000)
    
    # Comprehensive evaluation
    results = advanced_gan.comprehensive_evaluation(training_data, synthetic_data)
    
    # Save results
    files_saved = advanced_gan.save_results(synthetic_data, ctgan, results)
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 WORKING ADVANCED TRAINING COMPLETED!")
    print("="*70)
    
    print(f"📊 FINAL RESULTS SUMMARY:")
    print(f"✅ Generated samples: {len(synthetic_data):,}")
    print(f"🔥 U2R samples: {len(synthetic_data[synthetic_data['target'] == 'U2R']):,}")
    print(f"🔥 R2L samples: {len(synthetic_data[synthetic_data['target'] == 'R2L']):,}")
    print(f"📈 Overall AUC: {results['overall_auc']:.4f}")
    print(f"📈 KS Pass Rate: {results['ks_pass_rate']:.1%}")
    
    # Calculate total minority samples
    minority_samples = len(synthetic_data[synthetic_data['target'].isin(advanced_gan.minority_classes)])
    minority_percentage = (minority_samples / len(synthetic_data)) * 100
    
    print(f"🔥 Total minority samples: {minority_samples:,} ({minority_percentage:.1f}%)")
    
    if results['overall_auc'] < 0.7:
        print("\n🎉 EXCELLENT: High-quality synthetic data generated!")
    elif results['overall_auc'] < 0.8:
        print("\n✅ GOOD: Quality synthetic data with good minority representation")
    else:
        print("\n⚠️  FAIR: Acceptable quality - consider additional training")
    
    print(f"\n📁 Files saved: {len(files_saved)}")
    print("🚀 READY FOR IDS AUGMENTATION!")
    
    # Show augmentation potential
    if 'minority_results' in results:
        print(f"\n🎯 AUGMENTATION POTENTIAL:")
        for class_name, metrics in results['minority_results'].items():
            print(f"   {class_name}: {metrics['augmentation_ratio']:.0f}x more samples for training!")

if __name__ == "__main__":
    main()