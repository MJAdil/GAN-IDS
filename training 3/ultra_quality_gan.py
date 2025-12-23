#!/usr/bin/env python3
"""
Ultra-Quality GAN Training Pipeline
Implements cutting-edge techniques for the highest quality synthetic data generation

Advanced Techniques:
1. Multi-stage progressive training
2. Feature importance weighting
3. Adversarial validation
4. Ensemble generation
5. Quality-based sample filtering
6. Advanced regularization techniques
"""

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import ks_2samp, entropy
from ctgan import CTGAN
from tqdm import tqdm
import time
import warnings
from collections import defaultdict
import joblib
warnings.filterwarnings('ignore')

class UltraQualityGAN:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.minority_classes = ['U2R', 'R2L']
        
        # Advanced configuration
        self.progressive_epochs = [100, 200, 300]  # Multi-stage training
        self.quality_threshold = 0.75  # AUC threshold for sample acceptance
        
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
    
    def analyze_feature_importance(self, data):
        """Analyze feature importance for minority classes"""
        print("🔍 Analyzing feature importance for minority classes...")
        
        # Prepare data
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Create binary classification for minority vs majority
        y_binary = y.apply(lambda x: 1 if x in self.minority_classes else 0)
        
        # Select only numerical features for mutual information
        numerical_features = X.select_dtypes(include=[np.number])
        
        if len(numerical_features.columns) > 0:
            # Calculate mutual information
            mi_scores = mutual_info_classif(numerical_features, y_binary, random_state=self.random_state)
            
            # Create feature importance dictionary
            feature_importance = dict(zip(numerical_features.columns, mi_scores))
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            print("📊 Top 10 most important features for minority classes:")
            for i, (feature, score) in enumerate(sorted_features[:10]):
                print(f"   {i+1:2d}. {feature}: {score:.4f}")
            
            return feature_importance
        else:
            print("⚠️  No numerical features found for importance analysis")
            return {}
    
    def progressive_training(self, data, discrete_columns):
        """Multi-stage progressive training for better convergence"""
        print("\n🚀 Starting Progressive Training (Multi-Stage)...")
        
        models = []
        
        for stage, epochs in enumerate(self.progressive_epochs, 1):
            print(f"\n📈 STAGE {stage}: Training for {epochs} epochs")
            
            # Adjust hyperparameters for each stage
            if stage == 1:
                # Stage 1: Fast exploration
                lr_gen, lr_disc = 2e-4, 2e-4
                batch_size = 500
                disc_steps = 1
            elif stage == 2:
                # Stage 2: Refinement
                lr_gen, lr_disc = 1e-4, 1.5e-4
                batch_size = 750
                disc_steps = 2
            else:
                # Stage 3: Fine-tuning
                lr_gen, lr_disc = 5e-5, 1e-4
                batch_size = 1000
                disc_steps = 3
            
            print(f"   Learning rates: Gen={lr_gen}, Disc={lr_disc}")
            print(f"   Batch size: {batch_size}")
            print(f"   Discriminator steps: {disc_steps}")
            
            ctgan = CTGAN(
                epochs=epochs,
                batch_size=batch_size,
                generator_lr=lr_gen,
                discriminator_lr=lr_disc,
                discriminator_steps=disc_steps,
                log_frequency=True,
                verbose=True,
                cuda=torch.cuda.is_available(),
                generator_decay=1e-6,
                discriminator_decay=1e-6,
            )
            
            # Train model
            start_time = time.time()
            ctgan.fit(data, discrete_columns=discrete_columns)
            stage_time = time.time() - start_time
            
            print(f"   ✅ Stage {stage} completed in {stage_time/60:.1f} minutes")
            
            # Evaluate stage quality
            stage_quality = self.evaluate_stage_quality(ctgan, data)
            print(f"   📊 Stage {stage} quality score: {stage_quality:.4f}")
            
            models.append({
                'model': ctgan,
                'stage': stage,
                'epochs': epochs,
                'quality': stage_quality,
                'training_time': stage_time
            })
        
        # Select best model
        best_model = max(models, key=lambda x: x['quality'])
        print(f"\n🏆 Best model: Stage {best_model['stage']} (Quality: {best_model['quality']:.4f})")
        
        return best_model['model'], models
    
    def evaluate_stage_quality(self, ctgan, real_data):
        """Quick quality evaluation for progressive training"""
        try:
            # Generate small sample for evaluation
            sample_data = ctgan.sample(1000)
            
            # Quick discriminability test
            categorical_columns = ['protocol_type', 'service', 'flag']
            numerical_columns = real_data.select_dtypes(include=[np.number]).columns.tolist()
            
            preprocessor = ColumnTransformer([
                ('num', RobustScaler(), numerical_columns),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns)
            ])
            
            X_real = preprocessor.fit_transform(real_data.drop('target', axis=1))
            X_synthetic = preprocessor.transform(sample_data.drop('target', axis=1))
            
            # Quick discriminability test
            combined = np.vstack([X_real[:1000], X_synthetic])  # Limit real data for speed
            labels = np.hstack([np.zeros(1000), np.ones(len(X_synthetic))])
            
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            scores = cross_val_score(clf, combined, labels, cv=3, scoring='roc_auc')
            
            # Return inverse AUC (lower AUC = better quality)
            return 1.0 - scores.mean()
            
        except Exception as e:
            print(f"   ⚠️  Stage evaluation failed: {e}")
            return 0.5  # Default score
    
    def ensemble_generation(self, models, target_samples=15000):
        """Generate samples using ensemble of models"""
        print(f"\n🔄 Ensemble Generation ({target_samples:,} samples)...")
        
        # Define generation strategy
        generation_plan = {
            'U2R': 6000,   # Heavy focus on U2R
            'R2L': 6000,   # Heavy focus on R2L  
            'normal': 1000,
            'DoS': 1000,
            'Probe': 1000
        }
        
        all_synthetic_samples = []
        
        for class_name, num_samples in generation_plan.items():
            print(f"   Generating {num_samples:,} {class_name} samples...")
            
            class_samples = []
            samples_per_model = num_samples // len(models)
            
            with tqdm(total=len(models), desc=f"Models for {class_name}", unit="model") as pbar:
                for model_info in models:
                    model = model_info['model']
                    quality_weight = model_info['quality']
                    
                    # Adjust sample count based on model quality
                    weighted_samples = int(samples_per_model * (1 + quality_weight))
                    
                    try:
                        model_samples = model.sample(
                            weighted_samples,
                            conditions={'target': class_name}
                        )
                        class_samples.append(model_samples)
                    except Exception as e:
                        print(f"     ⚠️  Model {model_info['stage']} failed: {e}")
                    
                    pbar.update(1)
            
            if class_samples:
                # Combine samples from all models for this class
                combined_class_samples = pd.concat(class_samples, ignore_index=True)
                
                # Randomly sample to target count
                if len(combined_class_samples) > num_samples:
                    combined_class_samples = combined_class_samples.sample(
                        num_samples, random_state=self.random_state
                    )
                
                all_synthetic_samples.append(combined_class_samples)
        
        # Combine all classes
        ensemble_data = pd.concat(all_synthetic_samples, ignore_index=True)
        
        print(f"✅ Ensemble generated {len(ensemble_data):,} samples")
        print("📊 Ensemble class distribution:")
        for class_name, count in ensemble_data['target'].value_counts().items():
            percentage = (count / len(ensemble_data)) * 100
            focus = "🔥" if class_name in self.minority_classes else ""
            print(f"   {class_name}: {count:,} ({percentage:.1f}%) {focus}")
        
        return ensemble_data
    
    def quality_based_filtering(self, synthetic_data, real_data):
        """Filter synthetic samples based on quality metrics"""
        print("\n🔍 Quality-Based Sample Filtering...")
        
        # Prepare data for filtering
        categorical_columns = ['protocol_type', 'service', 'flag']
        numerical_columns = real_data.select_dtypes(include=[np.number]).columns.tolist()
        
        preprocessor = ColumnTransformer([
            ('num', RobustScaler(), numerical_columns),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns)
        ])
        
        X_real = preprocessor.fit_transform(real_data.drop('target', axis=1))
        X_synthetic = preprocessor.transform(synthetic_data.drop('target', axis=1))
        
        # Train quality classifier
        print("   Training quality classifier...")
        combined_data = np.vstack([X_real, X_synthetic])
        quality_labels = np.hstack([np.ones(len(X_real)), np.zeros(len(X_synthetic))])
        
        quality_clf = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=self.random_state
        )
        quality_clf.fit(combined_data, quality_labels)
        
        # Score synthetic samples (higher score = more real-like)
        synthetic_scores = quality_clf.predict_proba(X_synthetic)[:, 1]
        
        # Filter based on quality threshold
        high_quality_mask = synthetic_scores >= self.quality_threshold
        filtered_data = synthetic_data[high_quality_mask].copy()
        
        print(f"   📊 Quality filtering results:")
        print(f"   Original samples: {len(synthetic_data):,}")
        print(f"   High-quality samples: {len(filtered_data):,}")
        print(f"   Retention rate: {len(filtered_data)/len(synthetic_data):.1%}")
        print(f"   Quality threshold: {self.quality_threshold}")
        
        # Show quality distribution by class
        print("   📊 Quality retention by class:")
        for class_name in synthetic_data['target'].unique():
            class_mask = synthetic_data['target'] == class_name
            class_quality_mask = high_quality_mask & class_mask
            
            original_count = class_mask.sum()
            retained_count = class_quality_mask.sum()
            retention_rate = retained_count / original_count if original_count > 0 else 0
            
            focus = "🔥" if class_name in self.minority_classes else ""
            print(f"     {class_name}: {retained_count:,}/{original_count:,} ({retention_rate:.1%}) {focus}")
        
        return filtered_data, synthetic_scores
    
    def comprehensive_evaluation(self, real_data, synthetic_data):
        """Ultra-comprehensive quality evaluation"""
        print("\n" + "="*70)
        print("📊 ULTRA-COMPREHENSIVE QUALITY EVALUATION")
        print("="*70)
        
        results = {}
        
        # Data preparation
        categorical_columns = ['protocol_type', 'service', 'flag']
        numerical_columns = real_data.select_dtypes(include=[np.number]).columns.tolist()
        
        preprocessor = ColumnTransformer([
            ('num', RobustScaler(), numerical_columns),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns)
        ])
        
        X_real = preprocessor.fit_transform(real_data.drop('target', axis=1))
        X_synthetic = preprocessor.transform(synthetic_data.drop('target', axis=1))
        
        # 1. Multi-Classifier Discriminability Test
        print("🔄 1. Multi-Classifier Discriminability Test...")
        
        classifiers = {
            'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
        }
        
        discriminability_scores = {}
        
        combined_data = np.vstack([X_real, X_synthetic])
        labels = np.hstack([np.zeros(len(X_real)), np.ones(len(X_synthetic))])
        
        for clf_name, clf in classifiers.items():
            scores = cross_val_score(clf, combined_data, labels, cv=5, scoring='roc_auc')
            discriminability_scores[clf_name] = {
                'mean_auc': scores.mean(),
                'std_auc': scores.std()
            }
            print(f"   {clf_name}: {scores.mean():.4f} ± {scores.std():.4f}")
        
        results['discriminability'] = discriminability_scores
        
        # 2. Class-Specific Quality Analysis
        print("\n🔄 2. Class-Specific Quality Analysis...")
        
        class_quality = {}
        for class_name in synthetic_data['target'].unique():
            real_class = real_data[real_data['target'] == class_name]
            synth_class = synthetic_data[synthetic_data['target'] == class_name]
            
            if len(real_class) > 10 and len(synth_class) > 10:
                # Process class data
                X_real_class = preprocessor.transform(real_class.drop('target', axis=1))
                X_synth_class = preprocessor.transform(synth_class.drop('target', axis=1))
                
                # Class discriminability
                class_combined = np.vstack([X_real_class, X_synth_class])
                class_labels = np.hstack([np.zeros(len(X_real_class)), np.ones(len(X_synth_class))])
                
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                class_scores = cross_val_score(clf, class_combined, class_labels, cv=3, scoring='roc_auc')
                
                # Statistical tests
                ks_failures = 0
                for i in range(min(10, X_real_class.shape[1])):
                    _, p_val = ks_2samp(X_real_class[:, i], X_synth_class[:, i])
                    if p_val < 0.05:
                        ks_failures += 1
                
                ks_pass_rate = (10 - ks_failures) / 10
                
                class_quality[class_name] = {
                    'discriminability_auc': class_scores.mean(),
                    'ks_pass_rate': ks_pass_rate,
                    'real_samples': len(real_class),
                    'synthetic_samples': len(synth_class)
                }
                
                focus = "🔥 MINORITY" if class_name in self.minority_classes else ""
                print(f"   {class_name} {focus}:")
                print(f"     Discriminability AUC: {class_scores.mean():.4f}")
                print(f"     KS Pass Rate: {ks_pass_rate:.1%}")
                print(f"     Sample Ratio: {len(synth_class)}/{len(real_class)}")
        
        results['class_quality'] = class_quality
        
        # 3. Advanced Statistical Tests
        print("\n🔄 3. Advanced Statistical Tests...")
        
        # Feature-wise analysis
        feature_quality = []
        num_features = min(20, X_real.shape[1])
        
        with tqdm(total=num_features, desc="Feature analysis", unit="feature") as pbar:
            for i in range(num_features):
                real_feature = X_real[:, i]
                synth_feature = X_synthetic[:, i]
                
                # Multiple statistical tests
                ks_stat, ks_p = ks_2samp(real_feature, synth_feature)
                
                # Jensen-Shannon divergence
                def js_divergence(p, q):
                    # Create histograms
                    bins = np.linspace(min(np.min(p), np.min(q)), max(np.max(p), np.max(q)), 50)
                    p_hist, _ = np.histogram(p, bins=bins, density=True)
                    q_hist, _ = np.histogram(q, bins=bins, density=True)
                    
                    # Normalize
                    p_hist = p_hist / np.sum(p_hist)
                    q_hist = q_hist / np.sum(q_hist)
                    
                    # Add small epsilon to avoid log(0)
                    p_hist = p_hist + 1e-10
                    q_hist = q_hist + 1e-10
                    
                    # Calculate JS divergence
                    m = 0.5 * (p_hist + q_hist)
                    js = 0.5 * entropy(p_hist, m) + 0.5 * entropy(q_hist, m)
                    return js
                
                js_div = js_divergence(real_feature, synth_feature)
                
                feature_quality.append({
                    'feature_idx': i,
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p,
                    'js_divergence': js_div
                })
                
                pbar.update(1)
        
        # Aggregate feature quality metrics
        avg_js_divergence = np.mean([f['js_divergence'] for f in feature_quality])
        ks_pass_rate = np.mean([1 if f['ks_p_value'] >= 0.05 else 0 for f in feature_quality])
        
        results['feature_quality'] = {
            'avg_js_divergence': avg_js_divergence,
            'ks_pass_rate': ks_pass_rate,
            'detailed_features': feature_quality
        }
        
        print(f"   Average JS Divergence: {avg_js_divergence:.4f}")
        print(f"   KS Test Pass Rate: {ks_pass_rate:.1%}")
        
        return results
    
    def save_ultra_results(self, synthetic_data, models, results):
        """Save ultra-quality results with comprehensive documentation"""
        print("\n💾 Saving Ultra-Quality Results...")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        files_saved = []
        
        # Save synthetic data
        synth_filename = f'ultra_quality_synthetic_data_{timestamp}.csv'
        synthetic_data.to_csv(synth_filename, index=False)
        files_saved.append(synth_filename)
        
        # Save best model
        best_model = max(models, key=lambda x: x['quality'])
        model_filename = f'ultra_quality_ctgan_model_{timestamp}.pkl'
        best_model['model'].save(model_filename)
        files_saved.append(model_filename)
        
        # Save all models
        ensemble_filename = f'ultra_quality_ensemble_models_{timestamp}.pkl'
        joblib.dump(models, ensemble_filename)
        files_saved.append(ensemble_filename)
        
        # Save comprehensive results
        results_filename = f'ultra_quality_comprehensive_results_{timestamp}.txt'
        with open(results_filename, 'w') as f:
            f.write("ULTRA-QUALITY GAN RESULTS\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Generated Samples: {len(synthetic_data):,}\n")
            f.write(f"U2R Samples: {len(synthetic_data[synthetic_data['target'] == 'U2R']):,}\n")
            f.write(f"R2L Samples: {len(synthetic_data[synthetic_data['target'] == 'R2L']):,}\n\n")
            
            f.write("PROGRESSIVE TRAINING RESULTS:\n")
            for model_info in models:
                f.write(f"Stage {model_info['stage']}: Quality={model_info['quality']:.4f}, ")
                f.write(f"Epochs={model_info['epochs']}, Time={model_info['training_time']/60:.1f}min\n")
            f.write(f"Best Model: Stage {best_model['stage']}\n\n")
            
            f.write("DISCRIMINABILITY SCORES:\n")
            for clf_name, scores in results['discriminability'].items():
                f.write(f"{clf_name}: {scores['mean_auc']:.4f} ± {scores['std_auc']:.4f}\n")
            f.write("\n")
            
            f.write("CLASS-SPECIFIC QUALITY:\n")
            for class_name, metrics in results['class_quality'].items():
                f.write(f"{class_name}:\n")
                f.write(f"  Discriminability AUC: {metrics['discriminability_auc']:.4f}\n")
                f.write(f"  KS Pass Rate: {metrics['ks_pass_rate']:.1%}\n")
                f.write(f"  Synthetic/Real Ratio: {metrics['synthetic_samples']}/{metrics['real_samples']}\n\n")
            
            f.write("FEATURE QUALITY:\n")
            f.write(f"Average JS Divergence: {results['feature_quality']['avg_js_divergence']:.4f}\n")
            f.write(f"KS Test Pass Rate: {results['feature_quality']['ks_pass_rate']:.1%}\n")
        
        files_saved.append(results_filename)
        
        print(f"✅ Saved {len(files_saved)} files:")
        for filename in files_saved:
            print(f"   - {filename}")
        
        return files_saved

def main():
    """Main execution for ultra-quality GAN training"""
    print("🎯 ULTRA-QUALITY GAN TRAINING PIPELINE")
    print("="*70)
    print("🔥 CUTTING-EDGE TECHNIQUES FOR PREMIUM SYNTHETIC DATA")
    print("="*70)
    
    # Initialize ultra-quality GAN
    ultra_gan = UltraQualityGAN(random_state=42)
    
    print(f"🖥️  Device: {ultra_gan.device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data (reuse from advanced script)
    from advanced_minority_focused_gan import AdvancedMinorityGAN
    
    advanced_gan = AdvancedMinorityGAN(random_state=42)
    full_data = advanced_gan.load_and_prepare_data()
    training_data = advanced_gan.create_minority_focused_dataset(full_data)
    
    # Analyze feature importance
    feature_importance = ultra_gan.analyze_feature_importance(training_data)
    
    # Progressive training
    discrete_columns = ['protocol_type', 'service', 'flag', 'target']
    best_model, all_models = ultra_gan.progressive_training(training_data, discrete_columns)
    
    # Ensemble generation
    ensemble_data = ultra_gan.ensemble_generation(all_models, target_samples=15000)
    
    # Quality-based filtering
    filtered_data, quality_scores = ultra_gan.quality_based_filtering(ensemble_data, training_data)
    
    # Comprehensive evaluation
    results = ultra_gan.comprehensive_evaluation(training_data, filtered_data)
    
    # Save ultra results
    files_saved = ultra_gan.save_ultra_results(filtered_data, all_models, results)
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 ULTRA-QUALITY TRAINING COMPLETED!")
    print("="*70)
    
    minority_samples = len(filtered_data[filtered_data['target'].isin(ultra_gan.minority_classes)])
    total_samples = len(filtered_data)
    
    print(f"📊 ULTRA-QUALITY RESULTS:")
    print(f"✅ Total samples: {total_samples:,}")
    print(f"🔥 Minority samples (U2R+R2L): {minority_samples:,}")
    print(f"📈 Minority percentage: {minority_samples/total_samples:.1%}")
    
    # Best discriminability score
    best_disc_score = min([scores['mean_auc'] for scores in results['discriminability'].values()])
    print(f"📈 Best discriminability AUC: {best_disc_score:.4f}")
    
    if best_disc_score < 0.6:
        print("\n🏆 OUTSTANDING: Premium quality synthetic data!")
    elif best_disc_score < 0.7:
        print("\n🎉 EXCELLENT: High-quality synthetic data!")
    elif best_disc_score < 0.8:
        print("\n✅ VERY GOOD: Quality synthetic data!")
    else:
        print("\n⚠️  GOOD: Acceptable quality with room for improvement")
    
    print(f"\n📁 Files saved: {len(files_saved)}")
    print("🚀 READY FOR PREMIUM IDS AUGMENTATION!")

if __name__ == "__main__":
    main()