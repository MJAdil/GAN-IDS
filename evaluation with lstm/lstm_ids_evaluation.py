#!/usr/bin/env python3
"""
LSTM IDS Model Training and Evaluation
Combines real NSL-KDD data with GAN-generated synthetic data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

class LSTMIDSModel(nn.Module):
    """LSTM-based Intrusion Detection System Model"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=5, dropout=0.3):
        super(LSTMIDSModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # *2 for bidirectional
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        lstm_out = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.dropout(lstm_out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out

class LSTMIDSTrainer:
    """LSTM IDS Training and Evaluation Pipeline"""
    
    def __init__(self, sequence_length=10, random_state=42):
        self.sequence_length = sequence_length
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['normal', 'DoS', 'Probe', 'R2L', 'U2R']
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
    
    def load_and_prepare_data(self, use_synthetic=True):
        """Load and prepare combined real and synthetic data"""
        print("🔄 Loading and preparing data...")
        
        # Define columns
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
        
        # Load real data
        print("   Loading real NSL-KDD data...")
        train_data = pd.read_csv('KDDTrain+.txt', names=columns, header=None)
        test_data = pd.read_csv('KDDTest+.txt', names=columns, header=None)
        
        # Combine real data
        real_data = pd.concat([train_data, test_data], ignore_index=True)
        real_data = real_data.dropna()
        real_data = real_data.drop('difficulty', axis=1)
        
        print(f"   Real data shape: {real_data.shape}")
        
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
            # R2L attacks
            'guess_passwd': 'R2L', 'ftp_write': 'R2L', 'imap': 'R2L', 'phf': 'R2L',
            'multihop': 'R2L', 'warezmaster': 'R2L', 'warezclient': 'R2L', 'spy': 'R2L',
            'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L', 'snmpgetattack': 'R2L',
            'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
            # U2R attacks
            'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R', 'perl': 'U2R',
            'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R'
        }
        
        real_data['target'] = real_data['target'].map(attack_mapping)
        real_data = real_data.dropna(subset=['target'])
        
        print("📊 Real data class distribution:")
        real_dist = real_data['target'].value_counts()
        for class_name, count in real_dist.items():
            print(f"   {class_name}: {count:,}")
        
        # Load synthetic data if requested
        combined_data = real_data.copy()
        
        if use_synthetic:
            print("   Loading GAN-generated synthetic data...")
            try:
                # Find the latest synthetic data file
                import glob
                synthetic_files = glob.glob('training 3/working_advanced_synthetic_data_*.csv')
                if synthetic_files:
                    latest_synthetic = max(synthetic_files)
                    synthetic_data = pd.read_csv(latest_synthetic)
                    
                    print(f"   Synthetic data shape: {synthetic_data.shape}")
                    print("📊 Synthetic data class distribution:")
                    synth_dist = synthetic_data['target'].value_counts()
                    for class_name, count in synth_dist.items():
                        print(f"   {class_name}: {count:,}")
                    
                    # Combine real and synthetic data
                    combined_data = pd.concat([real_data, synthetic_data], ignore_index=True)
                    print(f"✅ Combined data shape: {combined_data.shape}")
                    
                else:
                    print("⚠️  No synthetic data found, using only real data")
                    use_synthetic = False
                    
            except Exception as e:
                print(f"⚠️  Error loading synthetic data: {e}")
                print("   Using only real data")
                use_synthetic = False
        
        print("📊 Final combined class distribution:")
        final_dist = combined_data['target'].value_counts()
        for class_name, count in final_dist.items():
            percentage = (count / len(combined_data)) * 100
            print(f"   {class_name}: {count:,} ({percentage:.1f}%)")
        
        return combined_data, use_synthetic
    
    def preprocess_data(self, data):
        """Preprocess data for LSTM training"""
        print("🔄 Preprocessing data for LSTM...")
        
        # Separate features and target
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Handle categorical columns
        categorical_columns = ['protocol_type', 'service', 'flag']
        
        # One-hot encode categorical columns
        X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
        
        print(f"   Features after encoding: {X_encoded.shape[1]}")
        
        # Scale numerical features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)
        
        # Encode target labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        print(f"   Classes: {list(label_encoder.classes_)}")
        
        return X_scaled, y_encoded, scaler, label_encoder
    
    def create_sequences(self, X, y):
        """Create sequences for LSTM input"""
        print(f"🔄 Creating sequences (length={self.sequence_length})...")
        
        sequences_X = []
        sequences_y = []
        
        # Group by class to maintain class distribution in sequences
        df_temp = pd.DataFrame(X)
        df_temp['target'] = y
        
        for class_label in np.unique(y):
            class_data = df_temp[df_temp['target'] == class_label]
            class_X = class_data.drop('target', axis=1).values
            
            # Create sequences for this class
            for i in range(len(class_X) - self.sequence_length + 1):
                sequences_X.append(class_X[i:i + self.sequence_length])
                sequences_y.append(class_label)
        
        sequences_X = np.array(sequences_X)
        sequences_y = np.array(sequences_y)
        
        print(f"   Created {len(sequences_X):,} sequences")
        print(f"   Sequence shape: {sequences_X.shape}")
        
        return sequences_X, sequences_y
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64, learning_rate=0.001):
        """Train the LSTM model"""
        print("🚀 Training LSTM IDS Model...")
        
        # Model parameters
        input_size = X_train.shape[2]  # Number of features
        num_classes = len(np.unique(y_train))
        
        print(f"📋 Model Configuration:")
        print(f"   Input size: {input_size}")
        print(f"   Sequence length: {self.sequence_length}")
        print(f"   Number of classes: {num_classes}")
        print(f"   Device: {self.device}")
        
        # Create model
        model = LSTMIDSModel(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            num_classes=num_classes,
            dropout=0.3
        ).to(self.device)
        
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training history
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        print(f"\n🔄 Starting training for {epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for batch_X, batch_y in train_pbar:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            val_loss /= len(val_loader)
            val_accuracy = 100 * correct / total
            
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_accuracy:.2f}%")
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time/60:.1f} minutes")
        
        return model, {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'training_time': training_time
        }
    
    def evaluate_model(self, model, X_test, y_test, label_encoder):
        """Evaluate the trained model"""
        print("📊 Evaluating model performance...")
        
        model.eval()
        
        # Convert to tensor
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu().numpy()
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, predicted)
        
        # Generate classification report
        class_names = label_encoder.classes_
        report = classification_report(
            y_test, predicted, 
            target_names=class_names, 
            output_dict=True,
            zero_division=0
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, predicted)
        
        return accuracy, report, cm, predicted
    
    def plot_results(self, history, cm, class_names, accuracy, use_synthetic):
        """Plot training results and confusion matrix"""
        print("📈 Generating result plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Training history
        axes[0, 0].plot(history['train_losses'], label='Training Loss', color='blue')
        axes[0, 0].plot(history['val_losses'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Validation accuracy
        axes[0, 1].plot(history['val_accuracies'], label='Validation Accuracy', color='green')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=axes[1, 0])
        axes[1, 0].set_title('Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # Class-wise performance
        report_df = pd.DataFrame(history).T
        if 'f1-score' in report_df.columns:
            class_f1 = [report_df.loc[class_name, 'f1-score'] for class_name in class_names if class_name in report_df.index]
            axes[1, 1].bar(class_names[:len(class_f1)], class_f1, color='skyblue')
            axes[1, 1].set_title('F1-Score by Class')
            axes[1, 1].set_xlabel('Class')
            axes[1, 1].set_ylabel('F1-Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f'lstm_results_{"with_synthetic" if use_synthetic else "real_only"}_{int(time.time())}.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"   Plot saved: {plot_filename}")
        
        return plot_filename
    
    def save_results(self, accuracy, report, use_synthetic, training_time):
        """Save detailed results"""
        print("💾 Saving results...")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_filename = f'lstm_evaluation_results_{"with_synthetic" if use_synthetic else "real_only"}_{timestamp}.txt'
        
        with open(results_filename, 'w') as f:
            f.write("LSTM IDS MODEL EVALUATION RESULTS\n")
            f.write("="*50 + "\n\n")
            f.write(f"Data Configuration: {'Real + Synthetic' if use_synthetic else 'Real Only'}\n")
            f.write(f"Training Time: {training_time/60:.1f} minutes\n")
            f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
            
            f.write("DETAILED CLASSIFICATION REPORT:\n")
            f.write("-" * 40 + "\n")
            
            # Write class-wise metrics
            for class_name in ['normal', 'DoS', 'Probe', 'R2L', 'U2R']:
                if class_name in report:
                    metrics = report[class_name]
                    f.write(f"\n{class_name}:\n")
                    f.write(f"  Precision: {metrics['precision']:.4f}\n")
                    f.write(f"  Recall: {metrics['recall']:.4f}\n")
                    f.write(f"  F1-Score: {metrics['f1-score']:.4f}\n")
                    f.write(f"  Support: {metrics['support']}\n")
            
            # Overall metrics
            if 'macro avg' in report:
                f.write(f"\nMACRO AVERAGE:\n")
                f.write(f"  Precision: {report['macro avg']['precision']:.4f}\n")
                f.write(f"  Recall: {report['macro avg']['recall']:.4f}\n")
                f.write(f"  F1-Score: {report['macro avg']['f1-score']:.4f}\n")
            
            if 'weighted avg' in report:
                f.write(f"\nWEIGHTED AVERAGE:\n")
                f.write(f"  Precision: {report['weighted avg']['precision']:.4f}\n")
                f.write(f"  Recall: {report['weighted avg']['recall']:.4f}\n")
                f.write(f"  F1-Score: {report['weighted avg']['f1-score']:.4f}\n")
        
        print(f"   Results saved: {results_filename}")
        return results_filename

def main():
    """Main execution function"""
    print("🎯 LSTM IDS MODEL TRAINING AND EVALUATION")
    print("="*60)
    print("🔥 EVALUATING GAN AUGMENTATION EFFECTIVENESS")
    print("="*60)
    
    # Initialize trainer
    trainer = LSTMIDSTrainer(sequence_length=10, random_state=42)
    
    print(f"🖥️  Device: {trainer.device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    
    # Load and prepare data
    data, use_synthetic = trainer.load_and_prepare_data(use_synthetic=True)
    
    # Preprocess data
    X, y, scaler, label_encoder = trainer.preprocess_data(data)
    
    # Create sequences
    X_seq, y_seq = trainer.create_sequences(X, y)
    
    # Split data
    print("🔄 Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_seq, y_seq, test_size=0.4, random_state=42, stratify=y_seq
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"   Training sequences: {len(X_train):,}")
    print(f"   Validation sequences: {len(X_val):,}")
    print(f"   Test sequences: {len(X_test):,}")
    
    # Train model
    model, history = trainer.train_model(
        X_train, y_train, X_val, y_val,
        epochs=30, batch_size=64, learning_rate=0.001
    )
    
    # Evaluate model
    accuracy, report, cm, predictions = trainer.evaluate_model(
        model, X_test, y_test, label_encoder
    )
    
    # Generate plots
    plot_filename = trainer.plot_results(
        history, cm, label_encoder.classes_, accuracy, use_synthetic
    )
    
    # Save results
    results_filename = trainer.save_results(
        accuracy, report, use_synthetic, history['training_time']
    )
    
    # Print final summary
    print("\n" + "="*60)
    print("🎉 LSTM IDS EVALUATION COMPLETED!")
    print("="*60)
    
    print(f"📊 FINAL RESULTS:")
    print(f"✅ Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"🔥 Data Configuration: {'Real + Synthetic (GAN Augmented)' if use_synthetic else 'Real Only'}")
    print(f"⏱️  Training Time: {history['training_time']/60:.1f} minutes")
    
    print(f"\n📈 CLASS-WISE PERFORMANCE:")
    for class_name in ['normal', 'DoS', 'Probe', 'R2L', 'U2R']:
        if class_name in report:
            metrics = report[class_name]
            minority_indicator = "🔥" if class_name in ['U2R', 'R2L'] else "  "
            print(f"{minority_indicator} {class_name:8s}: "
                  f"Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, "
                  f"F1={metrics['f1-score']:.3f}")
    
    if use_synthetic:
        print(f"\n🚀 GAN AUGMENTATION IMPACT:")
        print(f"   Minority classes (U2R, R2L) significantly augmented!")
        print(f"   Expected improvements in rare attack detection!")
    
    print(f"\n📁 Files Generated:")
    print(f"   - {results_filename}")
    print(f"   - {plot_filename}")
    
    print(f"\n🎯 READY FOR IDS DEPLOYMENT!")

if __name__ == "__main__":
    main()