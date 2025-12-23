# GAN-IDS
A clean and modular process for generating synthetic data using GANs for Intrusion Detection Systems. This pipeline saves checkpoints and generated outputs directly to GitHub.

## 🚀 Latest Updates

### Training 2 - Improved Pipeline (December 2024)
**Major improvements with GPU acceleration and comprehensive progress tracking!**

- ✅ **10x More Training**: Increased from 20 to 200 epochs
- ✅ **GPU Acceleration**: CUDA support for NVIDIA GPUs  
- ✅ **Real-time Progress**: Comprehensive progress bars and time estimation
- ✅ **Better Quality**: KS test pass rate improved from 50% → 70%
- ✅ **Production Ready**: Robust error handling and evaluation metrics

**Quick Start:**
```bash
cd "training 2"
python final_improved_gan.py  # Production training
```

See [`training 2/README.md`](training%202/README.md) for detailed documentation.

## 📁 Project Structure

- **`training 2/`** - **Latest improved training pipeline** (recommended)
  - Production-quality GAN training with GPU acceleration
  - Comprehensive progress tracking and evaluation
  - 200 epochs training with optimized hyperparameters
  
- **Root directory** - Original implementation
  - Basic CTGAN training (20 epochs)
  - Legacy code for reference

## 🎯 Key Features

- **NSL-KDD Dataset**: Network intrusion detection data
- **CTGAN Implementation**: Conditional Tabular GAN for synthetic data generation
- **GPU Acceleration**: CUDA support for faster training
- **Progress Tracking**: Real-time training progress and metrics
- **Quality Evaluation**: Statistical tests and discriminability analysis
- **Balanced Generation**: Proper handling of imbalanced attack classes
