# VGG6 Model Testing Guide

This guide explains how to use the `test.py` script to evaluate trained VGG6 models on the CIFAR-10 dataset.

## Quick Start

### Basic Usage
```bash
# Test the baseline model
python test.py --model_path "./results/baseline/best.pt" --data_root "../../data"

# Test with custom batch size
python test.py --model_path "./results/baseline/best.pt" --data_root "../../data" --batch_size 64

# Test without plots (faster)
python test.py --model_path "./results/baseline/best.pt" --data_root "../../data" --no_plots

# Test with detailed analysis
python test.py --model_path "./results/baseline/best.pt" --data_root "../../data" --detailed_analysis
```

### Save Results
```bash
# Save plots and summary to files
python test.py --model_path "./results/baseline/best.pt" --data_root "../../data" --save_plots --plot_dir "./test_results"
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model_path` | `./results/baseline/best.pt` | Path to the trained model checkpoint |
| `--data_root` | `./data` | Path to CIFAR-10 dataset |
| `--batch_size` | `128` | Batch size for testing |
| `--save_plots` | `False` | Save plots to files |
| `--plot_dir` | `./test_results` | Directory to save plots |
| `--seed` | `42` | Random seed for reproducibility |
| `--no_plots` | `False` | Skip plotting visualizations |
| `--detailed_analysis` | `False` | Perform detailed prediction analysis |

## Features

### 1. Model Loading
- Automatically loads VGG6 model with correct configuration
- Displays validation accuracy from training
- Handles different activation functions, batch normalization settings

### 2. Test Evaluation
- Evaluates model on CIFAR-10 test set
- Displays overall accuracy
- Shows per-class accuracy breakdown

### 3. Visualizations (if matplotlib available)
- Confusion matrix heatmap
- Per-class accuracy bar chart
- Saves plots as PNG files

### 4. Detailed Analysis
- Classification report with precision, recall, F1-score
- Most confident correct/incorrect predictions
- Confidence score analysis

### 5. Results Summary
- Saves test results to text file
- Includes all metrics and per-class accuracies

## Batch Testing Multiple Models

Use the provided `run_tests.py` script to test multiple models at once:

```bash
python run_tests.py
```

This will test all available models and provide a ranked summary.

## Dependency Requirements

### Core Requirements (Essential)
- PyTorch
- NumPy
- torchvision

### Optional Dependencies (Enhanced Features)
- tqdm (progress bars)
- matplotlib + seaborn (plotting)
- scikit-learn (detailed metrics)
- pandas (data analysis)

The script will work with just the core requirements, but some features will be disabled.

## Example Output

```
Using device: cpu
Loading model from: ./results/baseline/best.pt
Loaded model with validation accuracy: 88.00%
Loading CIFAR-10 test data from: ../../data

Evaluating model on test set...
Testing model...

Test Accuracy: 87.99% (8799/10000)

Basic Per-Class Accuracy:
==============================
  airplane:  89.3%
automobile:  94.0%
      bird:  81.8%
       cat:  79.0%
      deer:  86.1%
       dog:  84.7%
      frog:  92.3%
     horse:  87.7%
      ship:  92.9%
     truck:  92.1%

🎯 Final Test Accuracy: 87.99%
```

## Available Models

The script can test any model in the `results/` directory:

- `baseline/best.pt` - Baseline configuration
- `act_*/best.pt` - Different activation functions (relu, gelu, silu, tanh, sigmoid)
- `opt_*/best.pt` - Different optimizers (sgd, adam, rmsprop, adagrad, nadam)
- `bs_*/best.pt` - Different batch sizes (64, 128, 256)
- `ep_*/best.pt` - Different epoch counts (50, 100, 150)
- `lr_*/best.pt` - Different learning rates

## Troubleshooting

### Model Not Found
```bash
Error: Model file not found at ./results/baseline/best.pt
Available model files:
  ./results/act_relu/best.pt
  ./results/act_silu/best.pt
  ...
```

### CIFAR-10 Data Not Found
Make sure the CIFAR-10 dataset is available at the specified `--data_root` path. The script will automatically download it if needed.

### Missing Dependencies
The script will show warnings for missing optional dependencies but continue working with reduced functionality.