import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# Optional imports with fallbacks
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm not available. Progress bars will be disabled.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available. Plotting will be disabled.")

try:
    from sklearn.metrics import confusion_matrix, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available. Detailed metrics will be limited.")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available. Some analysis features will be limited.")

# Import necessary functions and classes from train.py
from train import VGG6, cifar10_loaders, set_seed, get_activation

# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

def load_model(model_path, device):
    """Load a trained model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model configuration from saved args
    if 'args' in checkpoint:
        args = checkpoint['args']
        activation = get_activation(args.get('activation', 'relu'))
        batch_norm = not args.get('no_bn', False)
        dropout = args.get('dropout', 0.0)
    else:
        # Default configuration if args not available
        activation = get_activation('relu')
        batch_norm = True
        dropout = 0.0
    
    # Create model with same configuration
    model = VGG6(
        num_classes=10, 
        batch_norm=batch_norm, 
        activation=activation, 
        dropout=dropout
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Print model info
    val_acc = checkpoint.get('val_acc', 'Unknown')
    print(f"Loaded model with validation accuracy: {val_acc:.2f}%" if isinstance(val_acc, float) else f"Loaded model (val_acc: {val_acc})")
    
    return model, checkpoint

def evaluate_model(model, test_loader, device, verbose=True):
    """Evaluate model on test dataset"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        if verbose and HAS_TQDM:
            test_loader = tqdm(test_loader, desc="Testing")
        elif verbose:
            print("Testing model...")
        
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
    
    accuracy = 100 * correct / total
    
    if verbose:
        print(f'\nTest Accuracy: {accuracy:.2f}% ({correct}/{total})')
    
    return accuracy, np.array(all_preds), np.array(all_targets), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix"""
    if not HAS_SKLEARN or not HAS_PLOTTING:
        print("Skipping confusion matrix plot (missing sklearn or matplotlib)")
        return
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CIFAR10_CLASSES, 
                yticklabels=CIFAR10_CLASSES)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()

def plot_class_accuracies(y_true, y_pred, save_path=None):
    """Plot per-class accuracies"""
    if not HAS_PLOTTING:
        print("Skipping class accuracies plot (missing matplotlib)")
        # Still calculate accuracies for return
        class_accuracies = []
        for i in range(len(CIFAR10_CLASSES)):
            class_mask = (y_true == i)
            if class_mask.sum() > 0:
                class_acc = (y_pred[class_mask] == y_true[class_mask]).mean() * 100
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0)
        return class_accuracies
    
    # Calculate per-class accuracy
    class_accuracies = []
    for i in range(len(CIFAR10_CLASSES)):
        class_mask = (y_true == i)
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == y_true[class_mask]).mean() * 100
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(CIFAR10_CLASSES, class_accuracies, color='skyblue', edgecolor='navy')
    plt.title('Per-Class Test Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, class_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class accuracies plot saved to: {save_path}")
    
    plt.show()
    
    return class_accuracies

def print_classification_report(y_true, y_pred):
    """Print detailed classification report"""
    if not HAS_SKLEARN:
        print("\nSkipping detailed classification report (sklearn not available)")
        # Calculate basic per-class accuracy manually
        print("\nBasic Per-Class Accuracy:")
        print("=" * 30)
        for i, class_name in enumerate(CIFAR10_CLASSES):
            class_mask = (y_true == i)
            if class_mask.sum() > 0:
                class_acc = (y_pred[class_mask] == y_true[class_mask]).mean() * 100
                print(f"{class_name:>10}: {class_acc:5.1f}%")
        return
    
    print("\nDetailed Classification Report:")
    print("=" * 50)
    report = classification_report(y_true, y_pred, target_names=CIFAR10_CLASSES, digits=4)
    print(report)

def analyze_predictions(y_true, y_pred, probabilities, top_k=5):
    """Analyze model predictions and find most/least confident predictions"""
    # Calculate confidence (max probability)
    confidences = np.max(probabilities, axis=1)
    
    # Find correct and incorrect predictions
    correct_mask = (y_true == y_pred)
    
    print(f"\nPrediction Analysis:")
    print("=" * 30)
    print(f"Total samples: {len(y_true)}")
    print(f"Correct predictions: {correct_mask.sum()} ({correct_mask.mean()*100:.2f}%)")
    print(f"Incorrect predictions: {(~correct_mask).sum()} ({(~correct_mask).mean()*100:.2f}%)")
    print(f"Average confidence: {confidences.mean():.4f}")
    print(f"Average confidence (correct): {confidences[correct_mask].mean():.4f}")
    print(f"Average confidence (incorrect): {confidences[~correct_mask].mean():.4f}")
    
    # Most confident correct predictions
    correct_confidences = confidences[correct_mask]
    correct_indices = np.where(correct_mask)[0]
    most_confident_correct = correct_indices[np.argsort(correct_confidences)[-top_k:]]
    
    print(f"\nTop {top_k} most confident CORRECT predictions:")
    for idx in reversed(most_confident_correct):
        print(f"  Sample {idx}: {CIFAR10_CLASSES[y_true[idx]]} -> {CIFAR10_CLASSES[y_pred[idx]]} "
              f"(confidence: {confidences[idx]:.4f})")
    
    # Most confident incorrect predictions
    if (~correct_mask).sum() > 0:
        incorrect_confidences = confidences[~correct_mask]
        incorrect_indices = np.where(~correct_mask)[0]
        most_confident_incorrect = incorrect_indices[np.argsort(incorrect_confidences)[-min(top_k, len(incorrect_indices)):]]
        
        print(f"\nTop {min(top_k, len(incorrect_indices))} most confident INCORRECT predictions:")
        for idx in reversed(most_confident_incorrect):
            print(f"  Sample {idx}: {CIFAR10_CLASSES[y_true[idx]]} -> {CIFAR10_CLASSES[y_pred[idx]]} "
                  f"(confidence: {confidences[idx]:.4f})")

def main():
    parser = argparse.ArgumentParser(description='Test VGG6 model on CIFAR-10')
    parser.add_argument('--model_path', type=str, default='./results/baseline/best.pt',
                        help='Path to the trained model checkpoint')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Path to CIFAR-10 dataset')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for testing')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save plots to files')
    parser.add_argument('--plot_dir', type=str, default='./test_results',
                        help='Directory to save plots')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip plotting visualizations')
    parser.add_argument('--detailed_analysis', action='store_true',
                        help='Perform detailed prediction analysis')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Available model files:")
        for root, dirs, files in os.walk("./results"):
            for file in files:
                if file.endswith('.pt'):
                    print(f"  {os.path.join(root, file)}")
        return
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model, checkpoint = load_model(args.model_path, device)
    
    # Load test data
    print(f"Loading CIFAR-10 test data from: {args.data_root}")
    _, test_loader = cifar10_loaders(args.data_root, args.batch_size, use_cutout=False, cutout_length=16)
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    test_accuracy, predictions, targets, probabilities = evaluate_model(model, test_loader, device)
    
    # Print classification report
    print_classification_report(targets, predictions)
    
    # Detailed analysis
    if args.detailed_analysis:
        analyze_predictions(targets, predictions, probabilities)
    
    # Create plot directory if saving plots
    if args.save_plots:
        os.makedirs(args.plot_dir, exist_ok=True)
    
    # Generate visualizations
    if not args.no_plots and HAS_PLOTTING:
        print("\nGenerating visualizations...")
        
        # Confusion matrix
        cm_path = os.path.join(args.plot_dir, 'confusion_matrix.png') if args.save_plots else None
        plot_confusion_matrix(targets, predictions, cm_path)
        
        # Per-class accuracies
        acc_path = os.path.join(args.plot_dir, 'class_accuracies.png') if args.save_plots else None
        class_accuracies = plot_class_accuracies(targets, predictions, acc_path)
        
        # Print class accuracies
        print("\nPer-class accuracies:")
        for class_name, acc in zip(CIFAR10_CLASSES, class_accuracies):
            print(f"  {class_name:>10}: {acc:5.1f}%")
    elif not args.no_plots:
        print("\nSkipping visualizations (matplotlib not available)")
        # Still calculate class accuracies without plotting
        class_accuracies = []
        for i in range(len(CIFAR10_CLASSES)):
            class_mask = (targets == i)
            if class_mask.sum() > 0:
                class_acc = (predictions[class_mask] == targets[class_mask]).mean() * 100
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0)
        
        # Print class accuracies
        print("\nPer-class accuracies:")
        for class_name, acc in zip(CIFAR10_CLASSES, class_accuracies):
            print(f"  {class_name:>10}: {acc:5.1f}%")
    else:
        class_accuracies = []  # Empty list for consistency
    
    # Save results summary
    if args.save_plots:
        results_summary = {
            'model_path': args.model_path,
            'test_accuracy': test_accuracy,
            'total_samples': len(targets),
            'correct_predictions': (predictions == targets).sum(),
            'class_accuracies': dict(zip(CIFAR10_CLASSES, class_accuracies)) if class_accuracies else {}
        }
        
        summary_path = os.path.join(args.plot_dir, 'test_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Test Results Summary\n")
            f.write(f"==================\n\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Test Accuracy: {test_accuracy:.2f}%\n")
            f.write(f"Total Samples: {len(targets)}\n")
            f.write(f"Correct Predictions: {(predictions == targets).sum()}\n\n")
            
            if class_accuracies:
                f.write("Per-class Accuracies:\n")
                for class_name, acc in zip(CIFAR10_CLASSES, class_accuracies):
                    f.write(f"  {class_name}: {acc:.1f}%\n")
        
        print(f"\nResults summary saved to: {summary_path}")
    
    print(f"\n🎯 Final Test Accuracy: {test_accuracy:.2f}%")

if __name__ == '__main__':
    main()
