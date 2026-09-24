"""
Evaluation script for BBB penetration prediction model.

Tests the trained model on held-out test data it has never seen.
Generates performance metrics, plots, and saves predictions.
"""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no window needed)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, classification_report
)
from tqdm import tqdm
import json
import argparse

from src.models import SMILESXModel
from src.data import load_bbbp_data, BBBDataset
from src.utils import SMILESTokenizer


class Evaluator:
    """Evaluate trained model on test data."""

    def __init__(self, model, test_loader, device):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def predict(self):
        """Run predictions on test set."""
        all_preds = []
        all_labels = []
        all_probs = []
        all_smiles = []

        print("🔮 Running predictions on test set...")
        for batch in tqdm(self.test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            logits, _ = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_smiles.extend(batch['smiles'])

        return {
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels),
            'probabilities': np.array(all_probs),
            'smiles': all_smiles
        }

    def calculate_metrics(self, results):
        """Calculate all evaluation metrics."""
        preds = results['predictions']
        labels = results['labels']
        probs = results['probabilities']

        metrics = {
            'accuracy': float(accuracy_score(labels, preds)),
            'roc_auc': float(roc_auc_score(labels, probs)),
            'precision': float(precision_score(labels, preds, zero_division=0)),
            'recall': float(recall_score(labels, preds, zero_division=0)),
            'f1': float(f1_score(labels, preds, zero_division=0)),
            'confusion_matrix': confusion_matrix(labels, preds).tolist()
        }

        return metrics

    def plot_confusion_matrix(self, results, save_path):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(results['labels'], results['predictions'])

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['BBB-', 'BBB+'],
                    yticklabels=['BBB-', 'BBB+'])
        plt.title('Confusion Matrix - BBB Penetration Prediction')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {save_path}")

    def plot_roc_curve(self, results, save_path):
        """Plot and save ROC curve."""
        fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'])
        auc = roc_auc_score(results['labels'], results['probabilities'])

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'Model (AUC = {auc:.3f})', linewidth=2, color='steelblue')
        plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - BBB Penetration Prediction')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {save_path}")

    def save_predictions(self, results, save_path):
        """Save predictions to CSV."""
        df = pd.DataFrame({
            'smiles': results['smiles'],
            'true_label': results['labels'],
            'predicted_label': results['predictions'],
            'bbb_positive_probability': results['probabilities']
        })
        df['correct'] = df['true_label'] == df['predicted_label']
        df.to_csv(save_path, index=False)
        print(f"   Saved: {save_path}")

    def evaluate(self, output_dir='./results/evaluation'):
        """Run full evaluation."""
        os.makedirs(output_dir, exist_ok=True)

        # Get predictions
        results = self.predict()

        # Calculate metrics
        metrics = self.calculate_metrics(results)

        # Print results
        print(f"\n{'='*70}")
        print(f"EVALUATION RESULTS (Test Set)")
        print(f"{'='*70}")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   ROC AUC:   {metrics['roc_auc']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1 Score:  {metrics['f1']:.4f}")
        print(f"\n   Classification Report:")
        print(classification_report(
            results['labels'],
            results['predictions'],
            target_names=['BBB-', 'BBB+']
        ))

        # Save metrics
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"   Saved: {metrics_path}")

        # Generate plots
        print(f"\n📊 Generating plots...")
        self.plot_confusion_matrix(results, os.path.join(output_dir, 'confusion_matrix.png'))
        self.plot_roc_curve(results, os.path.join(output_dir, 'roc_curve.png'))

        # Save predictions
        self.save_predictions(results, os.path.join(output_dir, 'predictions.csv'))

        print(f"\n{'='*70}")
        print(f"✓ Evaluation complete! Results in {output_dir}")
        print(f"{'='*70}")

        return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate BBB prediction model')
    parser.add_argument('--checkpoint', type=str, default='models/checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default='./data/BBBP.csv',
                        help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='./results/evaluation',
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cpu')
    print(f"🖥️  Using device: {device}")

    # Load checkpoint
    print(f"\n📂 Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint['config']
    print(f"   Trained for {checkpoint['epoch'] + 1} epochs")

    # Load data (same split as training)
    print(f"\n📖 Loading test data...")
    _, _, test_df = load_bbbp_data(
        args.data_path,
        test_size=config['data']['test_split'],
        val_size=config['data']['val_split'],
        random_seed=config['data']['random_seed']
    )

    # Create tokenizer and dataset
    tokenizer = SMILESTokenizer(max_length=config['data']['max_seq_length'])

    test_dataset = BBBDataset(
        test_df['smiles'].tolist(),
        test_df['p_np'].tolist(),
        tokenizer,
        augment=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Initialize model
    print(f"\n🧠 Loading model...")
    model = SMILESXModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        num_classes=config['model']['num_classes'],
        max_seq_length=config['data']['max_seq_length'],
        use_batch_norm=config['model']['use_batch_norm']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"   ✓ Model loaded")

    # Evaluate
    evaluator = Evaluator(model, test_loader, device)
    evaluator.evaluate(args.output_dir)


if __name__ == "__main__":
    main()