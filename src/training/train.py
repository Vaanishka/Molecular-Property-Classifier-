"""
Training script for BBB penetration prediction model.

What happens:
1. Load data (tokenized SMILES + labels)
2. For each epoch:
   - Train on all molecules (learn patterns)
   - Validate (check if learning is working)
   - Save best model
3. Early stopping (stop if no improvement)
4. Save results
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import yaml
import argparse
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

from src.models import SMILESXModel
from src.data import download_bbbp_dataset, load_bbbp_data, create_dataloaders
from src.utils import SMILESTokenizer


class Trainer:
    """
    Manages the training loop.
    
    What it does:
    - One epoch training (forward pass, backward pass, gradient update)
    - Validation (check accuracy without updating weights)
    - Checkpointing (save best model)
    - Early stopping (stop if no improvement)
    - Logging (track metrics)
    """

    def __init__(self, config: Dict, model, train_loader, val_loader, device):
        """
        Args:
            config: Configuration dictionary from YAML
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: torch.device (cpu or cuda)
        """
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss function (binary classification)
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer (learns by updating weights)
        self.optimizer = self._get_optimizer()

        # Learning rate scheduler (adjust learning rate over time)
        self.scheduler = self._get_scheduler()

        # Tracking best performance
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_auc = 0.0
        self.epochs_without_improvement = 0

        # Logging
        log_dir = config['logging']['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        self.writer = None  # Disabled for CPU (too slow)

        # Checkpoint directory
        self.checkpoint_dir = config['logging']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _get_optimizer(self):
        """Create optimizer (updates weights during training)."""
        optimizer_name = self.config['training']['optimizer'].lower()
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']

        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _get_scheduler(self):
        """Create learning rate scheduler (adjusts learning rate)."""
        scheduler_type = self.config['training']['scheduler'].lower()
        num_epochs = self.config['training']['num_epochs']

        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        elif scheduler_type == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        What happens:
        1. Forward pass (prediction)
        2. Calculate loss (how wrong we are)
        3. Backward pass (calculate gradients)
        4. Update weights (learn)
        5. Track metrics
        
        Returns:
            Dictionary with loss, accuracy, AUC
        """
        self.model.train()  # Set to training mode (enable dropout, etc.)
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")

        for batch in pbar:
            # Get batch data
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass (predict)
            logits, _ = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)

            # Backward pass (calculate gradients)
            self.optimizer.zero_grad()  # Clear old gradients
            loss.backward()  # Backpropagation

            # Gradient clipping (prevent exploding gradients)
            if self.config['training']['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )

            # Update weights
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'auc': auc
        }

    @torch.no_grad()  # Disable gradient calculation (faster, less memory)
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set (check if learning is working).
        
        Similar to train_epoch but:
        - No weight updates
        - No dropout (use learned statistics)
        - More detailed metrics
        
        Returns:
            Dictionary with loss, accuracy, AUC, precision, recall, F1
        """
        self.model.eval()  # Set to evaluation mode
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")

        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass (no backward)
            logits, _ = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)

            # Track metrics
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint (snapshot of weights).
        
        Args:
            metrics: Dictionary with validation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),  # Weights
            'optimizer_state_dict': self.optimizer.state_dict(),  # Optimizer state
            'metrics': metrics,
            'config': self.config
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_epoch_{self.current_epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"✓ New best model! (AUC: {metrics['auc']:.4f})")

    def train(self):
        """
        Main training loop.
        
        For each epoch:
        1. Train on training data
        2. Validate on validation data
        3. Check if improved (for early stopping)
        4. Save checkpoint
        5. Adjust learning rate
        """
        num_epochs = self.config['training']['num_epochs']
        patience = self.config['training']['early_stopping_patience']

        print(f"\n{'='*70}")
        print(f"Starting Training")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training epochs: {num_epochs}")
        print(f"Early stopping patience: {patience}")
        print(f"{'='*70}\n")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Print metrics
            print(f"\n📊 Epoch {epoch + 1}/{num_epochs}")
            print(f"   Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, "
                  f"AUC: {train_metrics['auc']:.4f}")
            print(f"   Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"AUC: {val_metrics['auc']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}")
            print(f"   LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Check for improvement
            is_best = val_metrics['auc'] > self.best_val_auc
            if is_best:
                self.best_val_auc = val_metrics['auc']
                self.best_val_loss = val_metrics['loss']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint
            if (epoch + 1) % self.config['logging']['save_every_n_epochs'] == 0 or is_best:
                self.save_checkpoint(val_metrics, is_best)

            # Early stopping
            if self.epochs_without_improvement >= patience:
                print(f"\n⏹️  Early stopping triggered after {epoch + 1} epochs")
                print(f"   Best validation AUC: {self.best_val_auc:.4f}")
                break

            print("-" * 70)

        print(f"\n{'='*70}")
        print(f"✓ Training Complete!")
        print(f"   Best validation AUC: {self.best_val_auc:.4f}")
        print(f"   Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*70}\n")


def main():
    """
    Main entry point.
    
    What it does:
    1. Parse command line arguments
    2. Load configuration
    3. Download/load data
    4. Create tokenizer and dataloaders
    5. Initialize model
    6. Train
    7. Save results
    """
    parser = argparse.ArgumentParser(description='Train BBB penetration prediction model')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                        help='Path to config file')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    args = parser.parse_args()

    # Load config
    print("📂 Loading configuration...")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f"✓ Config loaded from {args.config}\n")

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}\n")

    # Download and load data
    print("📥 Loading dataset...")
    data_path = download_bbbp_dataset(config['data']['data_dir'])
    train_df, val_df, test_df = load_bbbp_data(
        data_path,
        test_size=config['data']['test_split'],
        val_size=config['data']['val_split'],
        random_seed=config['data']['random_seed']
    )

    # Create tokenizer
    print("\n🔤 Initializing tokenizer...")
    tokenizer = SMILESTokenizer(max_length=config['data']['max_seq_length'])
    print(f"✓ Vocabulary size: {tokenizer.vocab_size}\n")

    # Create dataloaders
    print("🔄 Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df,
        tokenizer,
        batch_size=config['training']['batch_size'],
        num_workers=config['hardware']['num_workers'],
        augment_train=config['augmentation']['enabled'],
        augmentation_factor=config['augmentation']['augmentation_factor']
    )

    # Initialize model
    print("\n🧠 Initializing model...")
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
    print(f"✓ Model created\n")

    # Create trainer and train
    trainer = Trainer(config, model, train_loader, val_loader, device)
    trainer.train()

    # Save results
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)

    results = {
        'best_val_auc': float(trainer.best_val_auc),
        'best_val_loss': float(trainer.best_val_loss),
        'total_epochs': trainer.current_epoch + 1,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }

    results_path = os.path.join(results_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"💾 Results saved to {results_path}")


if __name__ == "__main__":
    main()