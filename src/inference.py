"""
Inference script for BBB penetration prediction.
Make predictions on new molecules.

Usage:
    # Single molecule
    python -m src.inference --smiles "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

    # From CSV
    python -m src.inference --input_csv molecules.csv --output_csv results.csv

    # Interactive mode
    python -m src.inference
"""
import torch
import numpy as np
import pandas as pd
import argparse
from typing import List, Dict

from src.models import SMILESXModel
from src.utils import SMILESTokenizer, check_bbb_rules


class BBBPredictor:
    """Predict BBB penetration for new molecules."""

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        self.device = torch.device(device)

        # Load checkpoint
        print(f"📂 Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = checkpoint['config']

        # Initialize tokenizer
        self.tokenizer = SMILESTokenizer(
            max_length=self.config['data']['max_seq_length']
        )

        # Initialize model
        self.model = SMILESXModel(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=self.config['model']['embedding_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            num_heads=self.config['model']['num_heads'],
            dropout=self.config['model']['dropout'],
            num_classes=self.config['model']['num_classes'],
            max_seq_length=self.config['data']['max_seq_length'],
            use_batch_norm=self.config['model']['use_batch_norm']
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"✓ Model loaded (trained {checkpoint['epoch'] + 1} epochs)")

    @torch.no_grad()
    def predict_single(self, smiles: str) -> Dict:
        """Predict BBB penetration for a single molecule."""
        # Tokenize
        input_ids, attention_mask = self.tokenizer.encode(smiles)

        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0).to(self.device)

        # Predict
        logits, attention_weights = self.model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)

        pred_class = torch.argmax(probs, dim=1).item()
        bbb_prob = probs[0, 1].item()

        # Check rules
        rules = check_bbb_rules(smiles)

        return {
            'smiles': smiles,
            'predicted_class': 'BBB+' if pred_class == 1 else 'BBB-',
            'bbb_positive_probability': float(bbb_prob),
            'confidence': float(max(probs[0].tolist())),
            'follows_bbb_rules': rules['overall'],
            'rule_details': rules
        }

    @torch.no_grad()
    def predict_batch(self, smiles_list: List[str], batch_size: int = 64) -> List[Dict]:
        """Predict BBB penetration for multiple molecules."""
        results = []

        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i+batch_size]

            input_ids_list = []
            attention_mask_list = []

            for smiles in batch_smiles:
                input_ids, attention_mask = self.tokenizer.encode(smiles)
                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)

            input_ids = torch.tensor(input_ids_list, dtype=torch.long).to(self.device)
            attention_mask = torch.tensor(attention_mask_list, dtype=torch.long).to(self.device)

            logits, _ = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)

            for j, smiles in enumerate(batch_smiles):
                pred_class = torch.argmax(probs[j]).item()
                bbb_prob = probs[j, 1].item()
                rules = check_bbb_rules(smiles)

                results.append({
                    'smiles': smiles,
                    'predicted_class': 'BBB+' if pred_class == 1 else 'BBB-',
                    'bbb_positive_probability': float(bbb_prob),
                    'confidence': float(max(probs[j].tolist())),
                    'follows_bbb_rules': rules['overall']
                })

        return results

    def predict_from_csv(self, input_csv: str, output_csv: str, smiles_column: str = 'smiles'):
        """Predict for molecules in a CSV file."""
        print(f"📖 Reading molecules from {input_csv}...")
        df = pd.read_csv(input_csv)

        if smiles_column not in df.columns:
            raise ValueError(f"Column '{smiles_column}' not found in CSV")

        smiles_list = df[smiles_column].tolist()
        print(f"🔮 Predicting for {len(smiles_list)} molecules...")

        results = self.predict_batch(smiles_list)

        results_df = pd.DataFrame(results)
        output_df = pd.concat([df, results_df.drop('smiles', axis=1)], axis=1)
        output_df.to_csv(output_csv, index=False)

        bbb_positive = sum(1 for r in results if r['predicted_class'] == 'BBB+')
        print(f"\n📊 Summary:")
        print(f"   BBB+: {bbb_positive} ({bbb_positive/len(results)*100:.1f}%)")
        print(f"   BBB-: {len(results) - bbb_positive} ({(len(results)-bbb_positive)/len(results)*100:.1f}%)")
        print(f"   Saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description='Predict BBB penetration')
    parser.add_argument('--checkpoint', type=str, default='models/checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--smiles', type=str, default=None,
                        help='Single SMILES string to predict')
    parser.add_argument('--input_csv', type=str, default=None,
                        help='CSV file with SMILES strings')
    parser.add_argument('--output_csv', type=str, default='predictions.csv',
                        help='Output CSV for predictions')
    parser.add_argument('--smiles_column', type=str, default='smiles',
                        help='Name of SMILES column in CSV')
    args = parser.parse_args()

    # Initialize predictor
    predictor = BBBPredictor(args.checkpoint)

    if args.smiles:
        # Single molecule
        result = predictor.predict_single(args.smiles)

        print(f"\n{'='*60}")
        print(f"PREDICTION")
        print(f"{'='*60}")
        print(f"   SMILES: {result['smiles']}")
        print(f"   Prediction: {result['predicted_class']}")
        print(f"   BBB+ Probability: {result['bbb_positive_probability']:.4f}")
        print(f"   Confidence: {result['confidence']:.4f}")
        print(f"   Follows BBB Rules: {result['follows_bbb_rules']}")
        print(f"\n   Rule Details:")
        for rule, passes in result['rule_details'].items():
            if rule != 'overall':
                symbol = "+" if passes else "-"
                print(f"      [{symbol}] {rule}")
        print(f"{'='*60}")

    elif args.input_csv:
        # Batch from CSV
        predictor.predict_from_csv(args.input_csv, args.output_csv, args.smiles_column)

    else:
        # Interactive mode
        print(f"\n{'='*60}")
        print(f"Interactive BBB Prediction")
        print(f"Type a SMILES string and press Enter.")
        print(f"Type 'quit' to exit.")
        print(f"{'='*60}")

        while True:
            try:
                smiles = input("\nSMILES> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if smiles.lower() in ['quit', 'exit', 'q']:
                break

            if not smiles:
                continue

            try:
                result = predictor.predict_single(smiles)
                print(f"   Prediction: {result['predicted_class']}")
                print(f"   BBB+ Prob: {result['bbb_positive_probability']:.4f}")
                print(f"   Rules OK: {result['follows_bbb_rules']}")
            except Exception as e:
                print(f"   Error: {e}")

        print("\nGoodbye!")


if __name__ == "__main__":
    main()