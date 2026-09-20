"""
SMILES Tokenizer for molecular representation.
Converts SMILES strings (text) into sequences of numbers.
"""
import re
from typing import List, Dict, Tuple
import numpy as np


class SMILESTokenizer:
    """Tokenizer for SMILES strings."""

    def __init__(self, max_length: int = 120):
        self.max_length = max_length

        # Common atoms
        self.atom_tokens = [
            'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I',
            'B', 'Si', 'Se', 'c', 'n', 'o', 's', 'p'
        ]

        # Structural symbols
        self.structural_tokens = [
            '(', ')', '[', ']', '=', '#', '-', '+',
            '/', '\\', '@', '@@'
        ]

        # Digits
        self.digit_tokens = [str(i) for i in range(10)]

        # Special tokens
        self.special_tokens = {
            'PAD': 0,
            'START': 1,
            'END': 2,
            'UNK': 3
        }

        self.vocab = self._build_vocab()
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.pattern = self._build_pattern()

    def _build_vocab(self) -> Dict[str, int]:
        vocab = self.special_tokens.copy()
        idx = len(vocab)
        all_tokens = self.atom_tokens + self.structural_tokens + self.digit_tokens
        for token in all_tokens:
            if token not in vocab:
                vocab[token] = idx
                idx += 1
        return vocab

    def _build_pattern(self) -> re.Pattern:
        all_tokens = self.atom_tokens + self.structural_tokens + self.digit_tokens
        all_tokens = sorted(set(all_tokens), key=len, reverse=True)
        escaped = [re.escape(token) for token in all_tokens]
        pattern = '|'.join(escaped)
        return re.compile(pattern)

    def tokenize(self, smiles: str) -> List[str]:
        tokens = self.pattern.findall(smiles)
        return tokens

    def encode(self, smiles: str, add_special_tokens: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        tokens = self.tokenize(smiles)
        if add_special_tokens:
            tokens = ['START'] + tokens + ['END']
        
        indices = [self.vocab.get(token, self.vocab['UNK']) for token in tokens]
        
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
            indices[-1] = self.vocab['END']
        
        mask = [1] * len(indices)
        padding_length = self.max_length - len(indices)
        indices.extend([self.vocab['PAD']] * padding_length)
        mask.extend([0] * padding_length)
        
        return np.array(indices, dtype=np.int64), np.array(mask, dtype=np.int64)

    def decode(self, indices: List[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for idx in indices:
            token = self.idx_to_token.get(idx, 'UNK')
            if skip_special_tokens and token in self.special_tokens:
                continue
            if token == 'PAD':
                break
            tokens.append(token)
        return ''.join(tokens)

    def batch_encode(self, smiles_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        all_tokens = []
        all_masks = []
        for smiles in smiles_list:
            tokens, mask = self.encode(smiles)
            all_tokens.append(tokens)
            all_masks.append(mask)
        return np.stack(all_tokens), np.stack(all_masks)


def smiles_enumeration(smiles: str, num_variants: int = 3) -> List[str]:
    """Generate random SMILES variants."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [smiles]
        variants = [smiles]
        for _ in range(num_variants - 1):
            random_smiles = Chem.MolToSmiles(mol, doRandom=True)
            variants.append(random_smiles)
        return variants
    except ImportError:
        return [smiles]


if __name__ == "__main__":
    tokenizer = SMILESTokenizer(max_length=120)
    test_smiles = [
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "CC(=O)Oc1ccccc1C(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    ]
    print(f"Vocabulary size: {tokenizer.vocab_size}\n")
    for smiles in test_smiles:
        print(f"SMILES: {smiles}")
        tokens = tokenizer.tokenize(smiles)
        print(f"Tokens: {tokens}")
        encoded, mask = tokenizer.encode(smiles)
        print(f"Encoded shape: {encoded.shape}")
        decoded = tokenizer.decode(encoded)
        print(f"Decoded: {decoded}")
        print("-" * 60)