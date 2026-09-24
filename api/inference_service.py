"""
Inference service for BBB prediction.
Handles model loading and prediction logic with caching.
"""
import torch
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

from src.models import SMILESXModel
from src.utils import SMILESTokenizer, check_bbb_rules

logger = logging.getLogger(__name__)


class InferenceService:
    """
    Singleton service for loading and running predictions.
    Model is loaded once and reused across requests.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InferenceService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.device = torch.device('cpu')
        self.model = None
        self.tokenizer = None
        self.config = None
        self.checkpoint_path = None
        self._initialized = True
    
    def load_model(self, checkpoint_path: str) -> bool:
        """
        Load model and tokenizer from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.checkpoint_path = checkpoint_path
            logger.info(f"Loading model from {checkpoint_path}...")
            
            checkpoint = torch.load(
                checkpoint_path, 
                map_location=self.device,
                weights_only=False
            )
            
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
            )
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("✅ Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            return False
    
    def is_ready(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model is not None and self.tokenizer is not None
    
    def predict(self, smiles: str) -> Dict:
        """
        Predict BBB penetration for a single molecule.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_ready():
            raise RuntimeError("Model not loaded")
        
        try:
            # Tokenize
            encoded = self.tokenizer.encode(smiles)
            if encoded is None:
                return {
                    "error": f"Invalid SMILES: {smiles}",
                    "bbb_score": None,
                    "prediction": None,
                    "confidence": None
                }
            
            tokens, mask = encoded
            tokens = torch.tensor(tokens).unsqueeze(0).to(self.device)
            mask = torch.tensor(mask).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.model(tokens, mask)
                
                # Handle case where model returns (logits, attention_weights)
                if isinstance(output, tuple):
                    logits = output[0]  # Get just the logits
                else:
                    logits = output
                
                probs = torch.softmax(logits, dim=1)

                bbb_score = probs[0, 1].item()  # Probability of BBB+
                prediction = "BBB+" if bbb_score > 0.5 else "BBB-"
                confidence = max(probs[0].tolist())
            
            # Check Lipinski's rules
            rules = check_bbb_rules(smiles)
            
            return {
                "smiles": smiles,
                "bbb_score": round(bbb_score, 4),
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "rules_check": rules if rules else None,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Prediction error for {smiles}: {str(e)}")
            return {
                "error": str(e),
                "bbb_score": None,
                "prediction": None,
                "confidence": None
            }
    
    def batch_predict(self, smiles_list: List[str]) -> Tuple[List[Dict], float]:
        """
        Predict for multiple molecules.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Tuple of (results list, processing_time_ms)
        """
        if not self.is_ready():
            raise RuntimeError("Model not loaded")
        
        import time
        start_time = time.time()
        
        results = [self.predict(smiles) for smiles in smiles_list]
        
        processing_time_ms = (time.time() - start_time) * 1000
        return results, processing_time_ms