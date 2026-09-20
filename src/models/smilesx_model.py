"""
Attention-based neural network for BBB penetration prediction.

Architecture:
1. Token Embedding (128D)
2. Positional Encoding
3. Multi-Head Attention Layer × 3
4. Feed-Forward Network (after each attention layer)
5. Global Pooling
6. Classification Head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class PositionalEncoding(nn.Module):
    """
    Add positional information to embeddings.
    
    Why? The model needs to know WHERE each token is in the sequence.
    Same atoms in different order = different molecule.
    
    Example:
        Embedding: [0.2, -0.5, 0.8, ...]  (what the token is)
        Position:  [0.1, 0.0, -0.1, ...]  (where it is)
        Combined:  [0.3, -0.5, 0.7, ...]  (what + where)
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Embedding dimension (128)
            max_len: Maximum sequence length (5000)
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create position encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Sinusoidal positional encoding formula
        # pos_encoding(t, 2i) = sin(t / 10000^(2i/d))
        # pos_encoding(t, 2i+1) = cos(t / 10000^(2i/d))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices: sine
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices: cosine
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)  # Don't train this, just use it

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Embedding tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Embeddings + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]  # Add position to each embedding
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    
    What it does:
    1. For each token, calculates "how much attention to pay to other tokens"
    2. Does this 8 times in parallel (8 heads, each learning different patterns)
    3. Combines results
    
    Mathematical formula:
    Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
    
    Where:
    - Q (Query): "What am I looking for?"
    - K (Key): "What does each token offer?"
    - V (Value): "What information to extract?"
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: Total embedding dimension (128)
            num_heads: Number of attention heads (8)
            dropout: Dropout probability
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head (128/8 = 16)

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)  # Output projection

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate scaled dot-product attention.
        
        Step 1: Calculate similarity scores
        Step 2: Scale by √d_k (prevents too-large gradients)
        Step 3: Apply softmax (convert to probabilities)
        Step 4: Multiply by values
        
        Returns:
            output: Attended values
            attention_weights: For visualization
        """
        # Step 1 & 2: Calculate scores and scale
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Step 3: Apply mask (ignore padding tokens)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Convert to probabilities (sum to 1)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Step 4: Apply attention to values
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query, key, value: Tensors of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            output: Attended representation (batch_size, seq_len, d_model)
            attention_weights: Attention probabilities (for visualization)
        """
        batch_size = query.size(0)

        # Linear projections in batch from d_model => h x d_k
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Reshape for multi-head attention
        # (batch_size, seq_len, d_model) → (batch_size, seq_len, num_heads, d_k)
        # → (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention (all heads in parallel)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # Expand for all heads

        attn_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads
        # (batch_size, num_heads, seq_len, d_k) → (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.W_o(attn_output)

        return output, attention_weights


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    Two linear layers with ReLU activation in between.
    Applied to each position separately (hence "position-wise").
    
    Formula:
    FFN(x) = max(0, x·W1 + b1)·W2 + b2
    
    Why?
    - Adds non-linearity (ReLU)
    - Allows model to learn complex relationships
    - Expands then contracts dimension
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: Input dimension (128)
            d_ff: Hidden dimension (256)
            dropout: Dropout probability
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor
            
        Returns:
            Transformed tensor (same shape as input)
        """
        # Expand: d_model → d_ff
        # Apply ReLU (activation function)
        # Contract: d_ff → d_model
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    Single encoder layer = Attention + Feed-Forward + Residual connections + Layer normalization.
    
    Architecture:
        Input
          ↓
        Multi-Head Attention
          ↓
        Residual Connection + Layer Norm
          ↓
        Feed-Forward Network
          ↓
        Residual Connection + Layer Norm
          ↓
        Output
    
    Residual connections: Skip connections that help training
    Layer norm: Normalize activations (improves stability)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        """
        Args:
            d_model: Embedding dimension (128)
            num_heads: Number of attention heads (8)
            d_ff: Feed-forward hidden dimension (256)
            dropout: Dropout probability
            use_batch_norm: Use BatchNorm (True) or LayerNorm (False)
        """
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # Normalization
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Attention mask
            
        Returns:
            output: Processed tensor (same shape)
            attention_weights: For visualization
        """
        # Attention sub-layer with residual connection
        attn_output, attention_weights = self.self_attention(x, x, x, mask)
        x = x + self.dropout(attn_output)  # Residual connection

        # Normalize
        if self.use_batch_norm:
            x = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = self.norm1(x)

        # Feed-forward sub-layer with residual connection
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)  # Residual connection

        # Normalize
        if self.use_batch_norm:
            x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = self.norm2(x)

        return x, attention_weights


class SMILESXModel(nn.Module):
    """
    Complete attention-based model for BBB penetration prediction.
    
    Architecture:
    1. Embedding Layer (128D vectors)
    2. Positional Encoding (position awareness)
    3. 3× Encoder Layers (attention + feed-forward)
    4. Global Pooling (compress sequence)
    5. Classification Head (predict BBB+/BBB-)
    
    Total parameters: ~1.5M (very small, fast training)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.3,
        num_classes: int = 2,
        max_seq_length: int = 120,
        use_batch_norm: bool = True
    ):
        """
        Args:
            vocab_size: Size of SMILES vocabulary
            embedding_dim: Dimension of token embeddings (128)
            hidden_dim: Feed-forward hidden dimension (256)
            num_layers: Number of encoder layers (3)
            num_heads: Number of attention heads (8)
            dropout: Dropout probability (0.3)
            num_classes: Number of output classes (2: BBB+/BBB-)
            max_seq_length: Maximum SMILES length (120)
            use_batch_norm: Use BatchNorm or LayerNorm
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # 1. Token Embedding Layer
        # Convert token IDs to dense vectors
        # Padding token (ID 0) gets zero embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 2. Positional Encoding
        self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_length, dropout)

        # 3. Stack of Encoder Layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                embedding_dim,
                num_heads,
                hidden_dim,
                dropout,
                use_batch_norm
            )
            for _ in range(num_layers)
        ])

        # 4. Classification Head
        # Takes pooled representation and predicts class
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better training."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
                       Example: [1, 4, 4, 6, 2, 0, 0, ...]
            attention_mask: Mask of shape (batch_size, seq_len)
                           1 for real tokens, 0 for padding
                           Example: [1, 1, 1, 1, 1, 0, 0, ...]
            
        Returns:
            logits: Class predictions (batch_size, num_classes)
                   Example: [[0.1, 0.9], [0.8, 0.2], ...]
            attention_weights: From last encoder layer (for visualization)
        """
        # 1. Embed tokens
        x = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)

        # 2. Add positional encoding
        x = self.pos_encoding(x)  # (batch_size, seq_len, embedding_dim)

        # 3. Pass through encoder layers
        attention_weights = None
        for layer in self.encoder_layers:
            x, attention_weights = layer(x, attention_mask)

        # 4. Global average pooling
        # Collapse sequence dimension: (batch_size, seq_len, embedding_dim) → (batch_size, embedding_dim)
        if attention_mask is not None:
            # Weight by mask (ignore padding)
            mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)  # Weighted average
        else:
            x = x.mean(dim=1)  # Simple average

        # 5. Classification head
        logits = self.classifier(x)  # (batch_size, num_classes)

        return logits, attention_weights

    def get_attention_maps(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> list:
        """
        Extract attention maps from all layers for visualization.
        
        Useful for understanding which atoms the model "pays attention to".
        
        Returns:
            List of attention weight tensors from each layer
        """
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)

        all_attention_weights = []
        for layer in self.encoder_layers:
            x, attention_weights = layer(x, attention_mask)
            all_attention_weights.append(attention_weights)

        return all_attention_weights


# ---- Test code ----
if __name__ == "__main__":
    print("=" * 70)
    print("SMILES-X Model Test")
    print("=" * 70)

    # Create model
    model = SMILESXModel(
        vocab_size=67,          # From our tokenizer
        embedding_dim=128,
        hidden_dim=256,
        num_layers=3,
        num_heads=8,
        dropout=0.3,
        num_classes=2
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Test forward pass
    batch_size = 16
    seq_len = 120
    input_ids = torch.randint(0, 67, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Set first 5 positions as padding for 2 samples
    attention_mask[0, 50:] = 0
    attention_mask[1, 80:] = 0

    print(f"\n🧪 Forward Pass Test:")
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Attention mask shape: {attention_mask.shape}")

    logits, attention_weights = model(input_ids, attention_mask)

    print(f"\n✓ Output Shapes:")
    print(f"   Logits: {logits.shape}")
    print(f"   Attention weights: {attention_weights.shape}")

    # Show example predictions
    probs = torch.softmax(logits, dim=1)
    print(f"\n📝 Example Predictions:")
    for i in range(3):
        pred_class = torch.argmax(probs[i]).item()
        bbb_prob = probs[i, 1].item()
        print(f"   Sample {i}: BBB+ prob = {bbb_prob:.3f} ({'BBB+' if pred_class == 1 else 'BBB-'})")

    print("\n" + "=" * 70)
    print("✓ Model test complete!")
    print("=" * 70)