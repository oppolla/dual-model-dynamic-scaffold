import torch
from torch import nn
from typing import List, Optional


class CrossAttentionLayer(nn.Module):
    """
    Implements a single cross-attention layer.
    """
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.1):
        """
        Args:
            hidden_size (int): The size of the hidden layer.
            num_heads (int): Number of attention heads.
            dropout_rate (float): Dropout rate for the attention mechanism.
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states: torch.Tensor, cross_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the cross-attention layer.

        Args:
            hidden_states (torch.Tensor): Input embeddings from the base model.
            cross_states (torch.Tensor): Input embeddings from the scaffold model.

        Returns:
            torch.Tensor: Updated hidden states after applying cross-attention.
        """
        # Query is the hidden states; Key and Value come from the cross states
        attn_output, _ = self.attention(query=hidden_states, key=cross_states, value=cross_states)
        # Add residual connection and apply layer normalization
        hidden_states = self.layer_norm(hidden_states + self.dropout(attn_output))
        return hidden_states


class CrossAttentionInjector:
    """
    Handles the injection of cross-attention layers into model architectures.
    """
    def __init__(self, hidden_size: int, num_heads: int, num_layers: int, dropout_rate: float = 0.1):
        """
        Args:
            hidden_size (int): The size of the hidden layer.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of cross-attention layers to inject.
            dropout_rate (float): Dropout rate for the attention mechanism.
        """
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

    def inject(self, base_model: nn.Module, scaffold_model: nn.Module, layers_to_inject: Optional[List[int]] = None):
        """
        Inject cross-attention layers into the specified model.

        Args:
            base_model (nn.Module): The base model into which cross-attention layers will be injected.
            scaffold_model (nn.Module): The scaffold model providing cross-attention embeddings.
            layers_to_inject (Optional[List[int]]): Specific layers where cross-attention will be injected.
                If None, cross-attention will be injected into all layers.

        Returns:
            nn.Module: The modified base model with cross-attention layers injected.
        """
        # Default behavior: inject into all layers if not specified
        if layers_to_inject is None:
            layers_to_inject = list(range(self.num_layers))

        # Ensure the base model has an encoder with layers
        if not hasattr(base_model, "encoder") or not isinstance(base_model.encoder, nn.ModuleList):
            raise ValueError("The base model must have an 'encoder' attribute of type nn.ModuleList.")

        # Inject cross-attention layers into the specified layers
        for layer_idx in layers_to_inject:
            if layer_idx >= len(base_model.encoder):
                raise ValueError(f"Layer index {layer_idx} is out of range for the base model's encoder.")
            
            original_layer = base_model.encoder[layer_idx]
            base_model.encoder[layer_idx] = self._wrap_with_cross_attention(original_layer, scaffold_model)

        return base_model

    def _wrap_with_cross_attention(self, original_layer: nn.Module, scaffold_model: nn.Module) -> nn.Module:
        """
        Wrap a layer with cross-attention functionality.

        Args:
            original_layer (nn.Module): The original layer to wrap.
            scaffold_model (nn.Module): The scaffold model providing cross-attention embeddings.

        Returns:
            nn.Module: A wrapped layer with cross-attention functionality.
        """
        class WrappedLayer(nn.Module):
            """
            Wraps a single layer to include cross-attention functionality.
            """
            def __init__(self, base_layer, cross_attention_layer, scaffold_model):
                super().__init__()
                self.base_layer = base_layer
                self.cross_attention_layer = cross_attention_layer
                self.scaffold_model = scaffold_model

            def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
                # Forward pass through the base layer
                hidden_states = self.base_layer(hidden_states, *args, **kwargs)

                # Get cross-attention embeddings from the scaffold model
                with torch.no_grad():
                    scaffold_output = self.scaffold_model(hidden_states)

                # Apply cross-attention
                hidden_states = self.cross_attention_layer(hidden_states, scaffold_output)

                return hidden_states

        # Create the cross-attention layer
        cross_attention_layer = CrossAttentionLayer(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate
        )

        # Return the wrapped layer
        return WrappedLayer(original_layer, cross_attention_layer, scaffold_model)


def inject_cross_attention(base_model: nn.Module, scaffold_model: nn.Module, config: dict) -> nn.Module:
    """
    Utility function to inject cross-attention layers using a configuration dictionary.

    Args:
        base_model (nn.Module): The base model into which cross-attention layers will be injected.
        scaffold_model (nn.Module): The scaffold model providing cross-attention embeddings.
        config (dict): Configuration dictionary with keys:
            - 'hidden_size': (int) Hidden size for the attention mechanism.
            - 'num_heads': (int) Number of attention heads.
            - 'num_layers': (int) Number of layers to inject cross-attention.
            - 'dropout_rate': (float) Dropout rate for the attention mechanism.
            - 'layers_to_inject': (Optional[List[int]]) Specific layers to inject; default is all layers.

    Returns:
        nn.Module: The modified base model with cross-attention layers injected.
    """
    injector = CrossAttentionInjector(
        hidden_size=config.get('hidden_size', 768),
        num_heads=config.get('num_heads', 12),
        num_layers=config.get('num_layers', 12),
        dropout_rate=config.get('dropout_rate', 0.1)
    )
    return injector.inject(
        base_model=base_model,
        scaffold_model=scaffold_model,
        layers_to_inject=config.get('layers_to_inject', None)
    )