import torch
from torch import nn
from typing import List, Optional, Tuple, Union
import warnings


class CrossAttentionLayer(nn.Module):
    """
    Enhanced cross-attention layer with gating mechanism and memory efficiency improvements.
    
    Features:
    - Gated attention mechanism for controlled information flow
    - Layer scaling for stable training
    - Memory-efficient attention computation
    - Optional residual connection
    """
    def __init__(self, 
                 hidden_size: int, 
                 num_heads: int, 
                 dropout_rate: float = 0.1,
                 use_gating: bool = True,
                 scale_attention: bool = True,
                 use_residual: bool = True):
        """
        Args:
            hidden_size (int): Size of input embeddings
            num_heads (int): Number of attention heads
            dropout_rate (float): Dropout probability
            use_gating (bool): Whether to use gating mechanism
            scale_attention (bool): Whether to scale attention outputs
            use_residual (bool): Whether to use residual connections
        """
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5 if scale_attention else 1.0
        self.use_gating = use_gating
        self.use_residual = use_residual
        
        # Attention projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Gating mechanism
        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid()
            )
        
        # Normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stability"""
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.02)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.02)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.02)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.02)
        nn.init.constant_(self.q_proj.bias, 0.)
        nn.init.constant_(self.k_proj.bias, 0.)
        nn.init.constant_(self.v_proj.bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, 
                hidden_states: torch.Tensor, 
                cross_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with memory-efficient attention computation.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            cross_states: (batch_size, seq_len, hidden_size)
            attention_mask: Optional mask for attention
            
        Returns:
            torch.Tensor: Updated hidden states
        """
        batch_size = hidden_states.size(0)
        
        # Project queries, keys, values
        q = self.q_proj(hidden_states)  # (batch_size, seq_len, hidden_size)
        k = self.k_proj(cross_states)   # (batch_size, seq_len, hidden_size)
        v = self.v_proj(cross_states)   # (batch_size, seq_len, hidden_size)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        
        # Reshape back to original dimensions
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.hidden_size)
        
        # Project back to hidden size
        attn_output = self.out_proj(attn_output)
        
        # Apply gating mechanism if enabled
        if self.use_gating:
            gate_values = self.gate(hidden_states)
            attn_output = gate_values * attn_output
        
        # Apply dropout
        attn_output = self.dropout(attn_output)
        
        # Residual connection and layer norm if enabled
        if self.use_residual:
            hidden_states = hidden_states + attn_output
            hidden_states = self.layer_norm(hidden_states)
        else:
            hidden_states = attn_output
        
        return hidden_states


class CrossAttentionInjector:
    """
    Enhanced cross-attention injector with:
    - Dynamic layer selection
    - Memory management
    - Gradient checkpointing support
    - Flexible injection strategies
    """
    def __init__(self, 
                 hidden_size: int, 
                 num_heads: int,
                 cross_attention_config: Optional[dict] = None,
                 gradient_checkpointing: bool = False):
        """
        Args:
            hidden_size: Dimension of hidden states
            num_heads: Number of attention heads
            cross_attention_config: Configuration dict for cross-attention layers
            gradient_checkpointing: Whether to enable gradient checkpointing
        """
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.gradient_checkpointing = gradient_checkpointing
        
        # Default cross-attention configuration
        self.default_config = {
            'dropout_rate': 0.1,
            'use_gating': True,
            'scale_attention': True,
            'use_residual': True
        }
        
        # Update with provided config
        if cross_attention_config:
            self.default_config.update(cross_attention_config)
    
    def find_model_layers(self, model: nn.Module) -> Tuple[List[nn.Module], List[str]]:
        """
        Identify layers suitable for cross-attention injection.
        
        Returns:
            Tuple of (layer_modules, layer_names)
        """
        candidates = []
        names = []
        
        # Common patterns in transformer architectures
        search_patterns = [
            'h.{i}',          # GPT-like
            'layer.{i}',      # BERT-like
            'layers.{i}',     # Other variants
            'transformer.h.{i}',
            'decoder.layers.{i}'
        ]
        
        # Search for layers in the model
        for name, module in model.named_modules():
            if any(pattern.split('.')[0] in name for pattern in search_patterns):
                if isinstance(module, nn.ModuleList):
                    candidates.extend(module)
                    names.extend([f"{name}.{i}" for i in range(len(module))])
        
        if not candidates:
            raise ValueError("No suitable layers found for cross-attention injection")
            
        return candidates, names
    
    def inject(self,
               base_model: nn.Module,
               scaffold_model: nn.Module,
               layers_to_inject: Optional[Union[List[int], str]] = None,
               injection_strategy: str = 'replace') -> nn.Module:
        """
        Inject cross-attention layers into the base model.
        
        Args:
            base_model: Model to modify
            scaffold_model: Model providing cross-attention context
            layers_to_inject: Either:
                - List of layer indices to inject
                - 'all' to inject all layers
                - 'alternate' to inject every other layer
                - 'last_half' to inject the latter half of layers
            injection_strategy: How to inject the cross-attention:
                - 'replace': Replace entire layer
                - 'parallel': Run in parallel with original layer
                - 'sequential': Add after original layer
        
        Returns:
            Modified base model
        """
        # Find suitable layers in the base model
        layers, layer_names = self.find_model_layers(base_model)
        
        # Determine which layers to inject
        if layers_to_inject == 'all':
            layers_to_inject = list(range(len(layers)))
        elif layers_to_inject == 'alternate':
            layers_to_inject = list(range(0, len(layers), 2))
        elif layers_to_inject == 'last_half':
            layers_to_inject = list(range(len(layers)//2, len(layers)))
        elif isinstance(layers_to_inject, list):
            # Validate layer indices
            layers_to_inject = [i for i in layers_to_inject if 0 <= i < len(layers)]
        else:
            raise ValueError(f"Invalid layers_to_inject: {layers_to_inject}")
        
        if not layers_to_inject:
            warnings.warn("No layers selected for cross-attention injection")
            return base_model
        
        print(f"Injecting cross-attention into layers: {[layer_names[i] for i in layers_to_inject]}")
        
        # Create cross-attention layers
        cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                **self.default_config
            )
            for _ in layers_to_inject
        ])
        
        # Apply injection based on strategy
        for i, layer_idx in enumerate(layers_to_inject):
            original_layer = layers[layer_idx]
            
            if injection_strategy == 'replace':
                layers[layer_idx] = self._create_wrapped_layer(
                    original_layer, 
                    cross_attention_layers[i], 
                    scaffold_model
                )
            elif injection_strategy == 'parallel':
                layers[layer_idx] = self._create_parallel_layer(
                    original_layer,
                    cross_attention_layers[i],
                    scaffold_model
                )
            elif injection_strategy == 'sequential':
                layers[layer_idx] = self._create_sequential_layer(
                    original_layer,
                    cross_attention_layers[i],
                    scaffold_model
                )
            else:
                raise ValueError(f"Unknown injection strategy: {injection_strategy}")
        
        return base_model
    
    def _create_wrapped_layer(self,
                             original_layer: nn.Module,
                             cross_attention_layer: CrossAttentionLayer,
                             scaffold_model: nn.Module) -> nn.Module:
        """Create a layer that replaces the original with cross-attention"""
        class WrappedLayer(nn.Module):
            def __init__(self, base_layer, cross_attn, scaffold):
                super().__init__()
                self.base_layer = base_layer
                self.cross_attn = cross_attn
                self.scaffold = scaffold
            
            def forward(self, hidden_states, *args, **kwargs):
                # Get scaffold context
                with torch.no_grad():
                    scaffold_output = self.scaffold(hidden_states)
                
                # Apply cross-attention
                hidden_states = self.cross_attn(hidden_states, scaffold_output)
                
                # Pass through original layer
                return self.base_layer(hidden_states, *args, **kwargs)
        
        return WrappedLayer(original_layer, cross_attention_layer, scaffold_model)
    
    def _create_parallel_layer(self,
                              original_layer: nn.Module,
                              cross_attention_layer: CrossAttentionLayer,
                              scaffold_model: nn.Module) -> nn.Module:
        """Create a layer that runs original and cross-attention in parallel"""
        class ParallelLayer(nn.Module):
            def __init__(self, base_layer, cross_attn, scaffold):
                super().__init__()
                self.base_layer = base_layer
                self.cross_attn = cross_attn
                self.scaffold = scaffold
                self.combine = nn.Linear(self.cross_attn.hidden_size * 2, self.cross_attn.hidden_size)
            
            def forward(self, hidden_states, *args, **kwargs):
                # Original layer path
                base_output = self.base_layer(hidden_states, *args, **kwargs)
                
                # Cross-attention path
                with torch.no_grad():
                    scaffold_output = self.scaffold(hidden_states)
                cross_output = self.cross_attn(hidden_states, scaffold_output)
                
                # Combine outputs
                combined = torch.cat([base_output, cross_output], dim=-1)
                return self.combine(combined)
        
        return ParallelLayer(original_layer, cross_attention_layer, scaffold_model)
    
    def _create_sequential_layer(self,
                                original_layer: nn.Module,
                                cross_attention_layer: CrossAttentionLayer,
                                scaffold_model: nn.Module) -> nn.Module:
        """Create a layer that runs cross-attention after original layer"""
        class SequentialLayer(nn.Module):
            def __init__(self, base_layer, cross_attn, scaffold):
                super().__init__()
                self.base_layer = base_layer
                self.cross_attn = cross_attn
                self.scaffold = scaffold
            
            def forward(self, hidden_states, *args, **kwargs):
                # Original layer
                hidden_states = self.base_layer(hidden_states, *args, **kwargs)
                
                # Cross-attention
                with torch.no_grad():
                    scaffold_output = self.scaffold(hidden_states)
                return self.cross_attn(hidden_states, scaffold_output)
        
        return SequentialLayer(original_layer, cross_attention_layer, scaffold_model)


def inject_cross_attention(
    base_model: nn.Module,
    scaffold_model: nn.Module,
    config: Optional[dict] = None
) -> nn.Module:
    """
    Enhanced cross-attention injection with automatic configuration.
    
    Args:
        base_model: Model to modify
        scaffold_model: Model providing cross-attention context
        config: Configuration dictionary with keys:
            - hidden_size: Dimension of hidden states
            - num_heads: Number of attention heads
            - layers_to_inject: Which layers to modify
            - injection_strategy: How to inject ('replace', 'parallel', 'sequential')
            - cross_attention_config: Configuration for CrossAttentionLayer
            - gradient_checkpointing: Whether to enable gradient checkpointing
    
    Returns:
        Modified base model
    """
    # Default configuration
    default_config = {
        'hidden_size': 768,
        'num_heads': 12,
        'layers_to_inject': 'all',
        'injection_strategy': 'replace',
        'cross_attention_config': None,
        'gradient_checkpointing': False
    }
    
    if config:
        default_config.update(config)
    
    # Initialize injector
    injector = CrossAttentionInjector(
        hidden_size=default_config['hidden_size'],
        num_heads=default_config['num_heads'],
        cross_attention_config=default_config['cross_attention_config'],
        gradient_checkpointing=default_config['gradient_checkpointing']
    )
    
    # Perform injection
    return injector.inject(
        base_model=base_model,
        scaffold_model=scaffold_model,
        layers_to_inject=default_config['layers_to_inject'],
        injection_strategy=default_config['injection_strategy']
    )
