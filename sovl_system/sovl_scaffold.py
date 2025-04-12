import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Dict
import warnings
from collections import defaultdict
import time

class CrossAttentionLayer(nn.Module):
    """
    Enhanced cross-attention layer with gating, memory efficiency, and dynamic control.

    Features:
    - Gated attention with dynamic influence weight and blend strength
    - Memory-efficient attention with optional scaled dot-product
    - Support for dream memory blending and pooling
    - Configurable residual connections and normalization
    - Gradient checkpointing for training
    - Optional sparse attention for long sequences
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        use_gating: bool = True,
        scale_attention: bool = True,
        use_residual: bool = True,
        use_pooling: bool = False,
        use_sparse_attention: bool = False,
        gradient_checkpointing: bool = False,
        quantization_mode: str = 'fp16',
        scale_factor: Optional[float] = None,
        sparse_pattern: str = 'window',
        sparse_window_size: int = 128,
    ):
        """
        Initialize the cross-attention layer.

        Args:
            hidden_size (int): Size of input embeddings.
            num_heads (int): Number of attention heads.
            dropout_rate (float): Dropout probability.
            use_gating (bool): Enable gating mechanism.
            scale_attention (bool): Scale attention scores.
            use_residual (bool): Use residual connections.
            use_pooling (bool): Pool cross-states (mean over sequence).
            use_sparse_attention (bool): Use sparse attention for long sequences.
            gradient_checkpointing (bool): Enable gradient checkpointing.
            quantization_mode (str): Quantization mode ('fp16', 'int8', 'int4').
            scale_factor (Optional[float]): Custom attention scaling factor.
            sparse_pattern (str): Sparse attention pattern ('window', 'block').
            sparse_window_size (int): Size of sparse window for 'window' pattern.

        Example:
            >>> layer = CrossAttentionLayer(hidden_size=768, num_heads=12, use_pooling=True)
            >>> hidden_states = torch.randn(2, 64, 768)
            >>> cross_states = torch.randn(2, 64, 768)
            >>> output = layer(hidden_states, cross_states)
        """
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_gating = use_gating
        self.use_residual = use_residual
        self.use_pooling = use_pooling
        self.use_sparse_attention = use_sparse_attention
        self.gradient_checkpointing = gradient_checkpointing
        self.quantization_mode = quantization_mode
        self.scale = scale_factor if scale_factor is not None else (self.head_dim ** -0.5 if scale_attention else 1.0)
        self.logger = None  # Will be set by injector if provided
        self.sparse_pattern = sparse_pattern
        self.sparse_window_size = sparse_window_size

        # Dynamic control parameters
        self.influence_weight = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.blend_strength = nn.Parameter(torch.tensor(0.5), requires_grad=False)

        # Attention projections
        if quantization_mode in ['int8', 'int4']:
            try:
                from bitsandbytes.nn import Linear8bitLt, Linear4bit
                LinearCls = Linear8bitLt if quantization_mode == 'int8' else Linear4bit
                self.q_proj = LinearCls(hidden_size, hidden_size)
                self.k_proj = LinearCls(hidden_size, hidden_size)
                self.v_proj = LinearCls(hidden_size, hidden_size)
                self.out_proj = LinearCls(hidden_size, hidden_size)
            except ImportError:
                warnings.warn("bitsandbytes not installed, falling back to fp16")
                self.q_proj = nn.Linear(hidden_size, hidden_size)
                self.k_proj = nn.Linear(hidden_size, hidden_size)
                self.v_proj = nn.Linear(hidden_size, hidden_size)
                self.out_proj = nn.Linear(hidden_size, hidden_size)
        else:
            self.q_proj = nn.Linear(hidden_size, hidden_size)
            self.k_proj = nn.Linear(hidden_size, hidden_size)
            self.v_proj = nn.Linear(hidden_size, hidden_size)
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

        # Sparse attention setup
        if use_sparse_attention:
            self.sparse_mask = self._init_sparse_mask(max_seq_len=512)
        else:
            self.sparse_mask = None

        # Key-value cache for inference
        self.use_cache = False
        self.kv_cache = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self, model_config: Optional[dict] = None):
        """Initialize weights with model-specific scaling."""
        gain = model_config.get('initializer_range', 0.02) if model_config else 0.02
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight, gain=gain)
                nn.init.constant_(module.bias, 0.0)
        if self.use_gating:
            nn.init.xavier_uniform_(self.gate[0].weight, gain=gain)
            nn.init.constant_(self.gate[0].bias, 0.0)

    def _init_sparse_mask(self, max_seq_len: int) -> torch.Tensor:
        """Initialize sparse attention mask based on pattern."""
        if self.sparse_pattern == 'window':
            mask = torch.zeros(max_seq_len, max_seq_len)
            for i in range(max_seq_len):
                start = max(0, i - self.sparse_window_size // 2)
                end = min(max_seq_len, i + self.sparse_window_size // 2 + 1)
                mask[i, start:end] = 1.0
            return mask.bool()
        elif self.sparse_pattern == 'block':
            block_size = self.sparse_window_size
            mask = torch.zeros(max_seq_len, max_seq_len)
            for i in range(0, max_seq_len, block_size):
                mask[i:i+block_size, i:i+block_size] = 1.0
            return mask.bool()
        else:
            raise ValueError(f"Unknown sparse pattern: {self.sparse_pattern}")

    def _prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Prepare attention mask in additive format."""
        if attention_mask is None:
            return None
        try:
            # Ensure mask is additive (negative infinity for masked positions)
            if attention_mask.dtype == torch.bool:
                attention_mask = attention_mask.float().masked_fill(~attention_mask, float('-inf'))
            elif attention_mask.dtype != torch.float:
                attention_mask = attention_mask.float()
            # Broadcast to [batch_size, num_heads, seq_len, seq_len]
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(1)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            if attention_mask.shape != (batch_size, self.num_heads, seq_len, seq_len):
                attention_mask = attention_mask.expand(batch_size, self.num_heads, seq_len, seq_len)
            return attention_mask.to(device)
        except Exception as e:
            if hasattr(self, 'logger') and callable(self.logger):
                self.logger({
                    "warning": f"Attention mask preparation failed: {str(e)}",
                    "timestamp": time.time(),
                    "mask_shape": list(attention_mask.shape) if attention_mask is not None else None
                })
            return None

    def reset_cache(self):
        """Reset key-value cache for inference."""
        self.kv_cache = None
        if hasattr(self, 'logger') and callable(self.logger):
            self.logger({
                "event": "cache_reset",
                "timestamp": time.time()
            })

    def set_influence_weight(self, weight: float):
        """Set the influence weight for attention output."""
        self.influence_weight.data = torch.tensor(max(0.0, weight), dtype=torch.float)

    def set_blend_strength(self, strength: float):
        """Set the blend strength for residual connection."""
        self.blend_strength.data = torch.tensor(max(0.0, min(1.0, strength)), dtype=torch.float)

    def set_lifecycle_weight(self, weight: float, curve: str = 'sigmoid_linear'):
        """Adjust influence weight based on lifecycle curve."""
        if curve == 'sigmoid_linear':
            adjusted = weight / (1 + torch.exp(-weight))
        elif curve == 'exponential':
            adjusted = weight * torch.exp(-weight)
        else:
            adjusted = weight
        self.set_influence_weight(adjusted)

    def _forward(
        self,
        hidden_states: torch.Tensor,
        cross_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory_tensors: Optional[torch.Tensor] = None,
        memory_weight: float = 0.0,
        dynamic_factor: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Core forward pass logic."""
        batch_size = hidden_states.size(0)

        # Pool cross-states if enabled
        if self.use_pooling:
            cross_states = cross_states.mean(dim=1, keepdim=True)

        # Blend with memory tensors if provided
        if memory_tensors is not None and memory_weight > 0:
            try:
                # Ensure memory_tensors shape matches cross_states
                if memory_tensors.shape[-1] != cross_states.shape[-1]:
                    raise ValueError(f"Memory tensors dimension mismatch: {memory_tensors.shape[-1]} vs {cross_states.shape[-1]}")
                # Blend with clamping to avoid extreme weights
                memory_weight = max(0.0, min(1.0, memory_weight))
                cross_states = (1 - memory_weight) * cross_states + memory_weight * memory_tensors
            except Exception as e:
                # Log warning via module's logger if available
                if hasattr(self, 'logger') and callable(self.logger):
                    self.logger({
                        "warning": f"Dream memory blending failed: {str(e)}",
                        "timestamp": time.time()
                    })
                # Proceed without blending
                pass

        # Project queries, keys, values
        q = self.q_proj(hidden_states)
        k = self.k_proj(cross_states)
        v = self.v_proj(cross_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Cache keys and values if enabled
        self.use_cache = use_cache
        if self.use_cache:
            if self.kv_cache is None:
                self.kv_cache = (k, v)
            else:
                prev_k, prev_v = self.kv_cache
                k = torch.cat([prev_k, k], dim=2)
                v = torch.cat([prev_v, v], dim=2)
                self.kv_cache = (k, v)
        else:
            self.kv_cache = None

        # Prepare attention mask
        seq_len = q.size(-2)
        attention_mask = self._prepare_attention_mask(attention_mask, batch_size, seq_len, q.device)

        # Attention computation
        if self.use_sparse_attention:
            seq_len = k.size(2)
            # Resize sparse mask if sequence length differs
            if self.sparse_mask.shape[-1] < seq_len:
                self.sparse_mask = self._init_sparse_mask(max_seq_len=seq_len).to(k.device)
            sparse_mask = self.sparse_mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(1)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_scores = attn_scores + (~sparse_mask).float() * float('-inf')
        elif hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask
            attn_probs = torch.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_probs, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.hidden_size)
        attn_output = self.out_proj(attn_output)

        # Apply dynamic factor if provided
        if dynamic_factor is not None:
            attn_output = attn_output * dynamic_factor

        # Apply gating
        if self.use_gating:
            gate_values = self.gate(hidden_states)
            attn_output = gate_values * attn_output * self.influence_weight

        # Apply dropout
        attn_output = self.dropout(attn_output)

        # Residual connection
        if self.use_residual:
            hidden_states = (1 - self.blend_strength) * hidden_states + self.blend_strength * attn_output
            hidden_states = self.layer_norm(hidden_states)
        else:
            hidden_states = attn_output

        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory_tensors: Optional[torch.Tensor] = None,
        memory_weight: float = 0.0,
        dynamic_factor: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with optional gradient checkpointing.

        Args:
            hidden_states (torch.Tensor): Input hidden states (batch_size, seq_len, hidden_size).
            cross_states (torch.Tensor): Cross-attention context (batch_size, seq_len, hidden_size).
            attention_mask (Optional[torch.Tensor]): Attention mask.
            memory_tensors (Optional[torch.Tensor]): Dream memory tensors for blending.
            memory_weight (float): Weight for memory blending.
            dynamic_factor (Optional[torch.Tensor]): Dynamic modulation factor.
            use_cache (bool): Enable key-value caching.

        Returns:
            torch.Tensor: Updated hidden states.
        """
        assert hidden_states.dim() == 3, "Expected batched input (batch_size, seq_len, hidden_size)"

        with torch.autocast(device_type=hidden_states.device.type, enabled=True):
            if self.training and self.gradient_checkpointing:
                return checkpoint(
                    self._forward,
                    hidden_states,
                    cross_states,
                    attention_mask,
                    memory_tensors,
                    memory_weight,
                    dynamic_factor,
                    use_cache,
                    use_reentrant=False
                )
            return self._forward(
                hidden_states,
                cross_states,
                attention_mask,
                memory_tensors,
                memory_weight,
                dynamic_factor,
                use_cache
            )


class CrossAttentionInjector:
    """
    Injector for adding cross-attention layers to a transformer model.

    Features:
    - Flexible layer selection (all, alternate, early, balanced, custom)
    - Multiple injection strategies (replace, parallel, sequential)
    - Support for hidden size projection
    - Integration with token mapping and logger
    - State saving/loading for cross-attention parameters
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        cross_attention_config: Optional[dict] = None,
        gradient_checkpointing: bool = False,
        custom_layers: Optional[List[int]] = None,
        logger: Optional[callable] = None,
        quantization_mode: str = 'fp16',
    ):
        """
        Initialize the injector.

        Args:
            hidden_size (int): Dimension of hidden states.
            num_heads (int): Number of attention heads.
            cross_attention_config (Optional[dict]): Config for CrossAttentionLayer.
            gradient_checkpointing (bool): Enable gradient checkpointing.
            custom_layers (Optional[List[int]]): Custom layer indices for injection.
            logger (Optional[callable]): Logger function for error reporting.
            quantization_mode (str): Quantization mode ('fp16', 'int8', 'int4').

        Example:
            >>> injector = CrossAttentionInjector(hidden_size=768, num_heads=12)
            >>> base_model = AutoModelForCausalLM.from_pretrained("gpt2")
            >>> scaffold_model = AutoModelForCausalLM.from_pretrained("gpt2")
            >>> modified_model = injector.inject(base_model, scaffold_model, layers_to_inject='early')
        """
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.gradient_checkpointing = gradient_checkpointing
        self.custom_layers = custom_layers or []
        self.logger = logger
        self.quantization_mode = quantization_mode
        self.scaffold_proj = None

        # Default cross-attention configuration
        self.default_config = {
            'dropout_rate': 0.1,
            'use_gating': True,
            'scale_attention': True,
            'use_residual': True,
            'use_pooling': False,
            'use_sparse_attention': False,
            'gradient_checkpointing': gradient_checkpointing,
            'quantization_mode': quantization_mode,
        }

        if cross_attention_config:
            self.default_config.update(cross_attention_config)

    def find_model_layers(self, model: nn.Module) -> Tuple[List[nn.Module], List[str]]:
        """
        Identify layers suitable for cross-attention injection.

        Args:
            model (nn.Module): Model to inspect.

        Returns:
            Tuple[List[nn.Module], List[str]]: List of layers and their names.
        """
        candidates = []
        names = []

        search_patterns = [
            'h.{i}',
            'layer.{i}',
            'layers.{i}',
            'transformer.h.{i}',
            'decoder.layers.{i}',
        ]

        for name, module in model.named_modules():
            if any(pattern.split('.')[0] in name for pattern in search_patterns):
                if isinstance(module, nn.ModuleList):
                    candidates.extend(module)
                    names.extend([f"{name}.{i}" for i in range(len(module))])

        if not candidates:
            for name, module in model.named_modules():
                if isinstance(module, nn.ModuleList):
                    candidates.extend(module)
                    names.extend([f"{name}.{i}" for i in range(len(module))])

        if not candidates:
            error_msg = "No suitable layers found for cross-attention injection"
            if self.logger:
                self.logger({"error": error_msg, "timestamp": time.time()})
            raise ValueError(error_msg)

        return candidates, names

    def inject(
        self,
        base_model: nn.Module,
        scaffold_model: Union[nn.Module, List[nn.Module]],
        layers_to_inject: Optional[Union[List[int], str]] = None,
        injection_strategy: str = 'sequential',
        token_map: Optional[Dict] = None,
    ) -> nn.Module:
        """
        Inject cross-attention layers into the base model.

        Args:
            base_model (nn.Module): Model to modify.
            scaffold_model (Union[nn.Module, List[nn.Module]]): Model(s) providing context.
            layers_to_inject (Union[List[int], str]): Layers to inject ('all', 'alternate', 'last_half', 'early', 'balanced', 'custom').
            injection_strategy (str): Injection method ('replace', 'parallel', 'sequential').
            token_map (Optional[Dict]): Token mapping for scaffold inputs.

        Returns:
            nn.Module: Modified base model.
        """
        if isinstance(scaffold_model, nn.Module):
            scaffold_models = [scaffold_model]
        else:
            scaffold_models = scaffold_model

        # Validate hidden sizes
        base_hidden_size = getattr(base_model.config, 'hidden_size', self.hidden_size)
        scaffold_hidden_size = getattr(scaffold_models[0].config, 'hidden_size', self.hidden_size)
        if base_hidden_size != scaffold_hidden_size:
            print(f"Hidden size mismatch ({base_hidden_size} vs {scaffold_hidden_size}). Adding projection.")
            self.scaffold_proj = nn.Linear(scaffold_hidden_size, base_hidden_size).to(base_model.device)

        # Find layers
        layers, layer_names = self.find_model_layers(base_model)

        # Determine layers to inject
        if layers_to_inject == 'all':
            layers_to_inject = list(range(len(layers)))
        elif layers_to_inject == 'alternate':
            layers_to_inject = list(range(0, len(layers), 2))
        elif layers_to_inject == 'last_half':
            layers_to_inject = list(range(len(layers)//2, len(layers)))
        elif layers_to_inject == 'early':
            layers_to_inject = list(range(0, len(layers)//3))
        elif layers_to_inject == 'balanced':
            layers_to_inject = list(range(len(layers)//3, 2*len(layers)//3))
        elif layers_to_inject == 'custom' and self.custom_layers:
            layers_to_inject = [i for i in self.custom_layers if 0 <= i < len(layers)]
        elif isinstance(layers_to_inject, list):
            layers_to_inject = [i for i in layers_to_inject if 0 <= i < len(layers)]
        else:
            warnings.warn("No valid layers selected for injection")
            return base_model

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
        # Set logger for each layer
        for layer in cross_attention_layers:
            layer.logger = self.logger

        # Apply injection
        for i, layer_idx in enumerate(layers_to_inject):
            original_layer = layers[layer_idx]
            if injection_strategy == 'replace':
                layers[layer_idx] = self._create_wrapped_layer(
                    original_layer, cross_attention_layers[i], scaffold_models, token_map
                )
            elif injection_strategy == 'parallel':
                layers[layer_idx] = self._create_parallel_layer(
                    original_layer, cross_attention_layers[i], scaffold_models, token_map
                )
            elif injection_strategy == 'sequential':
                layers[layer_idx] = self._create_sequential_layer(
                    original_layer, cross_attention_layers[i], scaffold_models, token_map
                )
            else:
                error_msg = f"Unknown injection strategy: {injection_strategy}"
                if self.logger:
                    self.logger({"error": error_msg, "timestamp": time.time()})
                raise ValueError(error_msg)

        return base_model
    
    def _create_wrapped_layer(
        self,
        original_layer: nn.Module,
        cross_attention_layer: CrossAttentionLayer,
        scaffold_models: List[nn.Module],
        token_map: Optional[Dict],
    ) -> nn.Module:
        """Create a layer that replaces the original layer with cross-attention."""
        class WrappedLayer(nn.Module):
            def __init__(self, cross_attn, scaffolds, token_map, parent):
                super().__init__()
                self.cross_attn = cross_attn
                self.scaffolds = scaffolds
                self.token_map = token_map or defaultdict(lambda: [0])
                self._parent = parent

            def forward(self, hidden_states, *args, scaffold_context=None, **kwargs):
                if scaffold_context is not None:
                    context = scaffold_context.to(hidden_states.device)
                    if self._parent.scaffold_proj is not None:
                        context = self._parent.scaffold_proj(context)
                    output = self.cross_attn(hidden_states, context, **kwargs)
                    return (output,) if isinstance(hidden_states, tuple) else output
                return hidden_states

        return WrappedLayer(cross_attention_layer, scaffold_models, token_map, self)

    def _create_sequential_layer(
        self,
        original_layer: nn.Module,
        cross_attention_layer: CrossAttentionLayer,
        scaffold_models: List[nn.Module],
        token_map: Optional[Dict],
    ) -> nn.Module:
        """Create a layer that runs cross-attention after original layer."""
        class SequentialLayer(nn.Module):
            def __init__(self, base_layer, cross_attn, scaffolds, token_map, parent):
                super().__init__()
                self.base_layer = base_layer
                self.cross_attn = cross_attn
                self.scaffolds = scaffolds
                self.token_map = token_map or defaultdict(lambda: [0])
                self._parent = parent

            def forward(self, hidden_states, *args, scaffold_context=None, **kwargs):
                outputs = self.base_layer(hidden_states, *args, **kwargs)
                hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
                if scaffold_context is not None:
                    context = scaffold_context.to(hidden_states.device)
                    if self._parent.scaffold_proj is not None:
                        context = self._parent.scaffold_proj(context)
                    hidden_states = self.cross_attn(hidden_states, context, **kwargs)
                return (hidden_states,) + outputs[1:] if isinstance(outputs, tuple) else hidden_states

        return SequentialLayer(original_layer, cross_attention_layer, scaffold_models, token_map, self)

    def _create_parallel_layer(
        self,
        original_layer: nn.Module,
        cross_attention_layer: CrossAttentionLayer,
        scaffold_models: List[nn.Module],
        token_map: Optional[Dict],
    ) -> nn.Module:
        """Create a layer that runs original and cross-attention in parallel."""
        class ParallelLayer(nn.Module):
            def __init__(self, base_layer, cross_attn, scaffolds, token_map, parent):
                super().__init__()
                self.base_layer = base_layer
                self.cross_attn = cross_attn
                self.scaffolds = scaffolds
                self.token_map = token_map or defaultdict(lambda: [0])
                self.combine = nn.Linear(cross_attn.hidden_size * 2, cross_attn.hidden_size)
                self._parent = parent

            def forward(self, hidden_states, *args, scaffold_context=None, **kwargs):
                base_output = self.base_layer(hidden_states, *args, **kwargs)
                base_output = base_output[0] if isinstance(base_output, tuple) else base_output
                if scaffold_context is not None:
                    context = scaffold_context.to(hidden_states.device)
                    if self._parent.scaffold_proj is not None:
                        context = self._parent.scaffold_proj(context)
                    cross_output = self.cross_attn(hidden_states, context, **kwargs)
                    combined = torch.cat([base_output, cross_output], dim=-1)
                    output = self.combine(combined)
                    return (output,) + base_output[1:] if isinstance(base_output, tuple) else output
                return base_output

        return ParallelLayer(original_layer, cross_attention_layer, scaffold_models, token_map, self)

    def verify_injection(self, model: nn.Module) -> bool:
        """Verify that cross-attention layers were injected."""
        layers, _ = self.find_model_layers(model)
        return any(hasattr(layer, 'cross_attn') for layer in layers)

    def save_state(self, path: str, state_dict: dict):
        """Save cross-attention parameters."""
        try:
            torch.save(
                {k: v for k, v in state_dict.items() if 'cross_attn' in k or 'scaffold_proj' in k},
                path
            )
        except Exception as e:
            if self.logger:
                self.logger({"error": f"Failed to save cross-attention state: {str(e)}", "timestamp": time.time()})
            raise

    def load_state(self, path: str, model: nn.Module):
        """Load cross-attention parameters."""
        try:
            state_dict = model.state_dict()
            state_dict.update(torch.load(path, map_location=model.device))
            model.load_state_dict(state_dict)
        except Exception as e:
            if self.logger:
                self.logger({"error": f"Failed to load cross-attention state: {str(e)}", "timestamp": time.time()})
            raise


def inject_cross_attention(
    base_model: nn.Module,
    scaffold_model: Union[nn.Module, List[nn.Module]],
    config: Optional[dict] = None,
    logger: Optional[callable] = None,
) -> nn.Module:
    """
    Inject cross-attention layers with automatic configuration.

    Args:
        base_model (nn.Module): Model to modify.
        scaffold_model (Union[nn.Module, List[nn.Module]]): Model(s) providing context.
        config (Optional[dict]): Configuration with keys:
            - hidden_size: Dimension of hidden states
            - num_heads: Number of attention heads
            - layers_to_inject: Layers to modify ('all', 'alternate', 'last_half', 'early', 'balanced', 'custom')
            - injection_strategy: Injection method ('replace', 'parallel', 'sequential')
            - cross_attention_config: Config for CrossAttentionLayer
            - gradient_checkpointing: Enable gradient checkpointing
            - custom_layers: Custom layer indices
            - quantization_mode: Quantization mode ('fp16', 'int8', 'int4')
            - token_map: Token mapping for scaffold inputs
        logger (Optional[callable]): Logger function.

    Returns:
        nn.Module: Modified base model.

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> base_model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> scaffold_model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> config = {
        ...     'hidden_size': 768,
        ...     'num_heads': 12,
        ...     'layers_to_inject': 'balanced',
        ...     'cross_attention_config': {'use_pooling': True}
        ... }
        >>> modified_model = inject_cross_attention(base_model, scaffold_model, config)
    """
    # Default configuration
    default_config = {
        'hidden_size': 768,
        'num_heads': 12,
        'layers_to_inject': 'all',
        'injection_strategy': 'sequential',
        'cross_attention_config': None,
        'gradient_checkpointing': False,
        'custom_layers': None,
        'quantization_mode': 'fp16',
        'token_map': None,
    }

    if config:
        default_config.update(config)

    try:
        # Initialize injector
        injector = CrossAttentionInjector(
            hidden_size=default_config['hidden_size'],
            num_heads=default_config['num_heads'],
            cross_attention_config=default_config['cross_attention_config'],
            gradient_checkpointing=default_config['gradient_checkpointing'],
            custom_layers=default_config['custom_layers'],
            logger=logger,
            quantization_mode=default_config['quantization_mode'],
        )

        # Perform injection
        return injector.inject(
            base_model=base_model,
            scaffold_model=scaffold_model,
            layers_to_inject=default_config['layers_to_inject'],
            injection_strategy=default_config['injection_strategy'],
            token_map=default_config['token_map'],
        )
    except Exception as e:
        error_msg = f"Cross-attention injection failed: {str(e)}"
        if logger:
            logger({"error": error_msg, "timestamp": time.time()})
        raise
