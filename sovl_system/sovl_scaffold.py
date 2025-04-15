import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Dict, Any
from collections import defaultdict
import time
import traceback
from threading import Lock
import contextlib
import logging
import math

# Assuming these are external dependencies that exist
from sovl_logger import Logger
from sovl_config import ConfigManager
from sovl_utils import NumericalGuard, safe_divide, validate_layer_indices

class ScaffoldTokenMapper:
    """
    Handles the creation and management of token mappings between the base tokenizer and the scaffold tokenizer.
    """

    def __init__(self, base_tokenizer, scaffold_tokenizer):
        """
        Initialize the ScaffoldTokenMapper with the given tokenizers.

        Args:
            base_tokenizer: The tokenizer for the base model.
            scaffold_tokenizer: The tokenizer for the scaffold model.
        """
        self.base_tokenizer = base_tokenizer
        self.scaffold_tokenizer = scaffold_tokenizer

        # Token maps
        self.token_map = defaultdict(lambda: [self.scaffold_tokenizer.unk_token_id])
        self.special_token_map = {}

        # Logging setup
        self.logger = logging.getLogger("ScaffoldTokenMapper")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(handler)

        # Build the token map and special token map
        self._build_token_map()
        self._initialize_special_token_map()

        self.logger.info("ScaffoldTokenMapper initialized successfully.")

    def _build_token_map(self):
        """
        Create the token map between base tokenizer tokens and scaffold tokenizer tokens.
        """
        self.logger.info("Building token map...")
        for base_token, base_id in self.base_tokenizer.get_vocab().items():
            normalized = self._normalize_token(base_token)
            scaffold_ids = self.scaffold_tokenizer.encode(
                normalized, add_special_tokens=False, max_length=3, truncation=True
            ) or [self.scaffold_tokenizer.unk_token_id]
            self.token_map[base_id] = {"ids": scaffold_ids, "weight": 1.0}

        self.logger.info(f"Token map built with {len(self.token_map)} entries.")

    def _initialize_special_token_map(self):
        """
        Initialize the mapping for special tokens (e.g., PAD, EOS, UNK).
        """
        self.logger.info("Initializing special token map...")
        self.special_token_map = {
            self.base_tokenizer.pad_token_id: self._get_special_token_id("pad_token_id"),
            self.base_tokenizer.eos_token_id: self._get_special_token_id("eos_token_id", fallback="sep_token_id"),
            self.base_tokenizer.unk_token_id: self._get_special_token_id("unk_token_id"),
        }
        self.logger.info("Special token map initialized.")

    def _get_special_token_id(self, token_attr: str, fallback: str = None) -> int:
        """
        Retrieve the ID of a special token from the scaffold tokenizer.

        Args:
            token_attr (str): The attribute name of the special token (e.g., "pad_token_id").
            fallback (str): An optional fallback attribute if the primary attribute is not set.

        Returns:
            int: The ID of the special token or the unknown token ID if not found.
        """
        token_id = getattr(self.scaffold_tokenizer, token_attr, None)
        if token_id is None and fallback:
            token_id = getattr(self.scaffold_tokenizer, fallback, None)
        return token_id or self.scaffold_tokenizer.unk_token_id

    def _normalize_token(self, token: str) -> str:
        """
        Normalize a token for compatibility between tokenizers.

        Args:
            token (str): The token to normalize.

        Returns:
            str: The normalized token.
        """
        return token.replace("Ġ", "").replace("##", "")

    def get_token_map(self) -> Dict[int, Dict[str, Union[List[int], float]]]:
        """
        Retrieve the token map.

        Returns:
            Dict[int, Dict[str, Union[List[int], float]]]: The token map.
        """
        return self.token_map

    def get_special_token_map(self) -> Dict[int, int]:
        """
        Retrieve the special token map.

        Returns:
            Dict[int, int]: The special token map.
        """
        return self.special_token_map

    def log_token_mapping(self, sample_size: int = 10):
        """
        Log a sample of the token mapping for debugging.

        Args:
            sample_size (int): The number of token mappings to log.
        """
        self.logger.info("Logging a sample of the token map...")
        for i, (base_id, mapping) in enumerate(self.token_map.items()):
            if i >= sample_size:
                break
            self.logger.info(f"Base Token ID {base_id}: Scaffold IDs {mapping['ids']} (Weight: {mapping['weight']})")

        self.logger.info("Logging a sample of the special token map...")
        for base_id, scaffold_id in self.special_token_map.items():
            self.logger.info(f"Special Token ID {base_id}: Mapped to Scaffold Token ID {scaffold_id}")

    def tokenize_and_map(self, prompts, max_length=None):
        """Tokenize prompts and map to scaffold token space."""
        try:
            if not isinstance(prompts, list):
                prompts = [prompts]
            
            max_length = max_length or 128  # Default max length
            base_inputs = self.base_tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Map to scaffold space
            scaffold_inputs = self.map_sequence(base_inputs['input_ids'])
            
            return {
                'input_ids': scaffold_inputs,
                'attention_mask': base_inputs['attention_mask']
            }
        except Exception as e:
            self.logger.error(f"Token mapping failed: {str(e)}")
            raise

    def update_token_map_memory(self, prompt, confidence):
        """Update token map memory based on prompt confidence."""
        try:
            base_inputs = self.base_tokenizer(prompt, return_tensors='pt').to('cuda' if torch.cuda.is_available() else 'cpu')
            base_ids = base_inputs['input_ids'][0].tolist()
            
            # Update weights
            for base_id in base_ids:
                if base_id in self.token_map:
                    current_weight = self.token_map[base_id]['weight']
                    new_weight = min(1.0, current_weight + (confidence * 0.1))
                    self.token_map[base_id]['weight'] = new_weight
                
            self.logger.info(f"Token map updated with confidence: {confidence}")
        except Exception as e:
            self.logger.error(f"Token map memory update failed: {str(e)}")
            raise

    def map_sequence(self, base_input_ids: torch.Tensor) -> torch.Tensor:
        """Map base model token IDs to scaffold token IDs."""
        try:
            scaffold_ids = []
            for base_id in base_input_ids.tolist():
                scaffold_ids.append(self.token_map.get(base_id, [self.scaffold_tokenizer.unk_token_id]))
            return torch.tensor(scaffold_ids, device=base_input_ids.device)
        except Exception as e:
            self.logger.error(f"Sequence mapping failed: {str(e)}")
            raise


def build_scaffold_token_mapping(base_tokenizer, scaffold_tokenizer) -> ScaffoldTokenMapper:
    """
    Utility function to create a ScaffoldTokenMapper instance.

    Args:
        base_tokenizer: The tokenizer for the base model.
        scaffold_tokenizer: The tokenizer for the scaffold model.

    Returns:
        ScaffoldTokenMapper: An instance of the ScaffoldTokenMapper class.
    """
    return ScaffoldTokenMapper(base_tokenizer, scaffold_tokenizer)

class SparseMaskFactory:
    """Handles creation of sparse attention masks."""
    
    @staticmethod
    def create(
        seq_len: int,
        sparse_pattern: str,
        window_size: int,
        device: str = 'cpu',
        logger: Optional[Logger] = None
    ) -> torch.Tensor:
        """Create a sparse attention mask based on the specified pattern."""
        try:
            with NumericalGuard():
                if sparse_pattern == 'window':
                    mask = torch.zeros(seq_len, seq_len, device=device)
                    for i in range(seq_len):
                        start = max(0, i - window_size // 2)
                        end = min(seq_len, i + window_size // 2 + 1)
                        mask[i, start:end] = 1.0
                    return mask.bool()
                elif sparse_pattern == 'block':
                    mask = torch.zeros(seq_len, seq_len, device=device)
                    for i in range(0, seq_len, window_size):
                        mask[i:i + window_size, i:i + window_size] = 1.0
                    return mask.bool()
                else:
                    raise ValueError(f"Unknown sparse pattern: {sparse_pattern}")
        except Exception as e:
            if logger:
                logger.record({
                    "error": f"Sparse mask creation failed: {str(e)}",
                    "sparse_pattern": sparse_pattern,
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc()
                })
            raise

class AttentionMaskPreparer:
    """Prepares attention masks for multi-head attention."""
    
    @staticmethod
    def prepare(
        attention_mask: torch.Tensor,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        device: torch.device,
        logger: Optional[Logger] = None
    ) -> torch.Tensor:
        """Prepare attention mask in additive format."""
        try:
            with NumericalGuard():
                if attention_mask is None:
                    return None
                if attention_mask.dim() < 2 or attention_mask.dim() > 4:
                    raise ValueError(f"Invalid attention mask dimensions: {attention_mask.shape}")
                if attention_mask.dtype == torch.bool:
                    attention_mask = attention_mask.float().masked_fill(~attention_mask, float('-inf'))
                elif attention_mask.dtype != torch.float:
                    attention_mask = attention_mask.float()
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(0).unsqueeze(1)
                elif attention_mask.dim() == 3:
                    attention_mask = attention_mask.unsqueeze(1)
                if attention_mask.shape != (batch_size, num_heads, seq_len, seq_len):
                    attention_mask = attention_mask.expand(batch_size, num_heads, seq_len, seq_len)
                return attention_mask.to(device)
        except Exception as e:
            if logger:
                logger.record({
                    "error": f"Attention mask preparation failed: {str(e)}",
                    "mask_shape": list(attention_mask.shape) if attention_mask is not None else None,
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc()
                })
            raise

class LayerDiscoveryStrategy:
    """Strategy for discovering transformer layers in a model."""
    
    def __init__(self, logger: Logger):
        self.logger = logger
        self.patterns = [
            'h.{i}',
            'layer.{i}',
            'layers.{i}',
            'transformer.h.{i}',
            'decoder.layers.{i}',
        ]

    def find_layers(self, model: nn.Module) -> Tuple[List[nn.Module], List[str]]:
        """Find layers suitable for cross-attention injection."""
        try:
            candidates = []
            names = []

            # Try specific patterns first
            for name, module in model.named_modules():
                if any(pattern.split('.')[0] in name for pattern in self.patterns):
                    if isinstance(module, nn.ModuleList):
                        candidates.extend(module)
                        names.extend([f"{name}.{i}" for i in range(len(module))])

            # Fallback to any ModuleList
            if not candidates:
                for name, module in model.named_modules():
                    if isinstance(module, nn.ModuleList):
                        candidates.extend(module)
                        names.extend([f"{name}.{i}" for i in range(len(module))])

            # Last resort: collect modules with 'layer' in name
            if not candidates:
                for name, module in model.named_modules():
                    if 'layer' in name.lower() and isinstance(module, nn.Module):
                        candidates.append(module)
                        names.append(name)

            if not candidates:
                raise ValueError("No suitable layers found for cross-attention injection")

            return candidates, names
        except Exception as e:
            self.logger.record({
                "error": f"Layer discovery failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

class CrossAttentionLayer(nn.Module):
    """Implements cross-attention between base and scaffold models."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger: Any,
        hidden_size: Optional[int] = None,
        num_heads: Optional[int] = None,
        device: str = 'cpu'
    ):
        """
        Initialize cross-attention layer.
        
        Args:
            config: Configuration dictionary with layer parameters
            logger: Logger instance for logging
            hidden_size: Optional hidden size override
            num_heads: Optional number of attention heads override
            device: Device for tensor operations
        """
        super().__init__()
        self.config = config
        self.logger = logger
        self.device = device
        
        # Load configuration
        self._load_config(hidden_size, num_heads)
        
        # Initialize components
        self._init_projections()
        self._init_gating()
        self._init_normalization()
        self._init_sparse_mask()
        self._init_weights()
        
        # Initialize cache
        self.reset_cache()
        
    def _load_config(self, hidden_size: Optional[int], num_heads: Optional[int]) -> None:
        """Load configuration parameters."""
        self.hidden_size = hidden_size or self.config.get('hidden_size', 768)
        self.num_heads = num_heads or self.config.get('num_heads', 12)
        self.head_dim = self.hidden_size // self.num_heads
        
        # Validate configuration
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
            )
            
        self.logger.record({
            "event": "cross_attention_config_loaded",
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim
        })
        
    def _init_projections(self) -> None:
        """Initialize projection layers."""
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
    def _init_gating(self) -> None:
        """Initialize gating mechanism."""
        self.gate = nn.Parameter(torch.ones(1))
        self.gate_bias = nn.Parameter(torch.zeros(1))
        
    def _init_normalization(self) -> None:
        """Initialize normalization layers."""
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
    def _init_sparse_mask(self) -> None:
        """Initialize sparse attention mask."""
        self.sparse_mask = None
        
    def _init_weights(self) -> None:
        """Initialize layer weights."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
        
    def reset_cache(self) -> None:
        """Reset attention cache."""
        self.cache = {
            'k': None,
            'v': None,
            'attention_mask': None
        }
        
    def set_influence_weight(self, weight: float) -> None:
        """Set influence weight for cross-attention."""
        self.gate.data.fill_(weight)
        
    def set_blend_strength(self, strength: float) -> None:
        """Set blend strength for cross-attention."""
        self.gate_bias.data.fill_(strength)
        
    def set_lifecycle_weight(self, weight: float, curve: str = 'sigmoid_linear') -> None:
        """Set lifecycle-based weight adjustment."""
        if curve == 'sigmoid_linear':
            self.gate.data.fill_(torch.sigmoid(torch.tensor(weight * 2 - 1)))
        elif curve == 'linear':
            self.gate.data.fill_(weight)
        else:
            raise ValueError(f"Unknown curve type: {curve}")
            
    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        seq_len: int,
        batch_size: int
    ) -> torch.Tensor:
        """Compute attention scores and apply mask."""
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
            
        # Apply sparse mask if available
        if self.sparse_mask is not None:
            scores = scores.masked_fill(self.sparse_mask == 0, float('-inf'))
            
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back to original dimensions
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        return attn_output
        
    def _forward(
        self,
        hidden_states: torch.Tensor,
        cross_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory_tensors: Optional[torch.Tensor] = None,
        memory_weight: float = 0.0,
        dynamic_factor: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> torch.Tensor:
        """Forward pass implementation."""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Apply layer normalization
        hidden_states = self.layer_norm(hidden_states)
        
        # Project queries, keys, and values
        q = self.q_proj(hidden_states)
        k = self.k_proj(cross_states)
        v = self.v_proj(cross_states)
        
        # Apply memory if provided
        if memory_tensors is not None and memory_weight > 0:
            k = k + memory_tensors[0] * memory_weight
            v = v + memory_tensors[1] * memory_weight
            
        # Compute attention
        attn_output = self._compute_attention(
            q, k, v, attention_mask, seq_len, batch_size
        )
        
        # Apply output projection
        attn_output = self.out_proj(attn_output)
        
        # Apply gating
        attn_output = attn_output * self.gate + self.gate_bias
        
        # Apply dynamic factor if provided
        if dynamic_factor is not None:
            attn_output = attn_output * dynamic_factor
            
        # Update cache if enabled
        if use_cache:
            self.cache['k'] = k
            self.cache['v'] = v
            self.cache['attention_mask'] = attention_mask
            
        return attn_output
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory_tensors: Optional[torch.Tensor] = None,
        memory_weight: float = 0.0,
        dynamic_factor: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> torch.Tensor:
        """Forward pass with error handling."""
        try:
            return self._forward(
                hidden_states, cross_states, attention_mask,
                memory_tensors, memory_weight, dynamic_factor, use_cache
            )
        except Exception as e:
            self.logger.record({
                "error": f"CrossAttentionLayer forward pass failed: {str(e)}",
                "stack_trace": traceback.format_exc()
            })
            raise

class CrossAttentionInjector:
    """Injector for adding cross-attention layers to a transformer model."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        self.config_manager = config_manager
        self.logger = logger
        self.lock = Lock()
        self.scaffold_proj = None
        self.scaffold_unk_id = config_manager.get("controls_config.scaffold_unk_id", 0)
        self.error_handler = ErrorHandler(config_manager, logger)

    def inject(
        self,
        base_model: nn.Module,
        scaffold_model: nn.Module,
        layers_to_inject: Union[str, List[int]],
        injection_strategy: str = 'sequential',
        token_map: Optional[Dict] = None
    ) -> nn.Module:
        """Inject cross-attention into the model."""
        try:
            with self.lock:
                # Determine layers to inject
                layer_indices = self.get_cross_attention_layers(base_model, layers_to_inject)
                
                # Backup original state
                original_state = {name: param.clone() for name, param in base_model.named_parameters()}
                
                self.logger.record({
                    "event": "cross_attention_injection_start",
                    "layers": layer_indices,
                    "timestamp": time.time()
                })
                
                # Inject layers
                for layer_idx in layer_indices:
                    self._inject_single_layer(
                        model=base_model,
                        scaffold_model=scaffold_model,
                        layer_idx=layer_idx,
                        injection_strategy=injection_strategy,
                        token_map=token_map
                    )
                
                # Verify injection
                if not self.verify_injection(base_model, layer_indices, base_model.config):
                    for name, param in base_model.named_parameters():
                        if name in original_state:
                            param.data.copy_(original_state[name])
                    raise RuntimeError("Cross-attention injection verification failed")
                
                self.logger.record({
                    "event": "cross_attention_injection_complete",
                    "status": "success",
                    "timestamp": time.time()
                })
                
                return base_model
                
        except Exception as e:
            self.logger.record({
                "error": f"Cross-attention injection failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    def _inject_single_layer(
        self,
        model: nn.Module,
        scaffold_model: nn.Module,
        layer_idx: int,
        injection_strategy: str,
        token_map: Optional[Dict]
    ) -> None:
        """Inject cross-attention into a single layer."""
        try:
            layers, _ = self.find_model_layers(model)
            if layer_idx >= len(layers):
                raise ValueError(f"Layer index {layer_idx} out of bounds")
                
            original_layer = layers[layer_idx]
            cross_attn_layer = CrossAttentionLayer(
                config=self.config_manager.get("core_config"),
                logger=self.logger,
                device=str(model.device)
            )
            
            # Create wrapped layer
            wrapped_layer = self._create_wrapped_layer(
                original_layer=original_layer,
                cross_attn_layer=cross_attn_layer,
                scaffold_model=scaffold_model,
                token_map=token_map,
                strategy=injection_strategy
            )
            
            layers[layer_idx] = wrapped_layer
            
            if not self._verify_single_layer(model, layer_idx):
                raise RuntimeError(f"Layer {layer_idx} injection verification failed")
                
            self.logger.record({
                "event": "layer_injected",
                "layer_idx": layer_idx,
                "strategy": injection_strategy,
                "timestamp": time.time()
            })
            
        except Exception as e:
            self.logger.record({
                "error": f"Failed to inject layer {layer_idx}: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    def get_cross_attention_layers(self, model: nn.Module, mode: Union[str, List[int]]) -> List[int]:
        """Determine which layers to inject cross-attention into."""
        try:
            if isinstance(mode, list):
                layers = mode
                total_layers = len(self.find_model_layers(model)[0])
                if not validate_layer_indices(layers, total_layers):
                    raise ValueError(f"Invalid layer indices: {layers}")
            else:
                total_layers = 0
                if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                    total_layers = len(model.transformer.h)
                elif hasattr(model, 'layers'):
                    total_layers = len(model.layers)
                elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
                    total_layers = len(model.model.layers)
                elif hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
                    total_layers = len(model.decoder.layers)

                if total_layers == 0:
                    raise ValueError("No layers found for cross-attention injection")

                if mode == "early":
                    layers = list(range(total_layers // 3))
                elif mode == "late":
                    layers = list(range(2 * total_layers // 3, total_layers))
                else:  # balanced
                    layers = list(range(total_layers // 3, 2 * total_layers // 3))

            self.logger.record({
                "event": "layer_selection",
                "mode": str(mode),
                "selected_layers": layers,
                "total_layers": total_layers if isinstance(mode, str) else len(self.find_model_layers(model)[0]),
                "timestamp": time.time()
            })
            return layers
        except Exception as e:
            self.logger.record({
                "error": f"Layer selection failed: {str(e)}",
                "mode": str(mode),
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    def find_model_layers(self, model: nn.Module) -> Tuple[List[nn.Module], List[str]]:
        """Find transformer layers in the model."""
        strategy = LayerDiscoveryStrategy(self.logger)
        return strategy.find_layers(model)

    def _create_wrapped_layer(
        self,
        original_layer: nn.Module,
        cross_attn_layer: CrossAttentionLayer,
        scaffold_model: nn.Module,
        token_map: Optional[Dict],
        strategy: str
    ) -> nn.Module:
        """Create a wrapped layer based on injection strategy."""
        class WrappedLayer(nn.Module):
            def __init__(self, base_layer, cross_attn, scaffold, token_map, parent, strategy):
                super().__init__()
                self.base_layer = base_layer if strategy != 'replace' else None
                self.cross_attn = cross_attn
                self.scaffold = scaffold
                self.token_map = token_map or defaultdict(lambda: [parent.scaffold_unk_id])
                self.strategy = strategy
                self.parent = parent
                self.combine = nn.Linear(cross_attn.hidden_size * 2, cross_attn.hidden_size) if strategy == 'parallel' else None

            def forward(self, hidden_states, *args, scaffold_context=None, **kwargs):
                try:
                    if self.strategy == 'replace':
                        if scaffold_context is None:
                            return hidden_states
                        context = scaffold_context.to(hidden_states.device)
                        if self.parent.scaffold_proj is not None:
                            context = self.parent.scaffold_proj(context)
                        output = self.cross_attn(hidden_states, context, **kwargs)
                        return (output,) if isinstance(hidden_states, tuple) else output

                    base_output = self.base_layer(hidden_states, *args, **kwargs)
                    base_output = base_output[0] if isinstance(base_output, tuple) else base_output
                    
                    if scaffold_context is None:
                        return base_output
                    
                    context = scaffold_context.to(hidden_states.device)
                    if self.parent.scaffold_proj is not None:
                        context = self.parent.scaffold_proj(context)
                    cross_output = self.cross_attn(hidden_states, context, **kwargs)
                    
                    if self.strategy == 'parallel':
                        combined = torch.cat([base_output, cross_output], dim=-1)
                        output = self.combine(combined)
                    else:  # sequential
                        output = cross_output
                        
                    return (output,) + base_output[1:] if isinstance(base_output, tuple) else output
                except Exception as e:
                    self.parent.logger.record({
                        "error": f"WrappedLayer forward failed: {str(e)}",
                        "hidden_states_shape": list(hidden_states.shape),
                        "timestamp": time.time(),
                        "stack_trace": traceback.format_exc()
                    })
                    raise

        return WrappedLayer(original_layer, cross_attn_layer, scaffold_model, token_map, self, strategy)

    def _verify_single_layer(self, model: nn.Module, layer_idx: int) -> bool:
        """Verify a single layer's cross-attention injection."""
        try:
            layers, _ = self.find_model_layers(model)
            layer = layers[layer_idx]
            if not hasattr(layer, 'cross_attn'):
                return False
            if layer.cross_attn.hidden_size != layer.cross_attn.q_proj.in_features:
                return False
            return True
        except Exception:
            return False

    def verify_injection(self, model: nn.Module, expected_layers: List[int], base_config: Any) -> bool:
        """
        Verify that cross-attention layers were properly injected.

        Args:
            model: The model to verify
            expected_layers: List of expected layer indices
            base_config: Base model configuration for dimension validation

        Returns:
            bool: True if verification succeeds, False otherwise
        """
        try:
            expected_layers = set(expected_layers)
            found_layers = set()

            # Scan model for cross-attention layers
            for name, module in model.named_modules():
                if "cross_attention" in name.lower():
                    try:
                        # Extract layer index from module name
                        parts = name.split('.')
                        if len(parts) >= 3 and parts[0] == 'transformer' and parts[1] == 'h':
                            layer_idx = int(parts[2])
                            found_layers.add(layer_idx)
                    except (ValueError, IndexError):
                        continue
                    
            # Log verification results
            self.logger.record({
                "event": "cross_attention_verification",
                "expected_layers": list(expected_layers),
                "found_layers": list(found_layers),
                "timestamp": time.time()
            })

            # Check if all expected layers were found
            if not expected_layers.issubset(found_layers):
                missing_layers = expected_layers - found_layers
                self.logger.record({
                    "warning": f"Missing cross-attention layers: {missing_layers}",
                    "timestamp": time.time()
                })
                return False

            # Verify layer dimensions and structure
            for layer_idx in expected_layers:
                try:
                    layer = model.transformer.h[layer_idx]
                    if not hasattr(layer, 'cross_attention'):
                        self.logger.record({
                            "warning": f"Layer {layer_idx} missing cross_attention attribute",
                            "timestamp": time.time()
                        })
                        return False

                    # Verify dimensions match
                    if layer.cross_attention.hidden_size != base_config.hidden_size:
                        self.logger.record({
                            "warning": f"Layer {layer_idx} dimension mismatch",
                            "timestamp": time.time()
                        })
                        return False

                    # Verify attention heads match
                    if layer.cross_attention.num_attention_heads != base_config.num_attention_heads:
                        self.logger.record({
                            "warning": f"Layer {layer_idx} attention heads mismatch",
                            "timestamp": time.time()
                        })
                        return False
                except Exception as e:
                    self.error_handler.handle_cross_attention_error(e, layer_idx)
                    return False

            return True
        except Exception as e:
            self.error_handler.handle_cross_attention_error(e)
            return False

    def save_state(self, path: str, state_dict: dict):
        """Save cross-attention parameters."""
        try:
            with self.lock:
                torch.save(
                    {k: v for k, v in state_dict.items() if 'cross_attn' in k or 'scaffold_proj' in k},
                    path
                )
                self.logger.record({
                    "event": "save_state",
                    "path": path,
                    "timestamp": time.time()
                })
        except Exception as e:
            self.logger.record({
                "error": f"Failed to save cross-attention state: {str(e)}",
                "path": path,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    def load_state(self, path: str, model: nn.Module):
        """Load cross-attention parameters."""
        try:
            with self.lock:
                state_dict = model.state_dict()
                checkpoint_dict = torch.load(path, map_location=model.device)
                state_dict.update({k: v for k, v in checkpoint_dict.items() if k in state_dict})
                model.load_state_dict(state_dict)
                self.logger.record({
                    "event": "load_state",
                    "path": path,
                    "timestamp": time.time()
                })
        except Exception as e:
            self.logger.record({
                "error": f"Failed to load cross-attention state: {str(e)}",
                "path": path,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    def inject_cross_attention(
        self,
        model: nn.Module,
        scaffold_model: nn.Module,
        core_config: Dict[str, Any],
        cross_attn_config: Dict[str, Any],
        lora_config: Dict[str, Any],
        token_map: Optional[Dict] = None,
        device: Optional[torch.device] = None
    ) -> None:
        """
        Inject cross-attention layers into the model with full configuration support.

        Args:
            model: The base model to inject cross-attention into
            scaffold_model: The scaffold model providing context
            core_config: Core configuration containing layer specifications
            cross_attn_config: Cross-attention specific configuration
            lora_config: LoRA configuration for adapter layers
            token_map: Optional token mapping between models
            device: Device to perform operations on
        """
        try:
            if not cross_attn_config.get("enable_cross_attention", True):
                self.logger.record({
                    "event": "cross_attention",
                    "status": "disabled",
                    "timestamp": time.time()
                })
                return

            print("Injecting cross-attention layers...")
            
            # Get layer configuration
            layers_to_inject = core_config.get("cross_attn_layers", [])
            injection_strategy = cross_attn_config.get("injection_strategy", "sequential")
            
            # Perform injection
            self.inject(
                base_model=model,
                scaffold_model=scaffold_model,
                layers_to_inject=layers_to_inject,
                injection_strategy=injection_strategy,
                token_map=token_map
            )
            
            # Verify injection
            if not self.verify_injection(model, layers_to_inject, model.config):
                raise ValueError("Cross-attention layer verification failed")
                
            self.logger.record({
                "event": "cross_attention_injected",
                "timestamp": time.time()
            })
            print("Cross-attention injection complete.")
            
        except Exception as e:
            self.error_handler.handle_cross_attention_error(e)
            raise

def calculate_confidence_score(logits: torch.Tensor, generated_ids: torch.Tensor) -> float:
    """Calculate confidence score for scaffold generation."""
    try:
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            selected_probs = probs[torch.arange(len(generated_ids)), generated_ids]
            return float(selected_probs.mean())
    except Exception as e:
        raise RuntimeError(f"Failed to calculate confidence score: {str(e)}")

class InsufficientDataError(Exception):
    """Exception raised when there is insufficient data for scaffold operations."""
    pass

class ScaffoldManager:
    """Manages scaffold-related operations and state."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        self.config_manager = config_manager
        self.logger = logger
        self.token_map = None
        self.scaffold_hidden_states = None
        self.scaffold_config = None
        self.validation_cache = {}
        self.lock = Lock()

    def validate_scaffold_config(self) -> bool:
        """Validate scaffold-specific configuration settings."""
        try:
            required_keys = [
                "core_config.scaffold_model_name",
                "core_config.cross_attn_layers",
                "controls_config.scaffold_weight_cap",
                "controls_config.scaffold_unk_id"
            ]
            
            for key in required_keys:
                if not self.config_manager.has_key(key):
                    self.logger.record({
                        "error": f"Missing required scaffold config key: {key}",
                        "timestamp": time.time()
                    })
                    return False

            cross_attn_layers = self.config_manager.get("core_config.cross_attn_layers", [])
            if not isinstance(cross_attn_layers, list):
                self.logger.record({
                    "error": "cross_attn_layers must be a list",
                    "timestamp": time.time()
                })
                return False

            numeric_validations = {
                "controls_config.scaffold_weight_cap": (0.0, 1.0),
                "controls_config.blend_strength": (0.0, 1.0),
                "controls_config.attention_weight": (0.0, None),
                "controls_config.memory_weight": (0.0, 1.0)
            }

            for key, (min_val, max_val) in numeric_validations.items():
                if self.config_manager.has_key(key):
                    value = self.config_manager.get(key)
                    if not isinstance(value, (int, float)):
                        self.logger.record({
                            "error": f"{key} must be numeric",
                            "timestamp": time.time()
                        })
                        return False
                    if min_val is not None and value < min_val:
                        self.logger.record({
                            "error": f"{key} must be >= {min_val}",
                            "timestamp": time.time()
                        })
                        return False
                    if max_val is not None and value > max_val:
                        self.logger.record({
                            "error": f"{key} must be <= {max_val}",
                            "timestamp": time.time()
                        })
                        return False

            self.validation_cache["config_valid"] = True
            return True

        except Exception as e:
            self.logger.record({
                "error": f"Scaffold config validation failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            return False

    def initialize_scaffold_state(self, model_name: str, device: str) -> bool:
        """Initialize scaffold model state and configuration."""
        try:
            from transformers import AutoConfig
            self.scaffold_config = AutoConfig.from_pretrained(model_name)
            
            required_attrs = ["hidden_size", "num_attention_heads", "num_hidden_layers"]
            for attr in required_attrs:
                if not hasattr(self.scaffold_config, attr):
                    raise ValueError(f"Scaffold model config missing {attr}")

            with self.lock:
                self.hidden_size = self.scaffold_config.hidden_size
                self.num_heads = self.scaffold_config.num_attention_heads
                self.num_layers = self.scaffold_config.num_hidden_layers
                self.device = device

            return True
        except Exception as e:
            self.logger.record({
                "error": f"Scaffold state initialization failed: {str(e)}",
                "model_name": model_name,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            return False

    def verify_scaffold_compatibility(self, base_config) -> bool:
        """Verify compatibility between base and scaffold models."""
        try:
            if not self.scaffold_config:
                raise ValueError("Scaffold config not initialized")

            if self.scaffold_config.hidden_size % base_config.hidden_size != 0:
                self.logger.record({
                    "error": "Incompatible hidden sizes",
                    "scaffold_size": self.scaffold_config.hidden_size,
                    "base_size": base_config.hidden_size,
                    "timestamp": time.time()
                })
                return False

            if self.scaffold_config.num_attention_heads % base_config.num_attention_heads != 0:
                self.logger.record({
                    "error": "Incompatible number of attention heads",
                    "scaffold_heads": self.scaffold_config.num_attention_heads,
                    "base_heads": base_config.num_attention_heads,
                    "timestamp": time.time()
                })
                return False

            return True
        except Exception as e:
            self.logger.record({
                "error": f"Scaffold compatibility check failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            return False

    def get_scaffold_stats(self) -> Dict:
        """Get current statistics about scaffold state."""
        try:
            with self.lock:
                stats = {
                    "hidden_size": getattr(self, "hidden_size", None),
                    "num_heads": getattr(self, "num_heads", None),
                    "num_layers": getattr(self, "num_layers", None),
                    "token_map_size": len(self.token_map) if self.token_map else 0,
                    "has_hidden_states": self.scaffold_hidden_states is not None,
                    "config_valid": self.validation_cache.get("config_valid", False),
                    "device": str(self.device) if hasattr(self, "device") else None,
                    "timestamp": time.time()
                }
                return stats
        except Exception as e:
            self.logger.record({
                "error": f"Failed to get scaffold stats: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            return {}

    def reset_scaffold_state(self):
        """Reset all scaffold-related state."""
        with self.lock:
            self.token_map = None
            self.scaffold_hidden_states = None
            self.validation_cache.clear()
            self.logger.record({
                "event": "scaffold_state_reset",
                "timestamp": time.time()
            })

    def build_token_map(self, base_tokenizer, scaffold_tokenizer):
        """Build mapping between base and scaffold tokenizers."""
        try:
            token_map = defaultdict(lambda: [scaffold_tokenizer.unk_token_id])
            for base_token, base_id in base_tokenizer.get_vocab().items():
                normalized = base_token.replace("Ġ", "").replace("##", "")
                scaffold_ids = scaffold_tokenizer.encode(
                    normalized,
                    add_special_tokens=False,
                    max_length=3,
                    truncation=True
                ) or [scaffold_tokenizer.unk_token_id]
                token_map[base_id] = {'ids': scaffold_ids, 'weight': 1.0}
                
            special_token_map = {
                base_tokenizer.pad_token_id: scaffold_tokenizer.pad_token_id,
                base_tokenizer.eos_token_id: scaffold_tokenizer.eos_token_id or scaffold_tokenizer.sep_token_id,
                base_tokenizer.unk_token_id: scaffold_tokenizer.unk_token_id,
            }
            token_map.update(special_token_map)
            
            self.logger.record({
                "event": "token_map_built",
                "map_size": len(token_map),
                "timestamp": time.time()
            })
            
            return token_map
        except Exception as e:
            self.logger.record({
                "error": f"Token map building failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    @contextlib.contextmanager
    def scaffold_context(self, hidden_states):
        """Context manager for scaffold hidden states."""
        try:
            prev_states = self.scaffold_hidden_states
            self.scaffold_hidden_states = hidden_states
            yield
        finally:
            self.scaffold_hidden_states = prev_states

    def get_scaffold_hidden_states(self, scaffold_inputs):
        """Get hidden states from scaffold inputs."""
        try:
            return self.scaffold_hidden_states
        except Exception as e:
            self.logger.record({
                "error": f"Failed to get scaffold hidden states: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    def clear_scaffold_cache(self):
        """Clear scaffold hidden states cache."""
        self.scaffold_hidden_states = None

    def map_sequence(self, base_input_ids: torch.Tensor) -> torch.Tensor:
        """Map base model token IDs to scaffold token IDs."""
        try:
            if self.token_map is None:
                raise ValueError("Token map not initialized")
                
            scaffold_ids = []
            for base_id in base_input_ids.tolist():
                scaffold_ids.append(self.token_map.get(base_id, self.config_manager.get("controls_config.scaffold_unk_id", 0)))
            return torch.tensor(scaffold_ids, device=base_input_ids.device)
        except Exception as e:
            self.logger.record({
                "error": f"Sequence mapping failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise
