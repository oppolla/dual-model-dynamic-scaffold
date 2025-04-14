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
    """Enhanced cross-attention layer with gating, memory efficiency, and dynamic control."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        hidden_size: Optional[int] = None,
        num_heads: Optional[int] = None,
        device: str = 'cpu'
    ):
        super().__init__()
        self.config_manager = config_manager
        self.logger = logger
        self.device = torch.device(device)
        self.lock = Lock()
        
        # Load and validate configuration
        self._load_config(hidden_size, num_heads)
        
        # Initialize components
        self._init_projections()
        self._init_gating()
        self._init_normalization()
        self._init_sparse_mask()
        self._init_weights()
        
        # Key-value cache
        self.use_cache = False
        self.kv_cache = None

    def _load_config(self, hidden_size: Optional[int], num_heads: Optional[int]):
        """Load and validate configuration parameters."""
        self.hidden_size = hidden_size or self.config_manager.get("core_config.hidden_size", 768)
        self.num_heads = num_heads or self.config_manager.get("core_config.num_heads", 12)
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.head_dim = self.hidden_size // self.num_heads
        self.dropout_rate = self.config_manager.get("controls_config.dropout_rate", 0.1)
        self.use_gating = self.config_manager.get("controls_config.use_gating", True)
        self.scale_attention = self.config_manager.get("controls_config.scale_attention", True)
        self.use_residual = self.config_manager.get("controls_config.use_residual", True)
        self.use_pooling = self.config_manager.get("controls_config.use_pooling", False)
        self.use_sparse_attention = self.config_manager.get("controls_config.use_sparse_attention", False)
        self.gradient_checkpointing = self.config_manager.get("core_config.gradient_checkpointing", False)
        self.quantization_mode = self.config_manager.get("core_config.quantization", "fp16")
        self.sparse_pattern = self.config_manager.get("controls_config.sparse_pattern", "window")
        self.sparse_window_size = self.config_manager.get("controls_config.sparse_window_size", 128)
        self.max_seq_len = self.config_manager.get("training_config.max_seq_length", 512)
        
        self.scale = self.head_dim ** -0.5 if self.scale_attention else 1.0
        self.influence_weight = nn.Parameter(
            torch.tensor(self.config_manager.get("controls_config.scaffold_weight_cap", 1.0)),
            requires_grad=False
        )
        self.blend_strength = nn.Parameter(
            torch.tensor(self.config_manager.get("controls_config.dream_prompt_weight", 0.5)),
            requires_grad=False
        )

    def _init_projections(self):
        """Initialize attention projection layers with quantization support."""
        try:
            if self.quantization_mode in ['int8', 'int4']:
                try:
                    from bitsandbytes.nn import Linear8bitLt, Linear4bit
                    LinearCls = Linear8bitLt if self.quantization_mode == 'int8' else Linear4bit
                    self.q_proj = LinearCls(self.hidden_size, self.hidden_size)
                    self.k_proj = LinearCls(self.hidden_size, self.hidden_size)
                    self.v_proj = LinearCls(self.hidden_size, self.hidden_size)
                    self.out_proj = LinearCls(self.hidden_size, self.hidden_size)
                    return
                except ImportError:
                    self.logger.record({
                        "warning": f"bitsandbytes not installed, falling back to fp16",
                        "timestamp": time.time()
                    })
                    self.quantization_mode = 'fp16'

            self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
            self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
            self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
            self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        except Exception as e:
            self.logger.record({
                "error": f"Projection initialization failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    def _init_gating(self):
        """Initialize gating mechanism if enabled."""
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Sigmoid()
        ) if self.use_gating else None

    def _init_normalization(self):
        """Initialize normalization and dropout."""
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_rate)

    def _init_sparse_mask(self):
        """Initialize sparse attention mask."""
        self.sparse_mask = (SparseMaskFactory.create(
            seq_len=self.max_seq_len,
            sparse_pattern=self.sparse_pattern,
            window_size=self.sparse_window_size,
            device=self.device,
            logger=self.logger
        ) if self.use_sparse_attention else None)

    def _init_weights(self):
        """Initialize weights with model-specific scaling."""
        gain = self.config_manager.get("core_config.initializer_range", 0.02)
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight, gain=gain)
                nn.init.constant_(module.bias, 0.0)
        if self.use_gating:
            nn.init.xavier_uniform_(self.gate[0].weight, gain=gain)
            nn.init.constant_(self.gate[0].bias, 0.0)

    def reset_cache(self):
        """Reset key-value cache for inference."""
        with self.lock:
            self.kv_cache = None
            self.use_cache = False
            self.logger.record({
                "event": "cache_reset",
                "timestamp": time.time()
            })

    def set_influence_weight(self, weight: float):
        """Set the influence weight for attention output."""
        with self.lock:
            self.influence_weight.data = torch.tensor(max(0.0, weight), dtype=torch.float)

    def set_blend_strength(self, strength: float):
        """Set the blend strength for residual connection."""
        with self.lock:
            self.blend_strength.data = torch.tensor(max(0.0, min(1.0, strength)), dtype=torch.float)

    def set_lifecycle_weight(self, weight: float, curve: str = 'sigmoid_linear'):
        """Adjust influence weight based on lifecycle curve."""
        try:
            with NumericalGuard():
                if curve == 'sigmoid_linear':
                    adjusted = safe_divide(weight, 1 + torch.exp(-weight), logger=self.logger)
                elif curve == 'exponential':
                    adjusted = weight * torch.exp(-weight)
                else:
                    adjusted = weight
                self.set_influence_weight(adjusted)
        except Exception as e:
            self.logger.record({
                "error": f"Lifecycle weight adjustment failed: {str(e)}",
                "curve": curve,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        seq_len: int,
        batch_size: int
    ) -> torch.Tensor:
        """Compute attention mechanism."""
        if self.use_sparse_attention:
            if self.sparse_mask.shape[-1] < seq_len:
                self.sparse_mask = SparseMaskFactory.create(
                    seq_len=seq_len,
                    sparse_pattern=self.sparse_pattern,
                    window_size=self.sparse_window_size,
                    device=k.device,
                    logger=self.logger
                )
            sparse_mask = self.sparse_mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(1)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_scores = attn_scores + (~sparse_mask).float() * float('-inf')
            attn_probs = torch.softmax(attn_scores, dim=-1)
            return torch.matmul(attn_probs, v)
        elif hasattr(F, 'scaled_dot_product_attention'):
            return F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout_rate if self.training else 0.0,
                is_causal=False
            )
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask
            attn_probs = torch.softmax(attn_scores, dim=-1)
            return torch.matmul(attn_probs, v)

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
        """Core forward pass logic."""
        try:
            with NumericalGuard(dtype=torch.float16 if self.quantization_mode == 'fp16' else torch.float32):
                batch_size = hidden_states.size(0)
                
                # Validate inputs
                if hidden_states.dim() != 3 or cross_states.dim() != 3:
                    raise ValueError(f"Expected 3D tensors, got hidden_states={hidden_states.shape}, cross_states={cross_states.shape}")
                if hidden_states.device != cross_states.device:
                    cross_states = cross_states.to(hidden_states.device)

                # Pool cross-states if enabled
                if self.use_pooling:
                    cross_states = cross_states.mean(dim=1, keepdim=True)

                # Blend with memory tensors
                if memory_tensors is not None and memory_weight > 0:
                    if memory_tensors.shape[-1] != cross_states.shape[-1]:
                        self.logger.record({
                            "warning": f"Memory tensors dimension mismatch: {memory_tensors.shape[-1]} vs {cross_states.shape[-1]}",
                            "timestamp": time.time()
                        })
                    else:
                        memory_weight = max(0.0, min(1.0, memory_weight))
                        cross_states = (1 - memory_weight) * cross_states + memory_weight * memory_tensors

                # Project queries, keys, values
                q = self.q_proj(hidden_states)
                k = self.k_proj(cross_states)
                v = self.v_proj(cross_states)

                # Reshape for multi-head attention
                q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

                # Cache keys and values
                with self.lock:
                    self.use_cache = use_cache
                    if use_cache:
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
                attention_mask = AttentionMaskPreparer.prepare(
                    attention_mask=attention_mask,
                    batch_size=batch_size,
                    num_heads=self.num_heads,
                    seq_len=seq_len,
                    device=q.device,
                    logger=self.logger
                )

                # Compute attention
                attn_output = self._compute_attention(q, k, v, attention_mask, seq_len, batch_size)

                # Reshape and project
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.view(batch_size, -1, self.hidden_size)
                attn_output = self.out_proj(attn_output)

                # Apply dynamic factor
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

        except Exception as e:
            self.logger.record({
                "error": f"CrossAttentionLayer forward failed: {str(e)}",
                "hidden_states_shape": list(hidden_states.shape),
                "cross_states_shape": list(cross_states.shape),
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

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
        """Forward pass with optional gradient checkpointing."""
        try:
            with torch.autocast(device_type=hidden_states.device.type, enabled=self.quantization_mode == 'fp16'):
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
        except Exception as e:
            self.logger.record({
                "error": f"CrossAttentionLayer forward failed: {str(e)}",
                "hidden_states_shape": list(hidden_states.shape),
                "cross_states_shape": list(cross_states.shape),
                "timestamp": time.time(),
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
                config_manager=self.config_manager,
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
