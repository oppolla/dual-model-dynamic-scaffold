from typing import Optional, Deque, Dict, Set, Tuple, DefaultDict, Any, List
from collections import deque, defaultdict
from dataclasses import dataclass, field
import torch
import uuid
from threading import Lock
import time
import traceback
import hashlib
from sovl_logger import Logger
from sovl_config import ConfigManager
from sovl_utils import NumericalGuard, safe_divide, safe_compare, synchronized
import json
import os
import threading
import collections

class StateError(Exception):
    """Raised for invalid state operations or data."""
    pass

@dataclass
class CuriosityConfig:
    """Configuration for curiosity-related parameters."""
    max_questions: int
    max_novelty_scores: int
    decay_rate: float
    hidden_size: int
    question_timeout: float

@dataclass
class ConversationConfig:
    """Configuration for conversation-related parameters."""
    max_messages: int

@dataclass
class SOVLConfig:
    """Configuration for SOVL state parameters."""
    # Core configuration
    dream_memory_maxlen: int
    temperament_history_maxlen: int
    confidence_history_maxlen: int
    hidden_size: int
    max_seen_prompts: int
    quantization_mode: str
    sleep_max_steps: int
    prompt_timeout: float
    temperament_decay_rate: float
    scaffold_unk_id: int
    lora_capacity: int
    max_dream_memory_mb: float = 512.0  # Default to 512MB max dream memory
    
    # Temperament configuration
    temperament_melancholy_noise: float = 0.1
    temperament_influence: float = 0.3
    temperament_base_temperature: float = 0.7
    temperament_swing_threshold: float = 0.2
    temperament_stability_threshold: float = 0.1
    
    # Training configuration
    learning_rate: float = 2e-5
    grad_accum_steps: int = 4
    weight_decay: float = 0.01
    warmup_steps: int = 0
    total_steps: int = 100000
    max_grad_norm: float = 1.0
    use_amp: bool = True
    max_patience: int = 2
    batch_size: int = 2
    max_epochs: int = 3
    validate_every_n_steps: int = 100
    checkpoint_interval: int = 1000
    checkpoint_path: str = "checkpoints/sovl_trainer"
    scheduler_type: str = "linear"
    cosine_min_lr: float = 1e-6
    warmup_ratio: float = 0.1
    dropout_rate: float = 0.1
    max_seq_length: int = 512
    metrics_to_track: List[str] = None
    
    # Lifecycle configuration
    enable_gestation: bool = True
    enable_sleep_training: bool = True
    enable_lifecycle_weighting: bool = True
    lifecycle_capacity_factor: float = 0.01
    lifecycle_curve: str = "sigmoid_linear"
    sleep_conf_threshold: float = 0.7
    sleep_log_min: int = 10
    exposure_gain_eager: int = 3
    exposure_gain_default: int = 2
    
    # Dream configuration
    dream_memory_weight: float = 0.1
    enable_dreaming: bool = True
    repetition_n: int = 3
    sigmoid_scale: float = 0.5
    sigmoid_shift: float = 5.0
    dream_noise_scale: float = 0.05
    dream_prompt_weight: float = 0.5
    dream_novelty_boost: float = 0.03
    dream_memory_decay: float = 0.95
    dream_prune_threshold: float = 0.1
    temp_melancholy_noise: float = 0.02
    enable_prompt_driven_dreams: bool = True
    dream_swing_var: float = 0.1
    dream_lifecycle_delta: float = 0.1
    dream_temperament_on: bool = True
    
    # Memory configuration
    memory_threshold: float = 0.85
    memory_decay_rate: float = 0.95
    use_scaffold_memory: bool = True
    use_token_map_memory: bool = True
    scaffold_weight: float = 1.0
    
    # Curiosity configuration
    weight_ignorance: float = 0.3
    weight_novelty: float = 0.7
    metrics_maxlen: int = 100
    novelty_threshold_spontaneous: float = 0.7
    novelty_threshold_curious: float = 0.5
    curiosity_decay_rate: float = 0.95
    curiosity_queue_maxlen: int = 10
    curiosity_question_timeout: float = 3600.0
    
    def __post_init__(self):
        if self.metrics_to_track is None:
            self.metrics_to_track = ["loss", "accuracy", "confidence"]
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters."""
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.grad_accum_steps >= 1, "Gradient accumulation steps must be at least 1"
        assert self.max_grad_norm > 0, "Max gradient norm must be positive"
        assert self.scheduler_type in ["linear", "cosine", "constant"], "Invalid scheduler type"
        assert self.lifecycle_curve in ["sigmoid_linear", "exponential"], "Invalid lifecycle curve"
        assert self.repetition_n >= 2, "Repetition check length must be at least 2"
        assert self.sigmoid_scale > 0, "Sigmoid scale must be positive"
        assert self.sigmoid_shift >= 0, "Sigmoid shift must be non-negative"
        assert self.dream_noise_scale >= 0, "Dream noise scale must be non-negative"
        assert 0 <= self.dream_prompt_weight <= 1, "Dream prompt weight must be in [0, 1]"
        assert self.dream_novelty_boost >= 0, "Dream novelty boost must be non-negative"
        assert 0 <= self.dream_memory_decay <= 1, "Dream memory decay must be in [0, 1]"
        assert 0 <= self.dream_prune_threshold <= 1, "Dream prune threshold must be in [0, 1]"
        assert self.dream_swing_var >= 0, "Dream swing variance must be non-negative"
        assert self.dream_lifecycle_delta >= 0, "Dream lifecycle delta must be non-negative"
        assert self.confidence_history_maxlen > 0, "Confidence history maxlen must be positive"
        assert self.temperament_history_maxlen > 0, "Temperament history maxlen must be positive"
        assert 0 <= self.weight_ignorance <= 1, "Weight ignorance must be in [0, 1]"
        assert 0 <= self.weight_novelty <= 1, "Weight novelty must be in [0, 1]"
        assert self.metrics_maxlen > 0, "Metrics maxlen must be positive"
        assert 0 <= self.novelty_threshold_spontaneous <= 1, "Novelty threshold spontaneous must be in [0, 1]"
        assert 0 <= self.novelty_threshold_curious <= 1, "Novelty threshold curious must be in [0, 1]"
        assert 0 <= self.curiosity_decay_rate <= 1, "Curiosity decay rate must be in [0, 1]"
        assert self.curiosity_queue_maxlen > 0, "Curiosity queue maxlen must be positive"
        assert self.curiosity_question_timeout > 0, "Curiosity question timeout must be positive"
        assert self.max_dream_memory_mb > 0, "Max dream memory MB must be positive"
        # Add temperament validation
        assert 0 <= self.temperament_melancholy_noise <= 1, "Temperament melancholy noise must be in [0, 1]"
        assert 0 <= self.temperament_influence <= 1, "Temperament influence must be in [0, 1]"
        assert 0 <= self.temperament_base_temperature <= 2, "Temperament base temperature must be in [0, 2]"
        assert 0 <= self.temperament_swing_threshold <= 1, "Temperament swing threshold must be in [0, 1]"
        assert 0 <= self.temperament_stability_threshold <= 1, "Temperament stability threshold must be in [0, 1]"

@dataclass
class TrainingState:
    """Manages training-specific state and metrics."""
    last_trained: float = 0.0
    last_weight: float = 0.0
    sleep_confidence_sum: float = 0.0
    sleep_confidence_count: int = 0
    data_exposure: float = 0.0
    lora_capacity: float = 0.0
    gestation_metrics: Dict[str, Any] = field(default_factory=dict)
    dream_metrics: Dict[str, Any] = field(default_factory=dict)
    sleep_metrics: Dict[str, Any] = field(default_factory=dict)

    def update_gestation_metrics(self, batch_size: int, avg_loss: float) -> None:
        """Update gestation training metrics."""
        self.gestation_metrics.update({
            "batch_size": batch_size,
            "avg_loss": avg_loss,
            "timestamp": time.time()
        })

    def update_dream_metrics(self, dream_prompt: str, is_novel: bool, memory_count: int) -> None:
        """Update dream cycle metrics."""
        self.dream_metrics.update({
            "dream_prompt": dream_prompt,
            "is_novel": is_novel,
            "memory_count": memory_count,
            "timestamp": time.time()
        })

    def update_sleep_metrics(self, batch_size: int, data_exposure: float) -> None:
        """Update sleep training metrics."""
        self.sleep_metrics.update({
            "batch_size": batch_size,
            "data_exposure": data_exposure,
            "timestamp": time.time()
        })

    def update_data_exposure(self, exposure: float) -> None:
        """Update data exposure."""
        self.data_exposure = exposure

    def get_state_hash(self) -> str:
        """Generate a hash of the current training state."""
        state_dict = {
            "last_trained": self.last_trained,
            "last_weight": self.last_weight,
            "sleep_confidence_sum": self.sleep_confidence_sum,
            "sleep_confidence_count": self.sleep_confidence_count,
            "data_exposure": self.data_exposure,
            "lora_capacity": self.lora_capacity
        }
        return hashlib.md5(json.dumps(state_dict, sort_keys=True).encode()).hexdigest()

def _load_config(config_manager: ConfigManager, section: str, key: str, default: Any) -> Any:
    """Safely load a configuration value with a default."""
    return config_manager.get(f"{section}.{key}", default)

class StateBase:
    """Base class for state management with common utilities."""
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        self.config_manager = config_manager
        self.logger = logger
        self.lock = Lock()

    def _log_event(self, event: str, message: str, level: str = "info", **kwargs):
        """Log an event with standardized fields."""
        self.logger.record({
            "event": event,
            "message": message,
            "level": level,
            "timestamp": time.time(),
            **kwargs
        })

    def _log_error(self, message: str, **kwargs):
        """Log an error with stack trace."""
        self.logger.record({
            "error": message,
            "timestamp": time.time(),
            "stack_trace": traceback.format_exc(),
            **kwargs
        })

    def _validate_number(self, value: Any, name: str, min_value: Optional[float] = None) -> float:
        """Validate a numeric value."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        if min_value is not None and value < min_value:
            raise ValueError(f"{name} must be >= {min_value}")
        return float(value)

    def _validate_tensor(self, tensor: torch.Tensor, expected_dim: int, name: str):
        """Validate a tensor's shape."""
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Invalid {name} type: {type(tensor)}")
        if tensor.shape[-1] != expected_dim:
            raise ValueError(f"{name} shape {tensor.shape} mismatches expected dimension {expected_dim}")

class CuriosityState(StateBase):
    """Manages curiosity-related state and question prioritization."""
    def __init__(self, config_manager: ConfigManager, logger: Logger, device: torch.device):
        super().__init__(config_manager, logger)
        self._config = self._load_curiosity_config()
        self.device = device
        self.unanswered_questions: Deque[Tuple[str, float, Optional[torch.Tensor]]] = deque(maxlen=self._config.max_questions)
        self.last_question_time: float = 0.0
        self.pressure: float = 0.0
        self.novelty_scores: Deque[float] = deque(maxlen=self._config.max_novelty_scores)
        self.question_count: int = 0
        self._log_event("curiosity_state_initialized", message="Curiosity state initialized", config=self._config)

    def _load_curiosity_config(self) -> CuriosityConfig:
        """Load curiosity configuration."""
        return CuriosityConfig(
            max_questions=_load_config(self.config_manager, "controls_config", "curiosity_queue_maxlen", 10),
            max_novelty_scores=_load_config(self.config_manager, "controls_config", "novelty_history_maxlen", 20),
            decay_rate=_load_config(self.config_manager, "controls_config", "curiosity_decay_rate", 0.9),
            hidden_size=_load_config(self.config_manager, "core_config", "hidden_size", 768),
            question_timeout=_load_config(self.config_manager, "controls_config", "curiosity_question_timeout", 3600.0)
        )

    @synchronized("lock")
    def update_question_history(self, question: str, timestamp: float) -> None:
        """Update question history and related state."""
        try:
            self.last_question_time = timestamp
            self.question_count += 1
            self._log_event(
                "question_history_updated",
                message="Question history updated",
                level="info",
                question=question,
                timestamp=timestamp,
                question_count=self.question_count,
                queue_length=len(self.unanswered_questions)
            )
            self._update_pressure()
        except Exception as e:
            self._log_error(f"Failed to update question history: {str(e)}")
            raise StateError(f"Question history update failed: {str(e)}")

    @synchronized("lock")
    def add_question(self, question: str, score: float, context_vector: Optional[torch.Tensor] = None):
        """Add a new question with score and optional context vector."""
        try:
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Question must be a non-empty string")
            score = self._validate_number(score, "Score", min_value=0.0)
            if context_vector is not None:
                self._validate_tensor(context_vector, self._config.hidden_size, "Context vector")
            with NumericalGuard():
                self.unanswered_questions.append((question, score, context_vector))
                self.question_count += 1
                self.last_question_time = time.time()
                self._update_pressure()
                self._log_event(
                    "question_added",
                    message="New question added to curiosity state",
                    level="info",
                    question=question,
                    score=score,
                    has_context_vector=context_vector is not None,
                    question_count=self.question_count,
                    queue_length=len(self.unanswered_questions)
                )
        except Exception as e:
            self._log_error(f"Failed to add question: {str(e)}", question=question, score=score)
            raise StateError(f"Add question failed: {str(e)}")

    @synchronized("lock")
    def prioritize_questions(self):
        """Sort unanswered questions by score."""
        try:
            sorted_questions = sorted(self.unanswered_questions, key=lambda x: x[1], reverse=True)
            self.unanswered_questions = deque(sorted_questions, maxlen=self._config.max_questions)
            self._log_event(
                "questions_prioritized",
                message="Questions prioritized by score",
                level="info",
                question_count=len(self.unanswered_questions)
            )
        except Exception as e:
            self._log_error("Question prioritization failed")
            raise StateError(f"Question prioritization failed: {str(e)}")

    @synchronized("lock")
    def prune_old_questions(self, timeout: float) -> None:
        """Remove questions older than timeout."""
        try:
            current_time = time.time()
            while self.unanswered_questions and current_time - self.last_question_time > timeout:
                question, _, _ = self.unanswered_questions.popleft()
                self._log_event(
                    "old_question_pruned",
                    message="Old question pruned",
                    level="info",
                    question=question
                )
            self._update_pressure()
        except Exception as e:
            self._log_error("Question pruning failed")
            raise StateError(f"Question pruning failed: {str(e)}")

    @synchronized("lock")
    def _update_pressure(self):
        """Update curiosity pressure based on questions and novelty."""
        try:
            with NumericalGuard():
                base_pressure = safe_divide(
                    len(self.unanswered_questions),
                    max(1, self._config.max_questions),
                    logger=self.logger
                )
                novelty_avg = safe_divide(
                    sum(self.novelty_scores),
                    max(1, len(self.novelty_scores)),
                    logger=self.logger
                ) if self.novelty_scores else 0.0
                self.pressure = base_pressure * (1.0 + novelty_avg) * self._config.decay_rate
                self.pressure = max(0.0, min(1.0, self.pressure))
                self._log_event(
                    "pressure_updated",
                    message="Curiosity pressure updated",
                    level="info",
                    pressure=self.pressure,
                    unanswered_count=len(self.unanswered_questions),
                    novelty_avg=novelty_avg
                )
        except Exception as e:
            self._log_error("Pressure update failed")
            raise StateError(f"Pressure update failed: {str(e)}")

    @synchronized("lock")
    def add_novelty_score(self, score: float):
        """Add a novelty score and decay existing scores."""
        try:
            score = self._validate_number(score, "Novelty score", min_value=0.0)
            with NumericalGuard():
                self.novelty_scores.append(score * self._config.decay_rate)
                self._update_pressure()
                self._log_event(
                    "novelty_score_added",
                    message="Novelty score added",
                    level="info",
                    score=score,
                    novelty_scores_count=len(self.novelty_scores)
                )
        except Exception as e:
            self._log_error(f"Failed to add novelty score: {str(e)}", score=score)
            raise StateError(f"Add novelty score failed: {str(e)}")

    @synchronized("lock")
    def get_context_vector(self) -> Optional[torch.Tensor]:
        """Compute a weighted average of context vectors from questions."""
        try:
            if not self.unanswered_questions:
                return None
            vectors = [v for _, _, v in self.unanswered_questions if v is not None]
            scores = [s for _, s, v in self.unanswered_questions if v is not None]
            if not vectors:
                return None
            with NumericalGuard():
                weights = torch.tensor(scores, dtype=torch.float32)
                weights = weights / (weights.sum() + 1e-8)
                stacked = torch.stack(vectors)
                weighted_avg = (stacked * weights.view(-1, 1)).sum(dim=0)
                self._log_event(
                    "context_vector_computed",
                    message="Context vector computed from questions",
                    level="info",
                    vector_shape=list(weighted_avg.shape),
                    question_count=len(vectors)
                )
                return weighted_avg
        except Exception as e:
            self._log_error("Context vector computation failed")
            return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize curiosity state to dictionary."""
        try:
            with self.lock:
                return {
                    "unanswered_questions": [
                        (q, s, v.cpu().numpy().tolist() if v is not None else None)
                        for q, s, v in self.unanswered_questions
                    ],
                    "last_question_time": self.last_question_time,
                    "pressure": self.pressure,
                    "novelty_scores": list(self.novelty_scores),
                    "question_count": self.question_count,
                    "version": "1.1"
                }
        except Exception as e:
            self._log_error("Curiosity state serialization failed")
            raise StateError(f"Curiosity state serialization failed: {str(e)}")

    def from_dict(self, data: Dict[str, Any]):
        """Load curiosity state from dictionary."""
        try:
            with self.lock:
                version = data.get("version", "1.0")
                if version not in ["1.0", "1.1"]:
                    self._log_event(
                        "unsupported_version",
                        message=f"Unsupported curiosity state version: {version}",
                        level="warning",
                        version=version
                    )
                self.unanswered_questions = deque(maxlen=self._config.max_questions)
                for q, s, v in data.get("unanswered_questions", []):
                    context_vector = (
                        torch.tensor(v, dtype=torch.float32, device=self.device)
                        if v is not None and len(v) == self._config.hidden_size else None
                    )
                    self.unanswered_questions.append((q, float(s), context_vector))
                self.last_question_time = float(data.get("last_question_time", 0.0))
                self.pressure = float(data.get("pressure", 0.0))
                self.novelty_scores = deque(
                    [float(s) for s in data.get("novelty_scores", [])],
                    maxlen=self._config.max_novelty_scores
                )
                self.question_count = int(data.get("question_count", 0))
                self._log_event(
                    "curiosity_state_loaded",
                    message="Curiosity state loaded from dictionary",
                    level="info",
                    question_count=self.question_count,
                    pressure=self.pressure,
                    version=version
                )
        except Exception as e:
            self._log_error("Failed to load curiosity state", data_keys=list(data.keys()))
            raise StateError(f"Curiosity state loading failed: {str(e)}")

    def generate_curiosity_question(self, state, tokenizer, model, context, spontaneous: bool = False) -> Optional[str]:
        """Generate a curiosity-driven question based on the current state."""
        try:
            if not self.config_manager.get("curiosity_enabled", True):
                return None
            question = "What is the meaning of life?"  # Placeholder logic
            self._log_event(
                "curiosity_question_generated",
                message="Curiosity-driven question generated",
                level="info",
                question=question,
                spontaneous=spontaneous
            )
            return question
        except Exception as e:
            self._log_error("Failed to generate curiosity question")
            return None

    def check_silence(self, state, tokenizer, model, context):
        """Check for prolonged silence and generate a question if needed."""
        try:
            current_time = time.time()
            if current_time - self.last_question_time > self._config.question_timeout:
                question = self.generate_curiosity_question(state, tokenizer, model, context, spontaneous=True)
                if question:
                    print(f"Curiosity Question: {question}")
                    self.last_question_time = current_time
        except Exception as e:
            self._log_error("Failed to check silence")

    def tune_curiosity(self, pressure: Optional[float] = None, decay_rate: Optional[float] = None, question_timeout: Optional[float] = None):
        """Tune curiosity parameters."""
        try:
            with self.lock:
                if pressure is not None:
                    self.pressure = self._validate_number(pressure, "Pressure", min_value=0.0)
                if decay_rate is not None:
                    self._config.decay_rate = self._validate_number(decay_rate, "Decay rate", min_value=0.0)
                if question_timeout is not None:
                    self._config.question_timeout = self._validate_number(question_timeout, "Question timeout", min_value=0.0)
                self._log_event(
                    "curiosity_tuned",
                    message="Curiosity parameters tuned",
                    level="info",
                    pressure=self.pressure,
                    decay_rate=self._config.decay_rate,
                    question_timeout=self._config.question_timeout
                )
        except Exception as e:
            self._log_error("Failed to tune curiosity")
            raise StateError(f"Tune curiosity failed: {str(e)}")

    def reset_for_conversation(self, conversation_id: str):
        """Reset curiosity state for a new conversation."""
        try:
            with self.lock:
                self.unanswered_questions.clear()
                self.last_question_time = time.time()
                self.pressure = 0.0
                self.novelty_scores.clear()
                self.question_count = 0
                self._log_event(
                    "curiosity_reset",
                    message="Curiosity state reset for new conversation",
                    level="info",
                    conversation_id=conversation_id
                )
        except Exception as e:
            self._log_error("Failed to reset curiosity state")
            raise StateError(f"Reset curiosity failed: {str(e)}")

class ConversationHistory:
    """Manages conversation messages with unique ID."""
    def __init__(self, maxlen: int, conversation_id: Optional[str] = None):
        self._config = ConversationConfig(max_messages=maxlen)
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.messages: Deque[Dict[str, str]] = deque(maxlen=self._config.max_messages)

    def add_message(self, role: str, content: str):
        """Add a message to the conversation."""
        try:
            if not isinstance(role, str) or not role.strip():
                raise ValueError("Role must be a non-empty string")
            if not isinstance(content, str):
                raise ValueError("Content must be a string")
            self.messages.append({"role": role, "content": content})
        except Exception as e:
            raise StateError(f"Failed to add message: {str(e)}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize conversation history to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "messages": list(self.messages)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], maxlen: int) -> 'ConversationHistory':
        """Create conversation history from dictionary."""
        history = cls(maxlen=maxlen, conversation_id=data.get("conversation_id"))
        for msg in data.get("messages", []):
            history.add_message(msg["role"], msg["content"])
        return history

class SOVLState(StateBase):
    """Manages the state of the SOVL system."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger, device: torch.device):
        super().__init__(config_manager, logger)
        self.device = device
        self.lock = threading.Lock()
        self._initialize_state()
        
    @synchronized("lock")
    def _initialize_state(self) -> None:
        """Initialize state components."""
        self.seen_prompts = set()
        self.temperament_score = 0.0
        self.last_temperament_score = 0.0
        self.confidence_history = deque(maxlen=self.config_manager.get("confidence_history_maxlen"))
        self.temperament_history = deque(maxlen=self.config_manager.get("temperament_history_maxlen"))
        self.dream_memory = deque(maxlen=self.config_manager.get("dream_memory_maxlen"))
        self.total_dream_memory_mb = 0.0
        self.history = ConversationHistory(
            maxlen=self.config_manager.get("max_messages"),
            conversation_id=str(uuid.uuid4())
        )
        
    @synchronized("lock")
    def add_dream_memory(self, tensor: torch.Tensor, weight: float, metadata: Dict) -> None:
        """Add a dream memory with device validation."""
        try:
            # Validate and move tensor to correct device
            tensor = self._validate_tensor_device(tensor)
            
            # Calculate memory size
            memory_size = tensor.element_size() * tensor.nelement() / (1024 * 1024)  # Convert to MB
            
            # Check memory limits
            if self.total_dream_memory_mb + memory_size > self.config_manager.get("max_dream_memory_mb"):
                self._log_warning("Memory limit exceeded, pruning old memories")
                self._prune_dream_memory()
                
            # Add to memory
            self.dream_memory.append({
                "tensor": tensor,
                "weight": weight,
                "metadata": metadata,
                "timestamp": time.time()
            })
            self.total_dream_memory_mb += memory_size
            
        except Exception as e:
            self._log_error(f"Failed to add dream memory: {str(e)}")
            raise
            
    @synchronized("lock")
    def _prune_dream_memory(self) -> None:
        """Prune old memories to maintain memory limits."""
        while self.total_dream_memory_mb > self.config_manager.get("max_dream_memory_mb") and self.dream_memory:
            removed = self.dream_memory.popleft()
            memory_size = removed["tensor"].element_size() * removed["tensor"].nelement() / (1024 * 1024)
            self.total_dream_memory_mb -= memory_size
            
    @synchronized("lock")
    def get_dream_memory_stats(self) -> Dict[str, Any]:
        """Get dream memory statistics with device validation."""
        try:
            stats = {
                "count": len(self.dream_memory),
                "total_memory_mb": self.total_dream_memory_mb,
                "max_memory_mb": self.config_manager.get("max_dream_memory_mb"),
                "average_weight": 0.0,
                "oldest_timestamp": None,
                "newest_timestamp": None
            }
            
            if self.dream_memory:
                weights = [m["weight"] for m in self.dream_memory]
                timestamps = [m["timestamp"] for m in self.dream_memory]
                stats["average_weight"] = sum(weights) / len(weights)
                stats["oldest_timestamp"] = min(timestamps)
                stats["newest_timestamp"] = max(timestamps)
                
            return stats
            
        except Exception as e:
            self._log_error(f"Failed to get dream memory stats: {str(e)}")
            return {}
            
    @synchronized("lock")
    def validate_dream_memory(self) -> bool:
        """Validate dream memory tensors are on correct device."""
        try:
            for memory in self.dream_memory:
                if not isinstance(memory["tensor"], torch.Tensor):
                    return False
                if memory["tensor"].device != self.device:
                    memory["tensor"] = memory["tensor"].to(self.device)
            return True
        except Exception as e:
            self._log_error(f"Failed to validate dream memory: {str(e)}")
            return False

class StateManager:
    """Manages state initialization and persistence for the SOVL system."""
    def __init__(self, config_manager: ConfigManager, logger: Logger, device: torch.device):
        self.config_manager = config_manager
        self.logger = logger
        self.device = device
        self.state = None
        self.lock = Lock()

    def initialize_state(self) -> SOVLState:
        """Initialize a new SOVL state with configuration."""
        try:
            sovl_config = SOVLConfig(
                # Core configuration
                dream_memory_maxlen=self.config_manager.get("controls_config.dream_memory_maxlen", 10),
                temperament_history_maxlen=self.config_manager.get("controls_config.temperament_history_maxlen", 5),
                confidence_history_maxlen=self.config_manager.get("controls_config.confidence_history_maxlen", 5),
                hidden_size=self.config_manager.get("core_config.hidden_size", 768),
                max_seen_prompts=self.config_manager.get("controls_config.max_seen_prompts", 1000),
                quantization_mode=self.config_manager.get("core_config.quantization", "fp16"),
                sleep_max_steps=self.config_manager.get("training_config.sleep_max_steps", 100),
                prompt_timeout=self.config_manager.get("controls_config.prompt_timeout", 86400.0),
                temperament_decay_rate=self.config_manager.get("controls_config.temperament_decay_rate", 0.95),
                scaffold_unk_id=self.config_manager.get("controls_config.scaffold_unk_id", 0),
                lora_capacity=self.config_manager.get("training_config.lora_capacity", 0),
                
                # Temperament configuration
                temperament_melancholy_noise=self.config_manager.get("controls_config.temp_melancholy_noise", 0.1),
                temperament_influence=self.config_manager.get("controls_config.temperament_influence", 0.3),
                temperament_base_temperature=self.config_manager.get("controls_config.temperament_base_temperature", 0.7),
                temperament_swing_threshold=self.config_manager.get("controls_config.temperament_swing_threshold", 0.2),
                temperament_stability_threshold=self.config_manager.get("controls_config.temperament_stability_threshold", 0.1),
                
                # Training configuration
                learning_rate=self.config_manager.get("training_config.learning_rate", 2e-5),
                grad_accum_steps=self.config_manager.get("training_config.grad_accum_steps", 4),
                weight_decay=self.config_manager.get("training_config.weight_decay", 0.01),
                warmup_steps=self.config_manager.get("training_config.warmup_steps", 0),
                total_steps=self.config_manager.get("training_config.total_steps", 100000),
                max_grad_norm=self.config_manager.get("training_config.max_grad_norm", 1.0),
                use_amp=self.config_manager.get("training_config.use_amp", True),
                max_patience=self.config_manager.get("training_config.max_patience", 2),
                batch_size=self.config_manager.get("training_config.batch_size", 2),
                max_epochs=self.config_manager.get("training_config.max_epochs", 3),
                validate_every_n_steps=self.config_manager.get("training_config.validate_every_n_steps", 100),
                checkpoint_interval=self.config_manager.get("training_config.checkpoint_interval", 1000),
                checkpoint_path=self.config_manager.get("training_config.checkpoint_path", "checkpoints/sovl_trainer"),
                scheduler_type=self.config_manager.get("training_config.scheduler_type", "linear"),
                cosine_min_lr=self.config_manager.get("training_config.cosine_min_lr", 1e-6),
                warmup_ratio=self.config_manager.get("training_config.warmup_ratio", 0.1),
                dropout_rate=self.config_manager.get("training_config.dropout_rate", 0.1),
                max_seq_length=self.config_manager.get("training_config.max_seq_length", 512),
                metrics_to_track=self.config_manager.get("training_config.metrics_to_track", ["loss", "accuracy", "confidence"]),
                
                # Lifecycle configuration
                enable_gestation=self.config_manager.get("lifecycle_config.enable_gestation", True),
                enable_sleep_training=self.config_manager.get("lifecycle_config.enable_sleep_training", True),
                enable_lifecycle_weighting=self.config_manager.get("lifecycle_config.enable_lifecycle_weighting", True),
                lifecycle_capacity_factor=self.config_manager.get("lifecycle_config.lifecycle_capacity_factor", 0.01),
                lifecycle_curve=self.config_manager.get("lifecycle_config.lifecycle_curve", "sigmoid_linear"),
                sleep_conf_threshold=self.config_manager.get("lifecycle_config.sleep_conf_threshold", 0.7),
                sleep_log_min=self.config_manager.get("lifecycle_config.sleep_log_min", 10),
                exposure_gain_eager=self.config_manager.get("lifecycle_config.exposure_gain_eager", 3),
                exposure_gain_default=self.config_manager.get("lifecycle_config.exposure_gain_default", 2),
                
                # Dream configuration
                dream_memory_weight=self.config_manager.get("dream_config.dream_memory_weight", 0.1),
                enable_dreaming=self.config_manager.get("dream_config.enable_dreaming", True),
                repetition_n=self.config_manager.get("dream_config.repetition_n", 3),
                sigmoid_scale=self.config_manager.get("dream_config.sigmoid_scale", 0.5),
                sigmoid_shift=self.config_manager.get("dream_config.sigmoid_shift", 5.0),
                dream_noise_scale=self.config_manager.get("dream_config.dream_noise_scale", 0.05),
                dream_prompt_weight=self.config_manager.get("dream_config.dream_prompt_weight", 0.5),
                dream_novelty_boost=self.config_manager.get("dream_config.dream_novelty_boost", 0.03),
                dream_memory_decay=self.config_manager.get("dream_config.dream_memory_decay", 0.95),
                dream_prune_threshold=self.config_manager.get("dream_config.dream_prune_threshold", 0.1),
                temp_melancholy_noise=self.config_manager.get("dream_config.temp_melancholy_noise", 0.02),
                enable_prompt_driven_dreams=self.config_manager.get("dream_config.enable_prompt_driven_dreams", True),
                dream_swing_var=self.config_manager.get("dream_config.dream_swing_var", 0.1),
                dream_lifecycle_delta=self.config_manager.get("dream_config.dream_lifecycle_delta", 0.1),
                dream_temperament_on=self.config_manager.get("dream_config.dream_temperament_on", True),
                
                # Memory configuration
                memory_threshold=self.config_manager.get("memory_config.memory_threshold", 0.85),
                memory_decay_rate=self.config_manager.get("memory_config.memory_decay_rate", 0.95),
                use_scaffold_memory=self.config_manager.get("memory_config.use_scaffold_memory", True),
                use_token_map_memory=self.config_manager.get("memory_config.use_token_map_memory", True),
                scaffold_weight=self.config_manager.get("memory_config.scaffold_weight", 1.0),
                
                # Curiosity configuration
                weight_ignorance=self.config_manager.get("curiosity_config.weight_ignorance", 0.3),
                weight_novelty=self.config_manager.get("curiosity_config.weight_novelty", 0.7),
                metrics_maxlen=self.config_manager.get("curiosity_config.metrics_maxlen", 100),
                novelty_threshold_spontaneous=self.config_manager.get("curiosity_config.novelty_threshold_spontaneous", 0.7),
                novelty_threshold_curious=self.config_manager.get("curiosity_config.novelty_threshold_curious", 0.5),
                curiosity_decay_rate=self.config_manager.get("curiosity_config.curiosity_decay_rate", 0.95),
                curiosity_queue_maxlen=self.config_manager.get("curiosity_config.curiosity_queue_maxlen", 10),
                curiosity_question_timeout=self.config_manager.get("curiosity_config.curiosity_question_timeout", 3600.0)
            )
            
            state = SOVLState(self.config_manager, self.logger, self.device)
            state.config = sovl_config
            self._log_event(
                "state_initialized",
                message="SOVL state initialized",
                level="info",
                state_hash=state.state_hash(),
                conversation_id=state.history.conversation_id
            )
            return state
            
        except Exception as e:
            self._log_error(
                "state_initialization_failed",
                message=f"Failed to initialize state: {str(e)}",
                stack_trace=traceback.format_exc()
            )
            raise

    def load_state(self, state_path: Optional[str] = None) -> SOVLState:
        """Load state from file or initialize new state if not found."""
        try:
            with self.lock:
                if state_path is None:
                    state_path = self.config_manager.get("state_config.state_path", "sovl_state.json")
                if os.path.exists(state_path):
                    with open(state_path, 'r') as f:
                        state_data = json.load(f)
                    if self.state is None:
                        self.state = SOVLState(self.config_manager, self.logger, self.device)
                    self.state = SOVLState.from_dict(state_data, self.config_manager, self.logger, self.device)
                    self.logger.record({
                        "event": "state_loaded",
                        "message": "State loaded from file",
                        "level": "info",
                        "timestamp": time.time(),
                        "state_path": state_path,
                        "state_hash": self.state.state_hash()
                    })
                else:
                    self.state = self.initialize_state()
                    self.logger.record({
                        "event": "new_state_created",
                        "message": "New state created as file not found",
                        "level": "info",
                        "timestamp": time.time(),
                        "state_path": state_path,
                        "state_hash": self.state.state_hash()
                    })
                return self.state
        except Exception as e:
            self.logger.record({
                "error": f"Failed to load state: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise StateError(f"State loading failed: {str(e)}")

    def save_state(self, state_path: Optional[str] = None) -> None:
        """Save current state to file."""
        try:
            with self.lock:
                if self.state is None:
                    raise StateError("No state to save")
                if state_path is None:
                    state_path = self.config_manager.get("state_config.state_path", "sovl_state.json")
                state_data = self.state.to_dict()
                os.makedirs(os.path.dirname(state_path), exist_ok=True)
                with open(state_path, 'w') as f:
                    json.dump(state_data, f, indent=2)
                self.logger.record({
                    "event": "state_saved",
                    "message": "State saved to file",
                    "level": "info",
                    "timestamp": time.time(),
                    "state_path": state_path,
                    "state_hash": self.state.state_hash()
                })
        except Exception as e:
            self.logger.record({
                "error": f"Failed to save state: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise StateError(f"State saving failed: {str(e)}")

    def get_state(self) -> SOVLState:
        """Get the current state instance."""
        if self.state is None:
            raise StateError("State not initialized")
        return self.state
