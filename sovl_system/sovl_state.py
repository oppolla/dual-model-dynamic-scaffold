from typing import Optional, Deque, Dict, Set, Tuple, DefaultDict, Any, List
from collections import deque, defaultdict
from dataclasses import dataclass, field
import torch
import uuid
from threading import Lock, RLock
import time
import traceback
import hashlib
from sovl_logger import Logger
from sovl_config import ConfigManager
from sovl_utils import NumericalGuard, safe_divide, safe_compare
import json
import os
import threading

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

    def _log_event(self, event: str, **kwargs):
        """Log an event with standardized fields."""
        self.logger.record({
            "event": event,
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
        self._log_event("curiosity_state_initialized", config=self._config)

    def _load_curiosity_config(self) -> CuriosityConfig:
        """Load curiosity configuration."""
        return CuriosityConfig(
            max_questions=_load_config(self.config_manager, "controls_config", "curiosity_queue_maxlen", 10),
            max_novelty_scores=_load_config(self.config_manager, "controls_config", "novelty_history_maxlen", 20),
            decay_rate=_load_config(self.config_manager, "controls_config", "curiosity_decay_rate", 0.9),
            hidden_size=_load_config(self.config_manager, "core_config", "hidden_size", 768),
            question_timeout=_load_config(self.config_manager, "controls_config", "curiosity_question_timeout", 3600.0)
        )

    def update_question_history(self, question: str, timestamp: float) -> None:
        """Update question history and related state."""
        try:
            with self.lock:
                self.last_question_time = timestamp
                self.question_count += 1
                
                # Log question history update
                self._log_event("question_history_updated", {
                    "question": question,
                    "timestamp": timestamp,
                    "question_count": self.question_count,
                    "queue_length": len(self.unanswered_questions)
                })
                
                # Update pressure based on new question
                self._update_pressure()
                
        except Exception as e:
            self._log_error(f"Failed to update question history: {str(e)}")
            raise StateError(f"Question history update failed: {str(e)}")
            
    def add_question(self, question: str, score: float, context_vector: Optional[torch.Tensor] = None):
        """Add a new question with score and optional context vector."""
        try:
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Question must be a non-empty string")
            score = self._validate_number(score, "Score", min_value=0.0)
            if context_vector is not None:
                self._validate_tensor(context_vector, self._config.hidden_size, "Context vector")
            
            with self.lock:
                with NumericalGuard():
                    self.unanswered_questions.append((question, score, context_vector))
                    self.question_count += 1
                    self.last_question_time = time.time()
                    self._update_pressure()
                    self._log_event(
                        "question_added",
                        question=question,
                        score=score,
                        has_context_vector=context_vector is not None,
                        question_count=self.question_count,
                        queue_length=len(self.unanswered_questions)
                    )
        except Exception as e:
            self._log_error(f"Failed to add question: {str(e)}", question=question, score=score)
            raise StateError(f"Add question failed: {str(e)}")

    def prioritize_questions(self):
        """Sort unanswered questions by score."""
        try:
            with self.lock:
                sorted_questions = sorted(self.unanswered_questions, key=lambda x: x[1], reverse=True)
                self.unanswered_questions = deque(sorted_questions, maxlen=self._config.max_questions)
                self._log_event(
                    "questions_prioritized",
                    question_count=len(self.unanswered_questions)
                )
        except Exception as e:
            self._log_error("Question prioritization failed")
            raise StateError(f"Question prioritization failed: {str(e)}")

    def prune_old_questions(self, timeout: float) -> None:
        """Remove questions older than timeout."""
        try:
            current_time = time.time()
            with self.lock:
                while self.unanswered_questions and current_time - self.last_question_time > timeout:
                    question, _, _ = self.unanswered_questions.popleft()
                    self._log_event("old_question_pruned", question=question)
                self._update_pressure()
        except Exception as e:
            self._log_error("Question pruning failed")
            raise StateError(f"Question pruning failed: {str(e)}")

    def _update_pressure(self):
        """Update curiosity pressure based on questions and novelty."""
        try:
            with self.lock:
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
                        pressure=self.pressure,
                        unanswered_count=len(self.unanswered_questions),
                        novelty_avg=novelty_avg
                    )
        except Exception as e:
            self._log_error("Pressure update failed")
            raise StateError(f"Pressure update failed: {str(e)}")

    def add_novelty_score(self, score: float):
        """Add a novelty score and decay existing scores."""
        try:
            score = self._validate_number(score, "Novelty score", min_value=0.0)
            with self.lock:
                with NumericalGuard():
                    self.novelty_scores.append(score * self._config.decay_rate)
                    self._update_pressure()
                    self._log_event(
                        "novelty_score_added",
                        score=score,
                        novelty_scores_count=len(self.novelty_scores)
                    )
        except Exception as e:
            self._log_error(f"Failed to add novelty score: {str(e)}", score=score)
            raise StateError(f"Add novelty score failed: {str(e)}")

    def get_context_vector(self) -> Optional[torch.Tensor]:
        """Compute a weighted average of context vectors from questions."""
        try:
            with self.lock:
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
                    self._log_event("unsupported_version", version=version)
                self.unanswered_questions = deque(maxlen=self._config.max_questions)
                for q, s, v in data.get("unanswered_questions", []):
                    context_vector = (
                        torch.tensor(v, dtype=torch.float32)
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
                    pressure=self.pressure,
                    decay_rate=self._config.decay_rate,
                    question_timeout=self._config.question_timeout
                )
        except Exception as e:
            self._log_error("Failed to tune curiosity")
            raise StateError(f"Tune curiosity failed: {str(e)}")

class ConversationHistory:
    """Manages conversation messages with unique ID."""
    def __init__(self, config_manager: ConfigManager, conversation_id: Optional[str] = None):
        self._config = ConversationConfig(
            max_messages=_load_config(config_manager, "controls_config", "conversation_history_maxlen", 10)
        )
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

class SOVLState(StateBase):
    """Manages the overall state of the SOVL system."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger, device: torch.device):
        """Initialize SOVL state with configuration and device."""
        super().__init__(config_manager, logger)
        self.device = device
        self.lock = threading.Lock()
        
        # Initialize state components
        self._initialize_state()
        self._log_event("state_initialized", {
            "state_hash": self.state_hash,
            "conversation_id": self.history.conversation_id
        })
        
    def _initialize_state(self) -> None:
        """Initialize all state components with proper validation."""
        with self.lock:
            # Initialize dream memory
            self.dream_memory = deque(maxlen=self.config_manager.get("controls_config.dream_memory_maxlen", 100))
            self.dream_memory_maxlen = self.config_manager.get("controls_config.dream_memory_maxlen", 100)
            
            # Initialize confidence tracking
            self.confidence_history = deque(
                maxlen=self.config_manager.get("controls_config.confidence_history_maxlen", 1000)
            )
            self.sleep_confidence_sum = 0.0
            self.sleep_confidence_count = 0
            
            # Initialize gestation state
            self.gestation_state = "normal"
            
            # Initialize token map
            self.token_map = {}
            
            # Initialize other state components
            self.history = ConversationHistory(
                maxlen=self.config_manager.get("controls_config.conversation_history_maxlen", 10)
            )
            self.curiosity = CuriosityState(
                config_manager=self.config_manager,
                logger=self.logger,
                device=self.device
            )
            
    def update_token_map(self, token_map: Dict[int, Dict[str, Any]]) -> None:
        """Update the token map with validation."""
        with self.lock:
            try:
                # Validate token map structure
                for base_id, mapping in token_map.items():
                    if not isinstance(base_id, int):
                        raise ValueError(f"Invalid base_id type: {type(base_id)}")
                    if not isinstance(mapping, dict):
                        raise ValueError(f"Invalid mapping type for base_id {base_id}")
                    if 'ids' not in mapping or 'weight' not in mapping:
                        raise ValueError(f"Missing required fields in mapping for base_id {base_id}")
                    if not isinstance(mapping['ids'], list):
                        raise ValueError(f"Invalid ids type for base_id {base_id}")
                    if not isinstance(mapping['weight'], (int, float)):
                        raise ValueError(f"Invalid weight type for base_id {base_id}")

                self.token_map = token_map
                self._log_event("token_map_updated", {
                    "token_map_size": len(token_map),
                    "state_hash": self.state_hash()
                })
            except Exception as e:
                self._log_error(f"Failed to update token map: {str(e)}")
                raise

    def get_token_map(self) -> Dict[int, Dict[str, Any]]:
        """Get a copy of the current token map."""
        with self.lock:
            return dict(self.token_map)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        with self.lock:
            return {
                "dream_memory": [(tensor.cpu().tolist(), weight) for tensor, weight in self.dream_memory],
                "dream_memory_maxlen": self.dream_memory_maxlen,
                "confidence_history": list(self.confidence_history),
                "sleep_confidence_sum": self.sleep_confidence_sum,
                "sleep_confidence_count": self.sleep_confidence_count,
                "gestation_state": self.gestation_state,
                "history": self.history.to_dict(),
                "curiosity": self.curiosity.to_dict(),
                "state_hash": self.state_hash
            }
            
    @classmethod
    def from_dict(cls, data: Dict[str, Any], device: torch.device) -> 'SOVLState':
        """Create state instance from dictionary."""
        state = cls(data["config_manager"], data["logger"], device)
        with state.lock:
            # Load dream memory
            if "dream_memory" in data:
                state.dream_memory = deque(
                    [(torch.tensor(t, device=device, dtype=torch.float), w) for t, w in data["dream_memory"]],
                    maxlen=data.get("dream_memory_maxlen", 100)
                )
            
            # Load confidence tracking
            state.confidence_history = deque(
                data.get("confidence_history", []),
                maxlen=data.get("confidence_history_maxlen", 1000)
            )
            state.sleep_confidence_sum = data.get("sleep_confidence_sum", 0.0)
            state.sleep_confidence_count = data.get("sleep_confidence_count", 0)
            
            # Load gestation state
            state.gestation_state = data.get("gestation_state", "normal")
            
            # Load other components
            if "history" in data:
                state.history = ConversationHistory.from_dict(data["history"])
            if "curiosity" in data:
                state.curiosity = CuriosityState.from_dict(data["curiosity"], device)
                
            # Validate state
            state._validate_state()
            
        return state
        
    def _validate_state(self) -> None:
        """Validate state components and initialize missing fields."""
        with self.lock:
            # Validate dream memory
            if not hasattr(self, 'dream_memory'):
                self.dream_memory = deque(maxlen=self.dream_memory_maxlen)
                self._log_event("state_validation", {"field": "dream_memory", "action": "initialized"})
                
            # Validate confidence tracking
            if not hasattr(self, 'confidence_history'):
                self.confidence_history = deque(maxlen=self.config_manager.get("controls_config.confidence_history_maxlen", 1000))
                self._log_event("state_validation", {"field": "confidence_history", "action": "initialized"})
                
            if not hasattr(self, 'sleep_confidence_sum'):
                self.sleep_confidence_sum = 0.0
                self._log_event("state_validation", {"field": "sleep_confidence_sum", "action": "initialized"})
                
            if not hasattr(self, 'sleep_confidence_count'):
                self.sleep_confidence_count = 0
                self._log_event("state_validation", {"field": "sleep_confidence_count", "action": "initialized"})
                
            # Validate gestation state
            if not hasattr(self, 'gestation_state'):
                self.gestation_state = "normal"
                self._log_event("state_validation", {"field": "gestation_state", "action": "initialized"})
                
            # Validate other components
            if not hasattr(self, 'history'):
                self.history = ConversationHistory(
                    maxlen=self.config_manager.get("controls_config.conversation_history_maxlen", 10)
                )
                self._log_event("state_validation", {"field": "history", "action": "initialized"})
                
            # Enhanced CuriosityState validation
            if not hasattr(self, 'curiosity'):
                self.curiosity = CuriosityState(
                    config_manager=self.config_manager,
                    logger=self.logger,
                    device=self.device
                )
                self._log_event("state_validation", {"field": "curiosity", "action": "initialized"})
            else:
                # Validate CuriosityState attributes
                required_curiosity_attrs = [
                    'pressure', 'novelty_threshold_spontaneous',
                    'novelty_threshold_response', 'pressure_threshold',
                    'pressure_drop', 'silence_threshold',
                    'question_cooldown', 'queue_maxlen'
                ]
                
                missing_attrs = [attr for attr in required_curiosity_attrs 
                               if not hasattr(self.curiosity, attr)]
                
                if missing_attrs:
                    self._log_event("state_validation", {
                        "field": "curiosity",
                        "action": "reinitialized",
                        "missing_attrs": missing_attrs
                    })
                    self.curiosity = CuriosityState(
                        config_manager=self.config_manager,
                        logger=self.logger,
                        device=self.device
                    )
                
                # Validate curiosity parameter ranges
                if not (0 <= self.curiosity.pressure <= 1):
                    self._log_event("state_validation", {
                        "field": "curiosity.pressure",
                        "action": "clamped",
                        "old_value": self.curiosity.pressure,
                        "new_value": max(0, min(1, self.curiosity.pressure))
                    })
                    self.curiosity.pressure = max(0, min(1, self.curiosity.pressure))
                
                if not (0 <= self.curiosity.novelty_threshold_spontaneous <= 1):
                    self._log_event("state_validation", {
                        "field": "curiosity.novelty_threshold_spontaneous",
                        "action": "clamped",
                        "old_value": self.curiosity.novelty_threshold_spontaneous,
                        "new_value": max(0, min(1, self.curiosity.novelty_threshold_spontaneous))
                    })
                    self.curiosity.novelty_threshold_spontaneous = max(0, min(1, self.curiosity.novelty_threshold_spontaneous))
                
                if not (0 <= self.curiosity.novelty_threshold_response <= 1):
                    self._log_event("state_validation", {
                        "field": "curiosity.novelty_threshold_response",
                        "action": "clamped",
                        "old_value": self.curiosity.novelty_threshold_response,
                        "new_value": max(0, min(1, self.curiosity.novelty_threshold_response))
                    })
                    self.curiosity.novelty_threshold_response = max(0, min(1, self.curiosity.novelty_threshold_response))

class StateManager:
    """Manages state initialization and persistence for the SOVL system."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger, device: torch.device):
        self.config_manager = config_manager
        self.logger = logger
        self.device = device
        self.state = None
        self.lock = Lock()
        
    def initialize_state(self) -> SOVLState:
        """Initialize a new SOVLState instance with default values."""
        try:
            with self.lock:
                # Load configuration values
                dream_memory_maxlen = self.config_manager.get("controls_config.dream_memory_maxlen", 10)
                temperament_history_maxlen = self.config_manager.get("controls_config.temperament_history_maxlen", 5)
                confidence_history_maxlen = self.config_manager.get("controls_config.confidence_history_maxlen", 5)
                hidden_size = self.config_manager.get("core_config.hidden_size", 768)
                max_seen_prompts = self.config_manager.get("controls_config.max_seen_prompts", 1000)
                quantization_mode = self.config_manager.get("core_config.quantization", "fp16")
                sleep_max_steps = self.config_manager.get("training_config.sleep_max_steps", 100)
                prompt_timeout = self.config_manager.get("controls_config.prompt_timeout", 86400.0)
                temperament_decay_rate = self.config_manager.get("controls_config.temperament_decay_rate", 0.95)
                scaffold_unk_id = self.config_manager.get("controls_config.scaffold_unk_id", 0)
                lora_capacity = self.config_manager.get("training_config.lora_capacity", 0)
                
                # Create SOVLConfig
                sovl_config = SOVLConfig(
                    dream_memory_maxlen=dream_memory_maxlen,
                    temperament_history_maxlen=temperament_history_maxlen,
                    confidence_history_maxlen=confidence_history_maxlen,
                    hidden_size=hidden_size,
                    max_seen_prompts=max_seen_prompts,
                    quantization_mode=quantization_mode,
                    sleep_max_steps=sleep_max_steps,
                    prompt_timeout=prompt_timeout,
                    temperament_decay_rate=temperament_decay_rate,
                    scaffold_unk_id=scaffold_unk_id,
                    lora_capacity=lora_capacity
                )
                
                # Initialize state
                self.state = SOVLState(
                    config_manager=self.config_manager,
                    logger=self.logger,
                    device=self.device
                )
                
                self.logger.record({
                    "event": "state_initialized",
                    "timestamp": time.time(),
                    "state_hash": self.state.get_state_hash()
                })
                
                return self.state
                
        except Exception as e:
            self.logger.record({
                "error": f"Failed to initialize state: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise StateError(f"State initialization failed: {str(e)}")
            
    def load_state(self, state_path: Optional[str] = None) -> SOVLState:
        """Load state from file or initialize new state if not found."""
        try:
            with self.lock:
                if state_path is None:
                    state_path = self.config_manager.get("state_config.state_path", "sovl_state.json")
                
                if os.path.exists(state_path):
                    with open(state_path, 'r') as f:
                        state_data = json.load(f)
                    
                    # Initialize state if not already done
                    if self.state is None:
                        self.initialize_state()
                    
                    # Load state data
                    self.state.from_dict(state_data, self.device)
                    
                    self.logger.record({
                        "event": "state_loaded",
                        "timestamp": time.time(),
                        "state_path": state_path,
                        "state_hash": self.state.get_state_hash()
                    })
                else:
                    # Initialize new state if file doesn't exist
                    self.state = self.initialize_state()
                    
                    self.logger.record({
                        "event": "new_state_created",
                        "timestamp": time.time(),
                        "state_path": state_path,
                        "state_hash": self.state.get_state_hash()
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
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(state_path), exist_ok=True)
                
                with open(state_path, 'w') as f:
                    json.dump(state_data, f, indent=2)
                
                self.logger.record({
                    "event": "state_saved",
                    "timestamp": time.time(),
                    "state_path": state_path,
                    "state_hash": self.state.get_state_hash()
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
