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

    def update_question_history(self, question: str, timestamp: float) -> None:
        """Update question history and related state."""
        try:
            with self.lock:
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

    def prioritize_questions(self):
        """Sort unanswered questions by score."""
        try:
            with self.lock:
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

    def prune_old_questions(self, timeout: float) -> None:
        """Remove questions older than timeout."""
        try:
            current_time = time.time()
            with self.lock:
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
                        message="Curiosity pressure updated",
                        level="info",
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
                        message="Novelty score added",
                        level="info",
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
    """Manages the overall state of the SOVL system."""
    def __init__(self, config_manager: ConfigManager, logger: Logger, device: torch.device):
        """Initialize SOVL state with configuration and device."""
        super().__init__(config_manager, logger)
        self.device = device
        self.lock = threading.Lock()
        self._initialize_state()
        self._log_event(
            "state_initialized",
            message="SOVL state initialized",
            level="info",
            state_hash=self.state_hash(),
            conversation_id=self.history.conversation_id
        )

    def _initialize_state(self) -> None:
        """Initialize all state components with proper validation."""
        with self.lock:
            self.dream_memory_maxlen = self.config_manager.get("controls_config.dream_memory_maxlen", 100)
            self.dream_memory = deque(maxlen=self.dream_memory_maxlen)
            self.confidence_history = deque(
                maxlen=self.config_manager.get("controls_config.confidence_history_maxlen", 1000)
            )
            self.sleep_confidence_sum = 0.0
            self.sleep_confidence_count = 0
            self.gestation_state = "normal"
            self.token_map = {}
            self.history = ConversationHistory(
                maxlen=self.config_manager.get("controls_config.conversation_history_maxlen", 10)
            )
            self.curiosity = CuriosityState(
                config_manager=self.config_manager,
                logger=self.logger,
                device=self.device
            )
            self.last_prompt_embedding = None

    def state_hash(self) -> str:
        """Generate a hash of the current state."""
        with self.lock:
            state_dict = {
                "token_map_size": len(self.token_map),
                "dream_memory_len": len(self.dream_memory),
                "confidence_history_len": len(self.confidence_history),
                "conversation_id": self.history.conversation_id,
                "curiosity_pressure": self.curiosity.pressure
            }
            return hashlib.md5(json.dumps(state_dict, sort_keys=True).encode()).hexdigest()

    def update_token_map(self, token_map: Dict[int, Dict[str, Any]]) -> None:
        """Update the token map with validation."""
        with self.lock:
            try:
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
                self._log_event(
                    "token_map_updated",
                    message="Token map updated",
                    level="info",
                    token_map_size=len(token_map),
                    state_hash=self.state_hash()
                )
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
            dream_memory_serialized = [
                {
                    "tensor": entry["tensor"].cpu().numpy().tolist(),
                    "weight": entry["weight"],
                    "metadata": entry["metadata"]
                }
                for entry in self.dream_memory
            ]
            return {
                "dream_memory": dream_memory_serialized,
                "dream_memory_maxlen": self.dream_memory_maxlen,
                "confidence_history": list(self.confidence_history),
                "sleep_confidence_sum": self.sleep_confidence_sum,
                "sleep_confidence_count": self.sleep_confidence_count,
                "gestation_state": self.gestation_state,
                "token_map": self.token_map,
                "history": self.history.to_dict(),
                "curiosity": self.curiosity.to_dict(),
                "last_prompt_embedding": (
                    self.last_prompt_embedding.cpu().numpy().tolist()
                    if self.last_prompt_embedding is not None else None
                ),
                "state_hash": self.state_hash()
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], config_manager: ConfigManager, logger: Logger, device: torch.device) -> 'SOVLState':
        """Create state instance from dictionary."""
        state = cls(config_manager, logger, device)
        with state.lock:
            if "dream_memory" in data:
                state.dream_memory = deque(maxlen=data.get("dream_memory_maxlen", 100))
                for entry in data["dream_memory"]:
                    state.dream_memory.append({
                        "tensor": torch.tensor(entry["tensor"], device=device, dtype=torch.float),
                        "weight": entry["weight"],
                        "metadata": entry.get("metadata", {"timestamp": time.time()})
                    })
            state.confidence_history = deque(
                data.get("confidence_history", []),
                maxlen=data.get("confidence_history_maxlen", 1000)
            )
            state.sleep_confidence_sum = data.get("sleep_confidence_sum", 0.0)
            state.sleep_confidence_count = data.get("sleep_confidence_count", 0)
            state.gestation_state = data.get("gestation_state", "normal")
            state.token_map = data.get("token_map", {})
            if "history" in data:
                state.history = ConversationHistory.from_dict(
                    data["history"],
                    maxlen=config_manager.get("controls_config.conversation_history_maxlen", 10)
                )
            if "curiosity" in data:
                state.curiosity.from_dict(data["curiosity"])
            if "last_prompt_embedding" in data and data["last_prompt_embedding"] is not None:
                state.last_prompt_embedding = torch.tensor(data["last_prompt_embedding"], device=device, dtype=torch.float)
            state._validate_state()
        return state

    def _validate_state(self) -> None:
        """Validate state components and initialize missing fields."""
        with self.lock:
            if not hasattr(self, 'dream_memory'):
                self.dream_memory = deque(maxlen=self.dream_memory_maxlen)
                self._log_event(
                    "state_validation",
                    message="Dream memory initialized",
                    level="info",
                    field="dream_memory",
                    action="initialized"
                )
            if not hasattr(self, 'confidence_history'):
                self.confidence_history = deque(maxlen=self.config_manager.get("controls_config.confidence_history_maxlen", 1000))
                self._log_event(
                    "state_validation",
                    message="Confidence history initialized",
                    level="info",
                    field="confidence_history",
                    action="initialized"
                )
            if not hasattr(self, 'sleep_confidence_sum'):
                self.sleep_confidence_sum = 0.0
                self._log_event(
                    "state_validation",
                    message="Sleep confidence sum initialized",
                    level="info",
                    field="sleep_confidence_sum",
                    action="initialized"
                )
            if not hasattr(self, 'sleep_confidence_count'):
                self.sleep_confidence_count = 0
                self._log_event(
                    "state_validation",
                    message="Sleep confidence count initialized",
                    level="info",
                    field="sleep_confidence_count",
                    action="initialized"
                )
            if not hasattr(self, 'gestation_state'):
                self.gestation_state = "normal"
                self._log_event(
                    "state_validation",
                    message="Gestation state initialized",
                    level="info",
                    field="gestation_state",
                    action="initialized"
                )
            if not hasattr(self, 'token_map'):
                self.token_map = {}
                self._log_event(
                    "state_validation",
                    message="Token map initialized",
                    level="info",
                    field="token_map",
                    action="initialized"
                )
            if not hasattr(self, 'history'):
                self.history = ConversationHistory(
                    maxlen=self.config_manager.get("controls_config.conversation_history_maxlen", 10)
                )
                self._log_event(
                    "state_validation",
                    message="Conversation history initialized",
                    level="info",
                    field="history",
                    action="initialized"
                )
            if not hasattr(self, 'curiosity'):
                self.curiosity = CuriosityState(
                    config_manager=self.config_manager,
                    logger=self.logger,
                    device=self.device
                )
                self._log_event(
                    "state_validation",
                    message="Curiosity state initialized",
                    level="info",
                    field="curiosity",
                    action="initialized"
                )
            else:
                required_curiosity_attrs = [
                    'unanswered_questions', 'last_question_time', 'pressure',
                    'novelty_scores', 'question_count'
                ]
                missing_attrs = [attr for attr in required_curiosity_attrs if not hasattr(self.curiosity, attr)]
                if missing_attrs:
                    self._log_event(
                        "state_validation",
                        message="Curiosity state reinitialized due to missing attributes",
                        level="warning",
                        field="curiosity",
                        action="reinitialized",
                        missing_attrs=missing_attrs
                    )
                    self.curiosity = CuriosityState(
                        config_manager=self.config_manager,
                        logger=self.logger,
                        device=self.device
                    )
                if not (0 <= self.curiosity.pressure <= 1):
                    old_pressure = self.curiosity.pressure
                    self.curiosity.pressure = max(0, min(1, self.curiosity.pressure))
                    self._log_event(
                        "state_validation",
                        message="Curiosity pressure clamped",
                        level="warning",
                        field="curiosity.pressure",
                        action="clamped",
                        old_value=old_pressure,
                        new_value=self.curiosity.pressure
                    )

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
                self.state = SOVLState(
                    config_manager=self.config_manager,
                    logger=self.logger,
                    device=self.device
                )
                self.logger.record({
                    "event": "state_initialized",
                    "message": "State manager initialized new state",
                    "level": "info",
                    "timestamp": time.time(),
                    "state_hash": self.state.state_hash()
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
