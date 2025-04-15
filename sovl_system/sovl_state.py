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
        super().__init__(config_manager, logger)
        self._config = self._load_sovl_config()
        self.device = device
        self.state_version = "1.1"
        self.state_hash: Optional[str] = None

        # Conversation and memory
        self.history = ConversationHistory(config_manager)
        self.dream_memory: Deque[Dict[str, Any]] = deque(maxlen=self._config.dream_memory_maxlen)
        self.seen_prompts: Deque[Tuple[str, float]] = deque(maxlen=self._config.max_seen_prompts)
        self.token_map: DefaultDict[int, List[int]] = defaultdict(lambda: [self._config.scaffold_unk_id])
        self.last_prompt_embedding: Optional[torch.Tensor] = None

        # Training state
        self.data_exposure: int = 0
        self.last_trained: float = 0.0
        self.global_step: int = 0
        self.best_valid_loss: float = float('inf')
        self.patience: int = 0
        self.lora_capacity: int = self._config.lora_capacity

        # Behavioral state
        self.temperament_score: float = 0.0
        self.last_temperament_score: float = 0.0
        self.temperament_history: Deque[float] = deque(maxlen=self._config.temperament_history_maxlen)
        self.confidence_history: Deque[float] = deque(maxlen=self._config.confidence_history_maxlen)
        self.sleep_confidence_sum: float = 0.0
        self.sleep_confidence_count: int = 0

        # Dynamic controls
        self.last_weight: float = 0.0
        self.is_sleeping: bool = False
        self.sleep_progress: int = 0
        self.sleep_batch: List = []
        self.sleep_total_loss: float = 0.0
        self.sleep_steps: int = 0

        # Curiosity state
        self.curiosity = CuriosityState(config_manager, logger, device)

        self._update_state_hash()
        self._log_event(
            "state_initialized",
            hidden_size=self._config.hidden_size,
            dream_memory_maxlen=self._config.dream_memory_maxlen,
            state_hash=self.state_hash,
            device=str(device),
            conversation_id=self.history.conversation_id
        )

    def _load_sovl_config(self) -> SOVLConfig:
        """Load SOVL configuration."""
        return SOVLConfig(
            dream_memory_maxlen=_load_config(self.config_manager, "controls_config", "dream_memory_maxlen", 10),
            temperament_history_maxlen=_load_config(self.config_manager, "controls_config", "temperament_history_maxlen", 5),
            confidence_history_maxlen=_load_config(self.config_manager, "controls_config", "confidence_history_maxlen", 5),
            hidden_size=_load_config(self.config_manager, "core_config", "hidden_size", 768),
            max_seen_prompts=_load_config(self.config_manager, "controls_config", "max_seen_prompts", 1000),
            quantization_mode=_load_config(self.config_manager, "core_config", "quantization", "fp16"),
            sleep_max_steps=_load_config(self.config_manager, "training_config", "sleep_max_steps", 100),
            prompt_timeout=_load_config(self.config_manager, "controls_config", "prompt_timeout", 86400.0),
            temperament_decay_rate=_load_config(self.config_manager, "controls_config", "temperament_decay_rate", 0.95),
            scaffold_unk_id=_load_config(self.config_manager, "controls_config", "scaffold_unk_id", 0),
            lora_capacity=_load_config(self.config_manager, "training_config", "lora_capacity", 0)
        )

    def _update_state_hash(self):
        """Compute a hash of critical state components."""
        try:
            state_str = (
                f"{self.temperament_score}:{self.curiosity.question_count}:"
                f"{len(self.dream_memory)}:{len(self.seen_prompts)}"
            )
            self.state_hash = hashlib.sha256(state_str.encode()).hexdigest()[:16]
        except Exception as e:
            self._log_error("State hash update failed")

    def set_scaffold_unk_id(self, unk_id: int):
        """Set the unknown token ID for the scaffold model."""
        try:
            if not isinstance(unk_id, int) or unk_id < 0:
                raise ValueError("unk_id must be a non-negative integer")
            with self.lock:
                self.token_map = defaultdict(lambda: [unk_id], {k: v for k, v in self.token_map.items()})
                self._update_state_hash()
                self._log_event(
                    "scaffold_unk_id_set",
                    unk_id=unk_id,
                    token_map_size=len(self.token_map),
                    state_hash=self.state_hash
                )
        except Exception as e:
            self._log_error("Failed to set scaffold unk_id", unk_id=unk_id)
            raise StateError(f"Set scaffold unk_id failed: {str(e)}")

    def append_dream_memory(self, tensor: torch.Tensor, weight: float, source: str = "unknown"):
        """Append a tensor to dream_memory with metadata."""
        try:
            self._validate_tensor(tensor, self._config.hidden_size, "Tensor")
            weight = self._validate_number(weight, "Weight", min_value=0.0)
            
            # Ensure tensor is on the correct device
            tensor = tensor.to(self.device)
            
            with self.lock:
                with NumericalGuard(dtype=torch.float16 if self._config.quantization_mode == "fp16" else torch.float32):
                    memory_entry = {
                        "tensor": tensor.to(dtype=torch.float16 if self._config.quantization_mode == "fp16" else torch.float32),
                        "weight": weight,
                        "source": source,
                        "timestamp": time.time(),
                        "device": str(tensor.device)
                    }
                    self.dream_memory.append(memory_entry)
                    self._update_state_hash()
                    self._log_event(
                        "dream_memory_appended",
                        tensor_shape=list(tensor.shape),
                        weight=weight,
                        source=source,
                        memory_length=len(self.dream_memory),
                        state_hash=self.state_hash,
                        device=str(tensor.device)
                    )
        except Exception as e:
            self._log_error(
                "Failed to append dream memory",
                tensor_shape=list(tensor.shape) if isinstance(tensor, torch.Tensor) else None,
                weight=weight,
                source=source,
                device=str(tensor.device) if isinstance(tensor, torch.Tensor) else None
            )
            raise StateError(f"Append dream memory failed: {str(e)}")

    def get_weighted_memory(self) -> Optional[torch.Tensor]:
        """Compute a weighted average of dream memory tensors."""
        try:
            with self.lock:
                if not self.dream_memory:
                    return None
                    
                # Ensure all tensors are on the correct device
                tensors = [entry["tensor"].to(self.device) for entry in self.dream_memory]
                weights = torch.tensor([entry["weight"] for entry in self.dream_memory], device=self.device)
                
                with NumericalGuard():
                    weights = weights / (weights.sum() + 1e-8)
                    stacked = torch.stack(tensors)
                    weighted_avg = (stacked * weights.view(-1, *([1] * (stacked.dim() - 1)))).sum(dim=0)
                    
                    self._log_event(
                        "weighted_memory_computed",
                        tensor_shape=list(weighted_avg.shape),
                        memory_length=len(self.dream_memory),
                        device=str(weighted_avg.device)
                    )
                    return weighted_avg
        except Exception as e:
            self._log_error(
                "Weighted memory computation failed",
                memory_length=len(self.dream_memory),
                device=str(self.device)
            )
            return None

    def reset_conversation(self):
        """Start fresh conversation while retaining memory."""
        try:
            with self.lock:
                old_id = self.history.conversation_id
                self.history = ConversationHistory(self.config_manager)
                self._update_state_hash()
                self._log_event(
                    "conversation_reset",
                    old_conversation_id=old_id,
                    new_conversation_id=self.history.conversation_id,
                    state_hash=self.state_hash,
                    dream_memory_length=len(self.dream_memory),
                    seen_prompts_count=len(self.seen_prompts)
                )
        except Exception as e:
            self._log_error("Conversation reset failed")
            raise StateError(f"Conversation reset failed: {str(e)}")

    def add_seen_prompt(self, prompt: str):
        """Add a seen prompt with timestamp."""
        try:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Prompt must be a non-empty string")
            with self.lock:
                current_time = time.time()
                self.seen_prompts.append((prompt, current_time))
                self._prune_seen_prompts(current_time)
                self._update_state_hash()
                self._log_event(
                    "seen_prompt_added",
                    prompt_length=len(prompt),
                    seen_prompts_count=len(self.seen_prompts),
                    state_hash=self.state_hash
                )
        except Exception as e:
            self._log_error("Failed to add seen prompt", prompt=prompt)
            raise StateError(f"Add seen prompt failed: {str(e)}")

    def _prune_seen_prompts(self, current_time: float):
        """Remove old prompts based on timeout."""
        while self.seen_prompts and current_time - self.seen_prompts[0][1] > self._config.prompt_timeout:
            prompt, _ = self.seen_prompts.popleft()
            self._log_event("seen_prompt_pruned", prompt=prompt)

    def update_temperament(self, new_temperament: Dict[str, float]) -> None:
        """
        Update temperament state with proper synchronization.
        
        Args:
            new_temperament: New temperament values
        """
        with self.lock:
            # Validate new temperament
            if not all(0.0 <= v <= 1.0 for v in new_temperament.values()):
                raise StateError("Invalid temperament values: must be between 0.0 and 1.0")
                
            # Update state
            self.temperament_score = new_temperament.copy()
            self.temperament_history.append(self.temperament_score.copy())
            self._update_temperament_score()
            self.state_hash = self._compute_state_hash()
            
            # Log update
            self.logger.record_event(
                event_type="state_temperament_updated",
                message="Temperament state updated",
                level="info",
                additional_info={
                    "new_temperament": new_temperament,
                    "temperament_score": self.temperament_score,
                    "conversation_id": self.history.conversation_id,
                    "state_hash": self.state_hash
                }
            )
            
    def _update_temperament_score(self) -> None:
        """Update the temperament score based on current state."""
        with self.lock:
            # Calculate weighted average of temperament traits
            weights = {
                "curiosity": 0.4,
                "confidence": 0.3,
                "stability": 0.2,
                "adaptability": 0.1
            }
            
            score = sum(
                self.temperament_score[trait] * weight
                for trait, weight in weights.items()
            )
            
            # Normalize to [-1.0, 1.0] range
            self.temperament_score = (score * 2.0) - 1.0
            
    @property
    def temperament_score(self) -> float:
        """Get the current temperament score."""
        with self.lock:
            return self.temperament_score
            
    def _compute_state_hash(self) -> str:
        """Compute a hash of the current state."""
        with self.lock:
            state_data = {
                "temperament": self.temperament_score,
                "conversation_id": self.history.conversation_id
            }
            return hashlib.sha256(
                json.dumps(state_data, sort_keys=True).encode()
            ).hexdigest()

    def update_confidence(self, confidence: float):
        """Update confidence history."""
        try:
            confidence = self._validate_number(confidence, "Confidence", min_value=0.0)
            with self.lock:
                with NumericalGuard():
                    self.confidence_history.append(confidence)
                    if self.is_sleeping:
                        self.sleep_confidence_sum += confidence
                        self.sleep_confidence_count += 1
                    self._update_state_hash()
                    self._log_event(
                        "confidence_updated",
                        confidence=confidence,
                        history_length=len(self.confidence_history),
                        is_sleeping=self.is_sleeping,
                        state_hash=self.state_hash
                    )
        except Exception as e:
            self._log_error("Confidence update failed", confidence=confidence)
            raise StateError(f"Confidence update failed: {str(e)}")

    def set_last_prompt_embedding(self, embedding: torch.Tensor):
        """Set the last prompt embedding with validation."""
        try:
            self._validate_tensor(embedding, self._config.hidden_size, "Embedding")
            
            # Ensure embedding is on the correct device
            embedding = embedding.to(self.device)
            
            with self.lock:
                with NumericalGuard():
                    self.last_prompt_embedding = embedding.to(
                        dtype=torch.float16 if self._config.quantization_mode == "fp16" else torch.float32
                    )
                    self._update_state_hash()
                    self._log_event(
                        "last_prompt_embedding_set",
                        embedding_shape=list(embedding.shape),
                        state_hash=self.state_hash,
                        device=str(embedding.device)
                    )
        except Exception as e:
            self._log_error(
                "Failed to set last prompt embedding",
                embedding_shape=list(embedding.shape) if isinstance(embedding, torch.Tensor) else None,
                device=str(embedding.device) if isinstance(embedding, torch.Tensor) else None
            )
            raise StateError(f"Set last prompt embedding failed: {str(e)}")

    def to_dict(self, max_retries: int = 3) -> Dict[str, Any]:
        """Serialize state to dictionary with retry logic."""
        for attempt in range(max_retries):
            try:
                with self.lock:
                    state_dict = {
                        "version": self.state_version,
                        "state_hash": self.state_hash,
                        "history": {
                            "conversation_id": self.history.conversation_id,
                            "messages": list(self.history.messages)
                        },
                        "dream_memory": [
                            {
                                "tensor": entry["tensor"].cpu().numpy().tolist(),
                                "weight": entry["weight"],
                                "source": entry["source"],
                                "timestamp": entry["timestamp"],
                                "device": entry["device"]
                            }
                            for entry in self.dream_memory
                        ],
                        "seen_prompts": [(p, t) for p, t in self.seen_prompts],
                        "token_map": {str(k): v for k, v in self.token_map.items()},
                        "last_prompt_embedding": (
                            self.last_prompt_embedding.cpu().numpy().tolist()
                            if self.last_prompt_embedding is not None else None
                        ),
                        "data_exposure": self.data_exposure,
                        "last_trained": self.last_trained,
                        "global_step": self.global_step,
                        "best_valid_loss": self.best_valid_loss,
                        "patience": self.patience,
                        "lora_capacity": self.lora_capacity,
                        "temperament_score": self.temperament_score,
                        "last_temperament_score": self.last_temperament_score,
                        "temperament_history": list(self.temperament_history),
                        "confidence_history": list(self.confidence_history),
                        "sleep_confidence_sum": self.sleep_confidence_sum,
                        "sleep_confidence_count": self.sleep_confidence_count,
                        "last_weight": self.last_weight,
                        "is_sleeping": self.is_sleeping,
                        "sleep_progress": self.sleep_progress,
                        "sleep_total_loss": self.sleep_total_loss,
                        "sleep_steps": self.sleep_steps,
                        "hidden_size": self._config.hidden_size,
                        "curiosity_state": self.curiosity.to_dict()
                    }
                    self._log_event(
                        "state_serialized",
                        state_keys=list(state_dict.keys()),
                        state_hash=self.state_hash
                    )
                    return state_dict
            except Exception as e:
                self._log_error(f"State serialization failed on attempt {attempt + 1}")
                if attempt == max_retries - 1:
                    raise StateError(f"State serialization failed after {max_retries} attempts: {str(e)}")
                time.sleep(0.1)

    def from_dict(self, data: Dict[str, Any], device: torch.device, max_retries: int = 3):
        """Load state from dictionary with retry logic."""
        for attempt in range(max_retries):
            try:
                with self.lock:
                    version = data.get("version", "1.0")
                    if version != self.state_version:
                        self._log_event("state_version_mismatch", {
                            "expected": self.state_version,
                            "got": version,
                            "conversation_id": self.history.conversation_id
                        })

                    # Conversation history
                    old_conversation_id = self.history.conversation_id
                    self.history = ConversationHistory(
                        self.config_manager,
                        data.get("history", {}).get("conversation_id")
                    )
                    self.history.messages = deque(
                        data.get("history", {}).get("messages", []),
                        maxlen=self._config.max_seen_prompts
                    )
                    
                    # Log conversation ID change if it occurred
                    if old_conversation_id != self.history.conversation_id:
                        self._log_event("conversation_id_changed", {
                            "old_conversation_id": old_conversation_id,
                            "new_conversation_id": self.history.conversation_id,
                            "state_hash": self.state_hash
                        })

                    # Dream memory
                    self.dream_memory = deque(maxlen=self._config.dream_memory_maxlen)
                    for entry in data.get("dream_memory", []):
                        try:
                            tensor = torch.tensor(
                                entry["tensor"],
                                dtype=torch.float16 if self._config.quantization_mode == "fp16" else torch.float32,
                                device=device
                            )
                            if tensor.shape[-1] == self._config.hidden_size:
                                self.dream_memory.append({
                                    "tensor": tensor,
                                    "weight": float(entry["weight"]),
                                    "source": entry.get("source", "unknown"),
                                    "timestamp": entry.get("timestamp", time.time()),
                                    "device": entry.get("device", str(device))
                                })
                        except Exception as e:
                            self._log_event(
                                "failed_dream_memory_entry",
                                tensor_shape=len(entry["tensor"]) if isinstance(entry["tensor"], list) else None
                            )

                    # Seen prompts
                    self.seen_prompts = deque(
                        [(p, t) for p, t in data.get("seen_prompts", [])],
                        maxlen=self._config.max_seen_prompts
                    )
                    self._prune_seen_prompts(time.time())

                    # Token map
                    self.token_map = defaultdict(
                        lambda: [self._config.scaffold_unk_id],
                        {int(k): v for k, v in data.get("token_map", {}).items()}
                    )

                    # Last prompt embedding
                    embedding_data = data.get("last_prompt_embedding")
                    self.last_prompt_embedding = None
                    if embedding_data is not None:
                        try:
                            embedding = torch.tensor(
                                embedding_data,
                                dtype=torch.float16 if self._config.quantization_mode == "fp16" else torch.float32,
                                device=device
                            )
                            if embedding.shape[-1] == self._config.hidden_size:
                                self.last_prompt_embedding = embedding
                        except Exception as e:
                            self._log_event(
                                "failed_prompt_embedding",
                                embedding_shape=len(embedding_data) if isinstance(embedding_data, list) else None
                            )

                    # Training state
                    self.data_exposure = int(data.get("data_exposure", 0))
                    self.last_trained = float(data.get("last_trained", 0))
                    self.global_step = int(data.get("global_step", 0))
                    self.best_valid_loss = float(data.get("best_valid_loss", float('inf')))
                    self.patience = int(data.get("patience", 0))
                    self.lora_capacity = int(data.get("lora_capacity", self.lora_capacity))

                    # Behavioral state
                    self.temperament_score = float(data.get("temperament_score", 0.0))
                    self.last_temperament_score = float(data.get("last_temperament_score", 0.0))
                    self.temperament_history = deque(
                        [float(x) for x in data.get("temperament_history", [])],
                        maxlen=self._config.temperament_history_maxlen
                    )
                    self.confidence_history = deque(
                        [float(x) for x in data.get("confidence_history", [])],
                        maxlen=self._config.confidence_history_maxlen
                    )
                    self.sleep_confidence_sum = float(data.get("sleep_confidence_sum", 0.0))
                    self.sleep_confidence_count = int(data.get("sleep_confidence_count", 0))

                    # Dynamic controls
                    self.last_weight = float(data.get("last_weight", 0.0))
                    self.is_sleeping = bool(data.get("is_sleeping", False))
                    self.sleep_progress = min(int(data.get("sleep_progress", 0)), self._config.sleep_max_steps)
                    self.sleep_total_loss = float(data.get("sleep_total_loss", 0.0))
                    self.sleep_steps = min(int(data.get("sleep_steps", 0)), self._config.sleep_max_steps)

                    # Curiosity state
                    self.curiosity.from_dict(data.get("curiosity_state", {}))

                    self._update_state_hash()
                    self._log_event(
                        "state_loaded",
                        state_version=version,
                        state_hash=self.state_hash,
                        dream_memory_length=len(self.dream_memory),
                        seen_prompts_count=len(self.seen_prompts),
                        conversation_id=self.history.conversation_id
                    )
                    break
            except Exception as e:
                self._log_error(f"State loading failed on attempt {attempt + 1}", {
                    "data_keys": list(data.keys()),
                    "conversation_id": self.history.conversation_id
                })
                if attempt == max_retries - 1:
                    raise StateError(f"State loading failed after {max_retries} attempts: {str(e)}")
                time.sleep(0.1)

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
