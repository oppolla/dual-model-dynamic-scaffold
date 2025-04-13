from typing import Optional, Deque, Dict, Set, Tuple, DefaultDict, Any, List
from collections import deque, defaultdict
from dataclasses import dataclass
import torch
import uuid
from threading import Lock
import time
import traceback
from sovl_logger import Logger
from sovl_config import ConfigManager
from sovl_utils import NumericalGuard, safe_compare

class StateError(Exception):
    """Raised for invalid state operations or data."""
    pass

@dataclass
class CuriosityState:
    """Encapsulates curiosity-related state variables."""
    unanswered_questions: Deque[Tuple[str, float]]  # (question, score) pairs
    last_question_time: float
    pressure: float
    novelty_scores: Deque[float]  # Track novelty over time
    question_count: int  # Total questions asked
    
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        """
        Initialize curiosity state with configuration.

        Args:
            config_manager: ConfigManager instance for parameters
            logger: Logger instance for recording events
        """
        self.max_questions = config_manager.get("controls_config.curiosity_queue_maxlen", 10)
        self.max_novelty_scores = config_manager.get("controls_config.novelty_history_maxlen", 20)
        self.decay_rate = config_manager.get("controls_config.curiosity_decay_rate", 0.9)
        self.logger = logger
        
        self.unanswered_questions = deque(maxlen=self.max_questions)
        self.last_question_time = 0.0
        self.pressure = 0.0
        self.novelty_scores = deque(maxlen=self.max_novelty_scores)
        self.question_count = 0

    def add_question(self, question: str, score: float):
        """Add a new question with score."""
        try:
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Question must be a non-empty string")
            if not isinstance(score, (int, float)) or score < 0:
                raise ValueError("Score must be a non-negative number")
            with NumericalGuard():
                self.unanswered_questions.append((question, score))
                self.question_count += 1
                self.update_pressure()
                self.logger.record({
                    "event": "question_added",
                    "question": question,
                    "score": score,
                    "question_count": self.question_count,
                    "timestamp": time.time()
                })
        except Exception as e:
            self.logger.record({
                "error": f"Failed to add question: {str(e)}",
                "question": question,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })

    def update_pressure(self):
        """Update curiosity pressure based on questions and novelty."""
        try:
            with NumericalGuard():
                base_pressure = len(self.unanswered_questions) / max(1, self.max_questions)
                novelty_avg = sum(self.novelty_scores) / max(1, len(self.novelty_scores)) if self.novelty_scores else 0.0
                self.pressure = base_pressure * (1.0 + novelty_avg) * self.decay_rate
                self.pressure = max(0.0, min(1.0, self.pressure))
                self.logger.record({
                    "event": "pressure_updated",
                    "pressure": self.pressure,
                    "unanswered_count": len(self.unanswered_questions),
                    "novelty_avg": novelty_avg,
                    "timestamp": time.time()
                })
        except Exception as e:
            self.logger.record({
                "error": f"Pressure update failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })

    def add_novelty_score(self, score: float):
        """Add a novelty score and decay existing scores."""
        try:
            if not isinstance(score, (int, float)) or score < 0:
                raise ValueError("Novelty score must be a non-negative number")
            with NumericalGuard():
                self.novelty_scores.append(score)
                self.update_pressure()
                self.logger.record({
                    "event": "novelty_score_added",
                    "score": score,
                    "novelty_scores_count": len(self.novelty_scores),
                    "timestamp": time.time()
                })
        except Exception as e:
            self.logger.record({
                "error": f"Failed to add novelty score: {str(e)}",
                "score": score,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })

    def to_dict(self) -> Dict[str, Any]:
        """Serialize curiosity state to dictionary."""
        return {
            "unanswered_questions": list(self.unanswered_questions),
            "last_question_time": self.last_question_time,
            "pressure": self.pressure,
            "novelty_scores": list(self.novelty_scores),
            "question_count": self.question_count,
            "version": "1.0"
        }

    def from_dict(self, data: Dict[str, Any]):
        """Load curiosity state from dictionary."""
        try:
            self.unanswered_questions = deque(
                [(q, float(s)) for q, s in data.get("unanswered_questions", [])],
                maxlen=self.max_questions
            )
            self.last_question_time = float(data.get("last_question_time", 0.0))
            self.pressure = float(data.get("pressure", 0.0))
            self.novelty_scores = deque(
                [float(s) for s in data.get("novelty_scores", [])],
                maxlen=self.max_novelty_scores
            )
            self.question_count = int(data.get("question_count", 0))
            self.logger.record({
                "event": "curiosity_state_loaded",
                "question_count": self.question_count,
                "pressure": self.pressure,
                "timestamp": time.time()
            })
        except Exception as e:
            self.logger.record({
                "error": f"Failed to load curiosity state: {str(e)}",
                "data_keys": list(data.keys()),
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise StateError(f"Curiosity state loading failed: {str(e)}")

class ConversationHistory:
    """Manages conversation messages with unique ID."""
    def __init__(self, conversation_id: Optional[str] = None):
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.messages = deque(maxlen=10)

    def add_message(self, role: str, content: str):
        """Add a message to the conversation."""
        self.messages.append({"role": role, "content": content})

class SOVLState:
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        """
        Initialize SOVLState with configuration and logger.

        Args:
            config_manager: ConfigManager instance for parameters
            logger: Logger instance for recording events
        """
        self.config_manager = config_manager
        self.logger = logger
        self.memory_lock = Lock()
        self.state_version = "1.0"

        # Configuration parameters
        self.dream_memory_maxlen = config_manager.get("controls_config.dream_memory_maxlen", 10)
        self.temperament_history_maxlen = config_manager.get("controls_config.temperament_history_maxlen", 5)
        self.confidence_history_maxlen = config_manager.get("controls_config.confidence_history_maxlen", 5)
        self.hidden_size = config_manager.get("core_config.hidden_size", 768)
        self.max_seen_prompts = config_manager.get("controls_config.max_seen_prompts", 1000)

        # Conversation and memory
        self.history = ConversationHistory()
        self.dream_memory: Deque[Tuple[torch.FloatTensor, float]] = deque(maxlen=self.dream_memory_maxlen)
        self.seen_prompts: Set[str] = set()
        self.token_map: DefaultDict[int, List[int]] = defaultdict(lambda: [0])
        self.last_prompt_embedding: Optional[torch.Tensor] = None

        # Training state
        self.data_exposure = 0
        self.last_trained = 0
        self.global_step = 0
        self.best_valid_loss = float('inf')
        self.patience = 0
        self.lora_capacity = 0

        # Behavioral state
        self.temperament_score = 0.0
        self.last_temperament_score = 0.0
        self.temperament_history: Deque[float] = deque(maxlen=self.temperament_history_maxlen)
        self.confidence_history: Deque[float] = deque(maxlen=self.confidence_history_maxlen)
        self.sleep_confidence_sum = 0.0
        self.sleep_confidence_count = 0

        # Dynamic controls
        self.last_weight = 0.0
        self.is_sleeping = False
        self.sleep_progress = 0
        self.sleep_batch = []
        self.sleep_total_loss = 0.0
        self.sleep_steps = 0

        # Curiosity state
        self.curiosity = CuriosityState(config_manager, logger)

        self.logger.record({
            "event": "state_initialized",
            "hidden_size": self.hidden_size,
            "dream_memory_maxlen": self.dream_memory_maxlen,
            "timestamp": time.time()
        })

    def set_scaffold_unk_id(self, unk_id: int):
        """Set the unknown token ID for the scaffold model."""
        try:
            if not isinstance(unk_id, int) or unk_id < 0:
                raise ValueError("unk_id must be a non-negative integer")
            with self.memory_lock:
                self.token_map = defaultdict(lambda: [unk_id], self.token_map)
                self.logger.record({
                    "event": "scaffold_unk_id_set",
                    "unk_id": unk_id,
                    "timestamp": time.time()
                })
        except Exception as e:
            self.logger.record({
                "error": f"Failed to set scaffold unk_id: {str(e)}",
                "unk_id": unk_id,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise StateError(f"Set scaffold unk_id failed: {str(e)}")

    def append_dream_memory(self, tensor: torch.Tensor, weight: float):
        """Append a tensor to dream_memory with shape validation."""
        try:
            with NumericalGuard():
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError(f"Invalid tensor type: {type(tensor)}")
                if tensor.shape[-1] != self.hidden_size:
                    raise ValueError(f"Tensor shape {tensor.shape} mismatches hidden_size {self.hidden_size}")
                if not isinstance(weight, (int, float)) or weight < 0:
                    raise ValueError(f"Invalid weight: {weight}")
                with self.memory_lock:
                    self.dream_memory.append((tensor.float(), weight))
                    self.logger.record({
                        "event": "dream_memory_appended",
                        "tensor_shape": list(tensor.shape),
                        "weight": weight,
                        "memory_length": len(self.dream_memory),
                        "timestamp": time.time()
                    })
        except Exception as e:
            self.logger.record({
                "error": f"Failed to append dream memory: {str(e)}",
                "tensor_shape": list(tensor.shape) if isinstance(tensor, torch.Tensor) else None,
                "weight": weight,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise StateError(f"Append dream memory failed: {str(e)}")

    def reset_conversation(self):
        """Start fresh conversation while retaining memory."""
        try:
            with self.memory_lock:
                old_id = self.history.conversation_id
                self.history = ConversationHistory()
                self.logger.record({
                    "event": "conversation_reset",
                    "old_conversation_id": old_id,
                    "new_conversation_id": self.history.conversation_id,
                    "timestamp": time.time()
                })
        except Exception as e:
            self.logger.record({
                "error": f"Conversation reset failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise StateError(f"Conversation reset failed: {str(e)}")

    def add_seen_prompt(self, prompt: str):
        """Thread-safe method to add a seen prompt."""
        try:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Prompt must be a non-empty string")
            with self.memory_lock:
                if len(self.seen_prompts) >= self.max_seen_prompts:
                    self.seen_prompts.pop()
                self.seen_prompts.add(prompt)
                self.logger.record({
                    "event": "seen_prompt_added",
                    "prompt_length": len(prompt),
                    "seen_prompts_count": len(self.seen_prompts),
                    "timestamp": time.time()
                })
        except Exception as e:
            self.logger.record({
                "error": f"Failed to add seen prompt: {str(e)}",
                "prompt": prompt,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise StateError(f"Add seen prompt failed: {str(e)}")

    def update_temperament(self, score: float):
        """Update temperament score and history."""
        try:
            with NumericalGuard():
                if not isinstance(score, (int, float)):
                    raise ValueError("Score must be a number")
                score = max(-1.0, min(1.0, score))
                with self.memory_lock:
                    self.last_temperament_score = self.temperament_score
                    self.temperament_score = score
                    self.temperament_history.append(score)
                    self.logger.record({
                        "event": "temperament_updated",
                        "score": score,
                        "history_length": len(self.temperament_history),
                        "timestamp": time.time()
                    })
        except Exception as e:
            self.logger.record({
                "error": f"Temperament update failed: {str(e)}",
                "score": score,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise StateError(f"Temperament update failed: {str(e)}")

    def update_confidence(self, confidence: float):
        """Update confidence history."""
        try:
            with NumericalGuard():
                if not isinstance(confidence, (int, float)) or confidence < 0:
                    raise ValueError("Confidence must be a non-negative number")
                with self.memory_lock:
                    self.confidence_history.append(confidence)
                    if self.is_sleeping:
                        self.sleep_confidence_sum += confidence
                        self.sleep_confidence_count += 1
                    self.logger.record({
                        "event": "confidence_updated",
                        "confidence": confidence,
                        "history_length": len(self.confidence_history),
                        "is_sleeping": self.is_sleeping,
                        "timestamp": time.time()
                    })
        except Exception as e:
            self.logger.record({
                "error": f"Confidence update failed: {str(e)}",
                "confidence": confidence,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise StateError(f"Confidence update failed: {str(e)}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        try:
            with self.memory_lock:
                state_dict = {
                    "version": self.state_version,
                    "history": {
                        "conversation_id": self.history.conversation_id,
                        "messages": list(self.history.messages)
                    },
                    "dream_memory": [(tensor.cpu().numpy().tolist(), weight) for tensor, weight in self.dream_memory],
                    "seen_prompts": list(self.seen_prompts),
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
                    "hidden_size": self.hidden_size,
                    "curiosity_state": self.curiosity.to_dict()
                }
                self.logger.record({
                    "event": "state_serialized",
                    "state_keys": list(state_dict.keys()),
                    "timestamp": time.time()
                })
                return state_dict
        except Exception as e:
            self.logger.record({
                "error": f"State serialization failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise StateError(f"State serialization failed: {str(e)}")

    def from_dict(self, data: Dict[str, Any], device: torch.device):
        """Load state from dictionary."""
        try:
            with self.memory_lock:
                version = data.get("version", "1.0")
                if version != self.state_version:
                    self.logger.record({
                        "warning": f"State version mismatch: expected {self.state_version}, got {version}",
                        "timestamp": time.time()
                    })

                # Conversation history
                self.history = ConversationHistory(data.get("history", {}).get("conversation_id"))
                self.history.messages = deque(
                    data.get("history", {}).get("messages", []),
                    maxlen=self.config_manager.get("controls_config.conversation_history_maxlen", 10)
                )

                # Dream memory
                self.dream_memory = deque(maxlen=self.dream_memory_maxlen)
                for tensor_data, weight in data.get("dream_memory", []):
                    try:
                        tensor = torch.tensor(tensor_data, dtype=torch.float32, device=device)
                        if tensor.shape[-1] == self.hidden_size:
                            self.dream_memory.append((tensor, float(weight)))
                    except Exception as e:
                        self.logger.record({
                            "warning": f"Failed to load dream memory tensor: {str(e)}",
                            "tensor_shape": tensor_data.shape if isinstance(tensor_data, list) else None,
                            "timestamp": time.time()
                        })

                # Seen prompts
                self.seen_prompts = set(data.get("seen_prompts", [])[:self.max_seen_prompts])

                # Token map
                self.token_map = defaultdict(
                    lambda: [self.config_manager.get("controls_config.scaffold_unk_id", 0)],
                    {int(k): v for k, v in data.get("token_map", {}).items()}
                )

                # Last prompt embedding
                embedding_data = data.get("last_prompt_embedding")
                self.last_prompt_embedding = (
                    torch.tensor(embedding_data, dtype=torch.float32, device=device)
                    if embedding_data is not None else None
                )

                # Training state
                self.data_exposure = int(data.get("data_exposure", 0))
                self.last_trained = float(data.get("last_trained", 0))
                self.global_step = int(data.get("global_step", 0))
                self.best_valid_loss = float(data.get("best_valid_loss", float('inf')))
                self.patience = int(data.get("patience", 0))
                self.lora_capacity = int(data.get("lora_capacity", 0))

                # Behavioral state
                self.temperament_score = float(data.get("temperament_score", 0.0))
                self.last_temperament_score = float(data.get("last_temperament_score", 0.0))
                self.temperament_history = deque(
                    [float(x) for x in data.get("temperament_history", [])],
                    maxlen=self.temperament_history_maxlen
                )
                self.confidence_history = deque(
                    [float(x) for x in data.get("confidence_history", [])],
                    maxlen=self.confidence_history_maxlen
                )
                self.sleep_confidence_sum = float(data.get("sleep_confidence_sum", 0.0))
                self.sleep_confidence_count = int(data.get("sleep_confidence_count", 0))

                # Dynamic controls
                self.last_weight = float(data.get("last_weight", 0.0))
                self.is_sleeping = bool(data.get("is_sleeping", False))
                self.sleep_progress = int(data.get("sleep_progress", 0))
                self.sleep_total_loss = float(data.get("sleep_total_loss", 0.0))
                self.sleep_steps = int(data.get("sleep_steps", 0))

                # Hidden size
                self.hidden_size = int(data.get("hidden_size", self.hidden_size))

                # Curiosity state
                self.curiosity.from_dict(data.get("curiosity_state", {}))

                self.logger.record({
                    "event": "state_loaded",
                    "state_version": version,
                    "dream_memory_length": len(self.dream_memory),
                    "seen_prompts_count": len(self.seen_prompts),
                    "timestamp": time.time()
                })

        except Exception as e:
            self.logger.record({
                "error": f"State loading failed: {str(e)}",
                "data_keys": list(data.keys()),
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise StateError(f"State loading failed: {str(e)}")
