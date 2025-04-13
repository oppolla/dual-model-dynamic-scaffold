from typing import Optional, Deque, Dict, Set, Tuple, DefaultDict, Any, List
from collections import deque, defaultdict
from dataclasses import dataclass
import torch
import uuid
from threading import Lock
import time
import traceback
import hashlib
from sovl_logger import Logger
from sovl_config import ConfigManager
from sovl_utils import NumericalGuard, safe_divide, safe_compare

class StateError(Exception):
    """Raised for invalid state operations or data."""
    pass

@dataclass
class CuriosityState:
    """Encapsulates curiosity-related state variables."""
    unanswered_questions: Deque[Tuple[str, float, Optional[torch.Tensor]]]  # (question, score, context_vector) triplets
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
        self.config_manager = config_manager
        self.logger = logger
        self.lock = Lock()

        self.max_questions = config_manager.get("controls_config.curiosity_queue_maxlen", 10)
        self.max_novelty_scores = config_manager.get("controls_config.novelty_history_maxlen", 20)
        self.decay_rate = config_manager.get("controls_config.curiosity_decay_rate", 0.9)
        self.hidden_size = config_manager.get("core_config.hidden_size", 768)
        self.question_timeout = config_manager.get("controls_config.curiosity_question_timeout", 3600.0)  # 1 hour
        
        self.unanswered_questions = deque(maxlen=self.max_questions)
        self.last_question_time = 0.0
        self.pressure = 0.0
        self.novelty_scores = deque(maxlen=self.max_novelty_scores)
        self.question_count = 0

    def add_question(self, question: str, score: float, context_vector: Optional[torch.Tensor] = None):
        """Add a new question with score and optional context vector."""
        try:
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Question must be a non-empty string")
            if not isinstance(score, (int, float)) or score < 0:
                raise ValueError("Score must be a non-negative number")
            if context_vector is not None:
                if not isinstance(context_vector, torch.Tensor):
                    raise ValueError(f"Invalid context vector type: {type(context_vector)}")
                if context_vector.shape[-1] != self.hidden_size:
                    raise ValueError(f"Context vector shape {context_vector.shape} mismatches hidden_size {self.hidden_size}")
            with self.lock:
                with NumericalGuard():
                    self.unanswered_questions.append((question, score, context_vector))
                    self.question_count += 1
                    self.last_question_time = time.time()
                    self.update_pressure()
                    self.logger.record({
                        "event": "question_added",
                        "question": question,
                        "score": score,
                        "has_context_vector": context_vector is not None,
                        "question_count": self.question_count,
                        "timestamp": self.last_question_time
                    })
        except Exception as e:
            self.logger.record({
                "error": f"Failed to add question: {str(e)}",
                "question": question,
                "score": score,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })

    def prioritize_questions(self):
        """Sort unanswered questions by score."""
        try:
            with self.lock:
                sorted_questions = sorted(self.unanswered_questions, key=lambda x: x[1], reverse=True)
                self.unanswered_questions = deque(sorted_questions, maxlen=self.max_questions)
                self.logger.record({
                    "event": "questions_prioritized",
                    "question_count": len(self.unanswered_questions),
                    "timestamp": time.time()
                })
        except Exception as e:
            self.logger.record({
                "error": f"Question prioritization failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })

    def prune_old_questions(self):
        """Remove questions older than timeout."""
        try:
            current_time = time.time()
            with self.lock:
                while self.unanswered_questions and current_time - self.last_question_time > self.question_timeout:
                    question, _, _ = self.unanswered_questions.popleft()
                    self.logger.record({
                        "event": "old_question_pruned",
                        "question": question,
                        "timestamp": current_time
                    })
                self.update_pressure()
        except Exception as e:
            self.logger.record({
                "error": f"Question pruning failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })

    def update_pressure(self):
        """Update curiosity pressure based on questions and novelty."""
        try:
            with NumericalGuard():
                base_pressure = safe_divide(
                    len(self.unanswered_questions),
                    max(1, self.max_questions),
                    logger=self.logger
                )
                novelty_avg = safe_divide(
                    sum(self.novelty_scores),
                    max(1, len(self.novelty_scores)),
                    logger=self.logger
                ) if self.novelty_scores else 0.0
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
            with self.lock:
                with NumericalGuard():
                    self.novelty_scores.append(score * self.decay_rate)
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
                    weights = weights / (weights.sum() + 1e-8)  # Normalize
                    stacked = torch.stack(vectors)
                    weighted_avg = (stacked * weights.view(-1, 1)).sum(dim=0)
                    self.logger.record({
                        "event": "context_vector_computed",
                        "vector_shape": list(weighted_avg.shape),
                        "question_count": len(vectors),
                        "timestamp": time.time()
                    })
                    return weighted_avg
        except Exception as e:
            self.logger.record({
                "error": f"Context vector computation failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
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
            self.logger.record({
                "error": f"Curiosity state serialization failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise StateError(f"Curiosity state serialization failed: {str(e)}")

    def from_dict(self, data: Dict[str, Any]):
        """Load curiosity state from dictionary."""
        try:
            with self.lock:
                version = data.get("version", "1.0")
                if version not in ["1.0", "1.1"]:
                    self.logger.record({
                        "warning": f"Unsupported curiosity state version: {version}",
                        "timestamp": time.time()
                    })
                self.unanswered_questions = deque(maxlen=self.max_questions)
                for q, s, v in data.get("unanswered_questions", []):
                    context_vector = (
                        torch.tensor(v, dtype=torch.float32)
                        if v is not None and len(v) == self.hidden_size else None
                    )
                    self.unanswered_questions.append((q, float(s), context_vector))
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
                    "version": version,
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
    def __init__(self, config_manager: ConfigManager, conversation_id: Optional[str] = None):
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.max_messages = config_manager.get("controls_config.conversation_history_maxlen", 10)
        self.messages = deque(maxlen=self.max_messages)

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
        self.state_version = "1.1"
        self.state_hash = None

        # Configuration parameters
        self.dream_memory_maxlen = config_manager.get("controls_config.dream_memory_maxlen", 10)
        self.temperament_history_maxlen = config_manager.get("controls_config.temperament_history_maxlen", 5)
        self.confidence_history_maxlen = config_manager.get("controls_config.confidence_history_maxlen", 5)
        self.hidden_size = config_manager.get("core_config.hidden_size", 768)
        self.max_seen_prompts = config_manager.get("controls_config.max_seen_prompts", 1000)
        self.quantization_mode = config_manager.get("core_config.quantization", "fp16")
        self.sleep_max_steps = config_manager.get("training_config.sleep_max_steps", 100)

        # Conversation and memory
        self.history = ConversationHistory(config_manager)
        self.dream_memory: Deque[Dict[str, Any]] = deque(maxlen=self.dream_memory_maxlen)
        self.seen_prompts: Deque[Tuple[str, float]] = deque(maxlen=self.max_seen_prompts)  # (prompt, timestamp)
        self.token_map: DefaultDict[int, List[int]] = defaultdict(
            lambda: [config_manager.get("controls_config.scaffold_unk_id", 0)]
        )
        self.last_prompt_embedding: Optional[torch.Tensor] = None

        # Training state
        self.data_exposure = 0
        self.last_trained = 0
        self.global_step = 0
        self.best_valid_loss = float('inf')
        self.patience = 0
        self.lora_capacity = config_manager.get("training_config.lora_capacity", 0)

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

        self._update_state_hash()
        self.logger.record({
            "event": "state_initialized",
            "hidden_size": self.hidden_size,
            "dream_memory_maxlen": self.dream_memory_maxlen,
            "state_hash": self.state_hash,
            "timestamp": time.time()
        })

    def _update_state_hash(self):
        """Compute a hash of critical state components for tracking changes."""
        try:
            state_str = (
                f"{self.temperament_score}:{self.curiosity.question_count}:"
                f"{len(self.dream_memory)}:{len(self.seen_prompts)}"
            )
            self.state_hash = hashlib.sha256(state_str.encode()).hexdigest()[:16]
        except Exception as e:
            self.logger.record({
                "error": f"State hash update failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })

    def set_scaffold_unk_id(self, unk_id: int):
        """Set the unknown token ID for the scaffold model."""
        try:
            if not isinstance(unk_id, int) or unk_id < 0:
                raise ValueError("unk_id must be a non-negative integer")
            with self.memory_lock:
                self.token_map = defaultdict(
                    lambda: [unk_id],
                    {k: v for k, v in self.token_map.items()}
                )
                self._update_state_hash()
                self.logger.record({
                    "event": "scaffold_unk_id_set",
                    "unk_id": unk_id,
                    "token_map_size": len(self.token_map),
                    "state_hash": self.state_hash,
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

    def append_dream_memory(self, tensor: torch.Tensor, weight: float, source: str = "unknown"):
        """Append a tensor to dream_memory with metadata."""
        try:
            with NumericalGuard(dtype=torch.float16 if self.quantization_mode == "fp16" else torch.float32):
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError(f"Invalid tensor type: {type(tensor)}")
                if tensor.shape[-1] != self.hidden_size:
                    raise ValueError(f"Tensor shape {tensor.shape} mismatches hidden_size {self.hidden_size}")
                if not isinstance(weight, (int, float)) or weight < 0:
                    raise ValueError(f"Invalid weight: {weight}")
                with self.memory_lock:
                    memory_entry = {
                        "tensor": tensor.to(dtype=torch.float16 if self.quantization_mode == "fp16" else torch.float32),
                        "weight": weight,
                        "source": source,
                        "timestamp": time.time()
                    }
                    self.dream_memory.append(memory_entry)
                    self._update_state_hash()
                    self.logger.record({
                        "event": "dream_memory_appended",
                        "tensor_shape": list(tensor.shape),
                        "weight": weight,
                        "source": source,
                        "memory_length": len(self.dream_memory),
                        "state_hash": self.state_hash,
                        "timestamp": memory_entry["timestamp"]
                    })
        except Exception as e:
            self.logger.record({
                "error": f"Failed to append dream memory: {str(e)}",
                "tensor_shape": list(tensor.shape) if isinstance(tensor, torch.Tensor) else None,
                "weight": weight,
                "source": source,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise StateError(f"Append dream memory failed: {str(e)}")

    def get_weighted_memory(self, device: torch.device) -> Optional[torch.Tensor]:
        """Compute a weighted average of dream memory tensors."""
        try:
            with self.memory_lock:
                if not self.dream_memory:
                    return None
                tensors = [entry["tensor"].to(device) for entry in self.dream_memory]
                weights = torch.tensor([entry["weight"] for entry in self.dream_memory], device=device)
                with NumericalGuard():
                    weights = weights / (weights.sum() + 1e-8)  # Normalize
                    stacked = torch.stack(tensors)
                    weighted_avg = (stacked * weights.view(-1, *([1] * (stacked.dim() - 1)))).sum(dim=0)
                    self.logger.record({
                        "event": "weighted_memory_computed",
                        "tensor_shape": list(weighted_avg.shape),
                        "memory_length": len(self.dream_memory),
                        "timestamp": time.time()
                    })
                    return weighted_avg
        except Exception as e:
            self.logger.record({
                "error": f"Weighted memory computation failed: {str(e)}",
                "memory_length": len(self.dream_memory),
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            return None

    def reset_conversation(self):
        """Start fresh conversation while retaining memory."""
        try:
            with self.memory_lock:
                old_id = self.history.conversation_id
                self.history = ConversationHistory(self.config_manager)
                self._update_state_hash()
                self.logger.record({
                    "event": "conversation_reset",
                    "old_conversation_id": old_id,
                    "new_conversation_id": self.history.conversation_id,
                    "state_hash": self.state_hash,
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
        """Thread-safe method to add a seen prompt with timestamp."""
        try:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Prompt must be a non-empty string")
            with self.memory_lock:
                current_time = time.time()
                self.seen_prompts.append((prompt, current_time))
                self._prune_seen_prompts(current_time)
                self._update_state_hash()
                self.logger.record({
                    "event": "seen_prompt_added",
                    "prompt_length": len(prompt),
                    "seen_prompts_count": len(self.seen_prompts),
                    "state_hash": self.state_hash,
                    "timestamp": current_time
                })
        except Exception as e:
            self.logger.record({
                "error": f"Failed to add seen prompt: {str(e)}",
                "prompt": prompt,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise StateError(f"Add seen prompt failed: {str(e)}")

    def _prune_seen_prompts(self, current_time: float):
        """Remove old prompts based on timeout."""
        timeout = self.config_manager.get("controls_config.prompt_timeout", 86400.0)  # 1 day
        while self.seen_prompts and current_time - self.seen_prompts[0][1] > timeout:
            prompt, _ = self.seen_prompts.popleft()
            self.logger.record({
                "event": "seen_prompt_pruned",
                "prompt": prompt,
                "timestamp": current_time
            })

    def update_temperament(self, score: float):
        """Update temperament score and history with decay."""
        try:
            with NumericalGuard():
                if not isinstance(score, (int, float)):
                    raise ValueError("Score must be a number")
                score = max(-1.0, min(1.0, score))
                decay = self.config_manager.get("controls_config.temperament_decay_rate", 0.95)
                with self.memory_lock:
                    self.last_temperament_score = self.temperament_score
                    self.temperament_score = score * decay + self.temperament_score * (1 - decay)
                    self.temperament_history.append(self.temperament_score)
                    self._update_state_hash()
                    self.logger.record({
                        "event": "temperament_updated",
                        "score": self.temperament_score,
                        "raw_score": score,
                        "history_length": len(self.temperament_history),
                        "state_hash": self.state_hash,
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
                    self._update_state_hash()
                    self.logger.record({
                        "event": "confidence_updated",
                        "confidence": confidence,
                        "history_length": len(self.confidence_history),
                        "is_sleeping": self.is_sleeping,
                        "state_hash": self.state_hash,
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

    def set_last_prompt_embedding(self, embedding: torch.Tensor):
        """Set the last prompt embedding with validation."""
        try:
            with NumericalGuard():
                if not isinstance(embedding, torch.Tensor):
                    raise ValueError(f"Invalid embedding type: {type(embedding)}")
                if embedding.shape[-1] != self.hidden_size:
                    raise ValueError(f"Embedding shape {embedding.shape} mismatches hidden_size {self.hidden_size}")
                with self.memory_lock:
                    self.last_prompt_embedding = embedding.to(
                        dtype=torch.float16 if self.quantization_mode == "fp16" else torch.float32
                    )
                    self._update_state_hash()
                    self.logger.record({
                        "event": "last_prompt_embedding_set",
                        "embedding_shape": list(embedding.shape),
                        "state_hash": self.state_hash,
                        "timestamp": time.time()
                    })
        except Exception as e:
            self.logger.record({
                "error": f"Failed to set last prompt embedding: {str(e)}",
                "embedding_shape": list(embedding.shape) if isinstance(embedding, torch.Tensor) else None,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise StateError(f"Set last prompt embedding failed: {str(e)}")

    def to_dict(self, max_retries: int = 3) -> Dict[str, Any]:
        """Serialize state to dictionary with retry logic."""
        for attempt in range(max_retries):
            try:
                with self.memory_lock:
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
                                "timestamp": entry["timestamp"]
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
                        "hidden_size": self.hidden_size,
                        "curiosity_state": self.curiosity.to_dict()
                    }
                    self.logger.record({
                        "event": "state_serialized",
                        "state_keys": list(state_dict.keys()),
                        "state_hash": self.state_hash,
                        "timestamp": time.time()
                    })
                    return state_dict
            except Exception as e:
                self.logger.record({
                    "error": f"State serialization failed on attempt {attempt + 1}: {str(e)}",
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc()
                })
                if attempt == max_retries - 1:
                    raise StateError(f"State serialization failed after {max_retries} attempts: {str(e)}")
                time.sleep(0.1)  # Brief pause before retry

    def from_dict(self, data: Dict[str, Any], device: torch.device, max_retries: int = 3):
        """Load state from dictionary with retry logic."""
        for attempt in range(max_retries):
            try:
                with self.memory_lock:
                    version = data.get("version", "1.0")
                    if version != self.state_version:
                        self.logger.record({
                            "warning": f"State version mismatch: expected {self.state_version}, got {version}",
                            "timestamp": time.time()
                        })

                    # Conversation history
                    self.history = ConversationHistory(
                        self.config_manager,
                        data.get("history", {}).get("conversation_id")
                    )
                    self.history.messages = deque(
                        data.get("history", {}).get("messages", []),
                        maxlen=self.config_manager.get("controls_config.conversation_history_maxlen", 10)
                    )

                    # Dream memory
                    self.dream_memory = deque(maxlen=self.dream_memory_maxlen)
                    for entry in data.get("dream_memory", []):
                        try:
                            tensor = torch.tensor(
                                entry["tensor"],
                                dtype=torch.float16 if self.quantization_mode == "fp16" else torch.float32,
                                device=device
                            )
                            if tensor.shape[-1] == self.hidden_size:
                                self.dream_memory.append({
                                    "tensor": tensor,
                                    "weight": float(entry["weight"]),
                                    "source": entry.get("source", "unknown"),
                                    "timestamp": entry.get("timestamp", time.time())
                                })
                        except Exception as e:
                            self.logger.record({
                                "warning": f"Failed to load dream memory entry: {str(e)}",
                                "tensor_shape": len(entry["tensor"]) if isinstance(entry["tensor"], list) else None,
                                "timestamp": time.time()
                            })

                    # Seen prompts
                    self.seen_prompts = deque(
                        [(p, t) for p, t in data.get("seen_prompts", [])],
                        maxlen=self.max_seen_prompts
                    )
                    self._prune_seen_prompts(time.time())

                    # Token map
                    self.token_map = defaultdict(
                        lambda: [self.config_manager.get("controls_config.scaffold_unk_id", 0)],
                        {int(k): v for k, v in data.get("token_map", {}).items()}
                    )

                    # Last prompt embedding
                    embedding_data = data.get("last_prompt_embedding")
                    self.last_prompt_embedding = None
                    if embedding_data is not None:
                        try:
                            embedding = torch.tensor(
                                embedding_data,
                                dtype=torch.float16 if self.quantization_mode == "fp16" else torch.float32,
                                device=device
                            )
                            if embedding.shape[-1] == self.hidden_size:
                                self.last_prompt_embedding = embedding
                        except Exception as e:
                            self.logger.record({
                                "warning": f"Failed to load last prompt embedding: {str(e)}",
                                "embedding_shape": len(embedding_data) if isinstance(embedding_data, list) else None,
                                "timestamp": time.time()
                            })

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
                    self.sleep_progress = min(int(data.get("sleep_progress", 0)), self.sleep_max_steps)
                    self.sleep_total_loss = float(data.get("sleep_total_loss", 0.0))
                    self.sleep_steps = min(int(data.get("sleep_steps", 0)), self.sleep_max_steps)

                    # Hidden size
                    self.hidden_size = int(data.get("hidden_size", self.hidden_size))

                    # Curiosity state
                    self.curiosity.from_dict(data.get("curiosity_state", {}))

                    self._update_state_hash()
                    self.logger.record({
                        "event": "state_loaded",
                        "state_version": version,
                        "state_hash": self.state_hash,
                        "dream_memory_length": len(self.dream_memory),
                        "seen_prompts_count": len(self.seen_prompts),
                        "timestamp": time.time()
                    })
                    break
            except Exception as e:
                self.logger.record({
                    "error": f"State loading failed on attempt {attempt + 1}: {str(e)}",
                    "data_keys": list(data.keys()),
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc()
                })
                if attempt == max_retries - 1:
                    raise StateError(f"State loading failed after {max_retries} attempts: {str(e)}")
                time.sleep(0.1)  # Brief pause before retry
