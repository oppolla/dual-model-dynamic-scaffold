import logging
from typing import Optional, Deque, Dict, Set, Tuple, DefaultDict, Any, List
from collections import deque, defaultdict
from dataclasses import dataclass
import torch
import uuid
from threading import Lock
import time

@dataclass
class CuriosityState:
    """Encapsulates all curiosity-related state variables"""
    unanswered_questions: Deque[Tuple[str, float]]  # (question, score) pairs
    last_question_time: float
    pressure: float
    novelty_scores: Deque[float]  # Track novelty over time
    question_count: int  # Total questions asked
    
    def __init__(self):
        self.unanswered_questions = deque(maxlen=10)
        self.last_question_time = 0.0
        self.pressure = 0.0
        self.novelty_scores = deque(maxlen=20)
        self.question_count = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unanswered_questions": list(self.unanswered_questions),
            "last_question_time": self.last_question_time,
            "pressure": self.pressure,
            "novelty_scores": list(self.novelty_scores),
            "question_count": self.question_count
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        self.unanswered_questions = deque(
            data.get("unanswered_questions", []),
            maxlen=10
        )
        self.last_question_time = data.get("last_question_time", 0.0)
        self.pressure = data.get("pressure", 0.0)
        self.novelty_scores = deque(
            data.get("novelty_scores", []),
            maxlen=20
        )
        self.question_count = data.get("question_count", 0)

class SOVLState:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SOVLState with configuration.
        Now includes integrated curiosity state management.
        """
        # Existing initialization
        self.dream_memory_maxlen = config.get("dream_memory_maxlen", 10)
        self.temperament_history_maxlen = config.get("temperament_history_maxlen", 5)
        self.confidence_history_maxlen = config.get("confidence_history_maxlen", 5)
        self.scaffold_unk_id = 0
        self.hidden_size = config.get("core_config", {}).get("hidden_size", 768)

        self.history = ConversationHistory()
        self.memory_lock = Lock()

        self.dream_memory: Deque[Tuple[torch.FloatTensor, float]] = deque(maxlen=self.dream_memory_maxlen)
        self.seen_prompts: Set[str] = set()
        self.token_map: DefaultDict[int, List[int]] = defaultdict(lambda: [self.scaffold_unk_id])
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

        # New: Integrated curiosity state
        self.curiosity = CuriosityState()
        
        self.logger = logging.getLogger(__name__)

    # Existing methods remain unchanged...
    def set_scaffold_unk_id(self, unk_id: int):
        """Set the unknown token ID for the scaffold model"""
        if not isinstance(unk_id, int):
            self.logger.error(f"unk_id must be an integer, got {type(unk_id)}")
            return
        self.scaffold_unk_id = unk_id
        self.token_map = defaultdict(lambda: [self.scaffold_unk_id], self.token_map)

    def append_dream_memory(self, tensor: torch.Tensor, weight: float):
        """Append a tensor to dream_memory with shape validation"""
        if not isinstance(tensor, torch.Tensor):
            self.logger.warning(f"Invalid dream memory tensor type: {type(tensor)}")
            return
        if not isinstance(weight, (int, float)) or weight < 0:
            self.logger.warning(f"Invalid dream memory weight: {weight}")
            return
        if tensor.shape[-1] != self.hidden_size:
            self.logger.warning(f"Dream memory tensor shape {tensor.shape} mismatches hidden_size {self.hidden_size}")
            return
        with self.memory_lock:
            self.dream_memory.append((tensor.float(), weight))

    def reset_conversation(self):
        """Start fresh conversation while retaining memory"""
        self.history = ConversationHistory()

    def add_seen_prompt(self, prompt: str):
        """Thread-safe method to add a seen prompt"""
        with self.memory_lock:
            self.seen_prompts.add(prompt)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dict for saving"""
        state_dict = {
            "history": {
                "conversation_id": self.history.conversation_id,
                "messages": list(self.history.messages)
            },
            "dream_memory": [(tensor.cpu().tolist(), weight) for tensor, weight in self.dream_memory],
            "seen_prompts": list(self.seen_prompts),
            "token_map": {str(k): v for k, v in self.token_map.items()},
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
            "scaffold_unk_id": self.scaffold_unk_id,
            "hidden_size": self.hidden_size,
            # New: Include curiosity state
            "curiosity_state": self.curiosity.to_dict()
        }
        return state_dict

    def from_dict(self, data: Dict[str, Any], device: torch.device):
        """Load state from dict"""
        # Existing loading logic...
        self.history = ConversationHistory(data["history"]["conversation_id"])
        self.history.messages = deque(data["history"]["messages"], maxlen=10)

        try:
            self.dream_memory = deque(
                [(torch.tensor(m, dtype=torch.float32).to(device), w) for m, w in data.get("dream_memory", [])],
                maxlen=self.dream_memory_maxlen
            )
        except (TypeError, ValueError) as e:
            self.logger.error(f"Error loading dream_memory: {e}")
            self.dream_memory = deque(maxlen=self.dream_memory_maxlen)

        self.seen_prompts = set(data.get("seen_prompts", []))

        loaded_map = data.get("token_map", {})
        self.token_map = defaultdict(lambda: [self.scaffold_unk_id],
                                     {int(k): v for k, v in loaded_map.items()})

        # Training state
        self.data_exposure = data.get("data_exposure", 0)
        self.last_trained = data.get("last_trained", 0)
        self.global_step = data.get("global_step", 0)
        self.best_valid_loss = data.get("best_valid_loss", float('inf'))
        self.patience = data.get("patience", 0)
        self.lora_capacity = data.get("lora_capacity", 0)

        # Behavioral state
        self.temperament_score = data.get("temperament_score", 0.0)
        self.last_temperament_score = data.get("last_temperament_score", 0.0)
        self.temperament_history = deque(data.get("temperament_history", []),
                                        maxlen=self.temperament_history_maxlen)
        self.confidence_history = deque(data.get("confidence_history", []),
                                       maxlen=self.confidence_history_maxlen)
        self.sleep_confidence_sum = data.get("sleep_confidence_sum", 0.0)
        self.sleep_confidence_count = data.get("sleep_confidence_count", 0)

        # Dynamic controls
        self.last_weight = data.get("last_weight", 0.0)
        self.is_sleeping = data.get("is_sleeping", False)
        self.scaffold_unk_id = data.get("scaffold_unk_id", 0)
        self.hidden_size = data.get("hidden_size", 768)

        # New: Load curiosity state
        if "curiosity_state" in data:
            self.curiosity.from_dict(data["curiosity_state"])
        else:
            self.curiosity = CuriosityState()  # Initialize fresh if not in saved state
