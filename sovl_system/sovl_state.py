from typing import Optional, Deque, Dict, Set, Tuple, DefaultDict, Any
from collections import deque, defaultdict
from dataclasses import dataclass
import torch
import uuid
import json
import time
import random
from threading import Lock

@dataclass
class ConversationHistory:
    conversation_id: str
    messages: Deque[Dict[str, str]]  # {"prompt": str, "response": str}

    def __init__(self, conversation_id=None):
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.messages = deque(maxlen=10)

    def add_message(self, prompt: str, response: str):
        self.messages.append({"prompt": prompt, "response": response})

class CuriosityPressure:
    def __init__(self):
        self.value = 0.0

    def update(self, temperament: float, confidence: float, silence: float):
        self.value += (temperament * 0.1 + (1 - confidence) * 0.05 + silence * 0.02)
        self.value = max(0.0, min(1.0, self.value))

    def should_erupt(self, threshold):
        return self.value > threshold and random.random() < 0.3

class SOVLState:
    def __init__(self, config: Dict[str, Any]):
        # Configuration-dependent initialization
        self.dream_memory_maxlen = config.get("dream_memory_maxlen", 10)
        self.temperament_history_maxlen = config.get("temperament_history_maxlen", 5)
        self.confidence_history_maxlen = config.get("confidence_history_maxlen", 5)
        self.curiosity_queue_maxlen = config.get("curiosity_queue_maxlen", 10)
        self.scaffold_unk_id = 0  # To be set explicitly by main system

        # Conversation state
        self.history = ConversationHistory()
        self.memory_lock = Lock()
        
        # Memory systems
        self.dream_memory: Deque[Tuple[torch.Tensor, float]] = deque(maxlen=self.dream_memory_maxlen)
        self.seen_prompts: Set[str] = set()
        self.token_map: DefaultDict[int, Dict] = defaultdict(lambda: [self.scaffold_unk_id])
        self.last_prompt_embedding: Optional[torch.Tensor] = None

        # Training state
        self.data_exposure = 0
        self.last_trained = 0
        self.global_step = 0
        self.best_valid_loss = float('inf')
        self.patience = 0
        self.lora_capacity = 0  # Will be set by system

        # Behavioral state
        self.temperament_score = 0.0
        self.last_temperament_score = 0.0
        self.temperament_history: Deque[float] = deque(maxlen=self.temperament_history_maxlen)
        self.confidence_history: Deque[float] = deque(maxlen=self.confidence_history_maxlen)
        self.sleep_confidence_sum = 0.0
        self.sleep_confidence_count = 0

        # Curiosity system
        self.pressure = CuriosityPressure()
        self.unanswered_q: Deque[Tuple[str, float]] = deque(maxlen=self.curiosity_queue_maxlen)

        # Dynamic controls
        self.last_weight = 0.0
        self.is_sleeping = False
        self.sleep_progress = 0
        self.sleep_batch = []
        self.sleep_total_loss = 0.0
        self.sleep_steps = 0

    def set_scaffold_unk_id(self, unk_id: int):
        """Set the unknown token ID for the scaffold model"""
        self.scaffold_unk_id = unk_id
        # Update the defaultdict factory function
        self.token_map = defaultdict(lambda: [self.scaffold_unk_id], self.token_map)

    def reset_conversation(self):
        """Start fresh conversation while retaining memory"""
        self.history = ConversationHistory()

    def add_seen_prompt(self, prompt: str):
        """Thread-safe method to add a seen prompt"""
        with self.memory_lock:
            self.seen_prompts.add(prompt)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dict for saving"""
        return {
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
            "pressure_value": self.pressure.value,
            "unanswered_q": list(self.unanswered_q),
            "last_weight": self.last_weight,
            "is_sleeping": self.is_sleeping,
            "scaffold_unk_id": self.scaffold_unk_id
        }

    def from_dict(self, data: Dict[str, Any], device: torch.device):
        """Load state from dict"""
        self.history = ConversationHistory(data["history"]["conversation_id"])
        self.history.messages = deque(data["history"]["messages"], maxlen=10)
        
        self.dream_memory = deque(
            [(torch.tensor(m, dtype=torch.float32).to(device), w) for m, w in data.get("dream_memory", [])],
            maxlen=self.dream_memory_maxlen
        )
        
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
        
        # Curiosity system
        self.pressure.value = data.get("pressure_value", 0.0)
        self.unanswered_q = deque(data.get("unanswered_q", []), 
                                maxlen=self.curiosity_queue_maxlen)
        
        # Dynamic controls
        self.last_weight = data.get("last_weight", 0.0)
        self.is_sleeping = data.get("is_sleeping", False)
        self.scaffold_unk_id = data.get("scaffold_unk_id", 0)

    def prune_dream_memory(self, threshold: float = 0.1):
        """Remove low-weight dream memories"""
        if not isinstance(threshold, (int, float)) or threshold < 0:
            raise ValueError("Threshold must be a non-negative number")
            
        with self.memory_lock:
            self.dream_memory = deque(
                [(t, w) for t, w in self.dream_memory if w >= threshold],
                maxlen=self.dream_memory_maxlen
            )

    def update_token_map(self, token_id: int, scaffold_ids: list, weight: float = 1.0):
        """Update token mapping with new scaffold association"""
        if not isinstance(token_id, int) or token_id < 0:
            raise ValueError("Token ID must be a non-negative integer")
        if not isinstance(scaffold_ids, list) or not all(isinstance(x, int) for x in scaffold_ids):
            raise ValueError("Scaffold IDs must be a list of integers")
        if not isinstance(weight, (int, float)) or weight < 0:
            raise ValueError("Weight must be a non-negative number")
            
        with self.memory_lock:
            self.token_map[token_id] = {"ids": scaffold_ids, "weight": weight}

    def decay_memory_weights(self, decay_rate: float = 0.95):
        """Gradually forget less important memories"""
        if not isinstance(decay_rate, (int, float)) or not 0 <= decay_rate <= 1:
            raise ValueError("Decay rate must be between 0 and 1")
            
        with self.memory_lock:
            # Decay dream memory weights
            self.dream_memory = deque(
                [(t, w * decay_rate) for t, w in self.dream_memory],
                maxlen=self.dream_memory_maxlen
            )
            
            # Decay token map weights
            for token_id in list(self.token_map.keys()):
                if isinstance(self.token_map[token_id], dict):
                    self.token_map[token_id]["weight"] *= decay_rate
                    # Remove if weight becomes too small
                    if self.token_map[token_id]["weight"] < 0.01:
                        del self.token_map[token_id]
