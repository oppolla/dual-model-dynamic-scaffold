from dataclasses import dataclass
from typing import List
import torch
import threading
from collections import deque

@dataclass
class DreamMemoryConfig:
    """Centralized configuration for dream memory behavior."""
    max_memories: int = 100
    novelty_boost: float = 0.03
    base_weight: float = 0.1
    max_weight: float = 1.5
    decay_rate: float = 0.95
    prune_threshold: float = 0.1
    noise_scale: float = 0.05
    melancholy_noise: float = 0.02
    prompt_weight: float = 0.5
    confidence_history_maxlen: int = 5
    temperament_history_maxlen: int = 5
    swing_var: float = 0.1
    lifecycle_delta: float = 0.1
    temperament_on: bool = True

    def __post_init__(self):
        assert self.max_memories > 0, "Max memories must be positive"
        assert self.novelty_boost >= 0, "Novelty boost must be non-negative"
        assert self.base_weight >= 0, "Base weight must be non-negative"
        assert self.max_weight > self.base_weight, "Max weight must be greater than base weight"
        assert 0 <= self.decay_rate <= 1, "Decay rate must be in [0, 1]"
        assert 0 <= self.prune_threshold <= 1, "Prune threshold must be in [0, 1]"
        assert self.noise_scale >= 0, "Noise scale must be non-negative"
        assert self.melancholy_noise >= 0, "Melancholy noise must be non-negative"
        assert 0 <= self.prompt_weight <= 1, "Prompt weight must be in [0, 1]"
        assert self.confidence_history_maxlen > 0, "Confidence history maxlen must be positive"
        assert self.temperament_history_maxlen > 0, "Temperament history maxlen must be positive"
        assert self.swing_var >= 0, "Swing variance must be non-negative"
        assert self.lifecycle_delta >= 0, "Lifecycle delta must be non-negative"

class DreamMemory:
    """Thread-safe dream memory management system."""
    def __init__(self, config: DreamMemoryConfig, device: torch.device):
        self.memory = deque(maxlen=config.max_memories)
        self.config = config
        self.device = device
        self.lock = threading.Lock()

    def add_memory(self, prompt: str, hidden_state: torch.Tensor, is_novel: bool, temperament: float = 0.0) -> None:
        """Add and maintain dream memories with automatic pruning.

        Args:
            prompt: Input prompt.
            hidden_state: Hidden state tensor.
            is_novel: Whether the memory is novel.
            temperament: Temperament score for noise adjustment.
        """
        with self.lock:
            self._maintain_memory()
            weight = self._calculate_memory_weight(is_novel)
            noisy_state = self._apply_noise(hidden_state, temperament)
            self.memory.append({
                "vector": noisy_state,
                "weight": weight,
                "prompt": prompt,
                "timestamp": time.time()
            })

    def _calculate_memory_weight(self, is_novel: bool) -> float:
        """Calculate weight with novelty boost."""
        weight = self.config.base_weight
        if is_novel:
            weight += self.config.novelty_boost
        return min(weight, self.config.max_weight)

    def _apply_noise(self, hidden_state: torch.Tensor, temperament: float) -> torch.Tensor:
        """Apply temperament-adjusted noise to hidden state."""
        noise_level = self.config.noise_scale
        if temperament < -0.5:
            noise_level += self.config.melancholy_noise
        noise = torch.randn_like(hidden_state) * noise_level
        return (hidden_state + noise).detach().cpu()

    def _maintain_memory(self) -> None:
        """Apply decay and prune weak memories."""
        self.memory = deque(
            {**m, "weight": m["weight"] * self.config.decay_rate}
            for m in self.memory
            if m["weight"] * self.config.decay_rate > self.config.prune_threshold
        )

    def get_memories(self, n: int = 5) -> List[dict]:
        """Get top-n most relevant memories by weight."""
        with self.lock:
            return sorted(self.memory, key=lambda x: -x["weight"])[:n]

    def __len__(self) -> int:
        """Current number of memories."""
        with self.lock:
            return len(self.memory)

    def get_state(self) -> dict:
        """Get serialized state for checkpointing."""
        with self.lock:
            return {
                'memory': list(self.memory),
                'config': {
                    'max_memories': self.memory.maxlen,
                    'novelty_boost': self.config.novelty_boost,
                    'base_weight': self.config.base_weight,
                    'max_weight': self.config.max_weight,
                    'decay_rate': self.config.decay_rate,
                    'prune_threshold': self.config.prune_threshold,
                    'noise_scale': self.config.noise_scale,
                    'melancholy_noise': self.config.melancholy_noise
                }
            }

    def load_state(self, state: dict) -> None:
        """Load state from checkpoint."""
        with self.lock:
            self.memory = deque(
                state['memory'],
                maxlen=state['config']['max_memories']
            )
            self.config.novelty_boost = state['config']['novelty_boost']
            self.config.base_weight = state['config']['base_weight']
            self.config.max_weight = state['config']['max_weight']
            self.config.decay_rate = state['config']['decay_rate']
            self.config.prune_threshold = state['config']['prune_threshold']
            self.config.noise_scale = state['config']['noise_scale']
            self.config.melancholy_noise = state['config']['melancholy_noise']

    def get_stats(self) -> dict:
        """Get detailed statistics about dream memory usage."""
        base_stats = {
            'status': 'active' if self.memory else 'empty',
            'count': 0,
            'average_weight': 0.0,
            'max_weight': 0.0,
            'min_weight': 0.0,
            'oldest': None,
            'newest': None,
            'config': {
                'max_memories': self.memory.maxlen,
                'decay_rate': self.config.decay_rate,
                'prune_threshold': self.config.prune_threshold
            }
        }

        with self.lock:
            if not self.memory:
                return base_stats

            weights = [m['weight'] for m in self.memory]
            timestamps = [m['timestamp'] for m in self.memory]
            base_stats.update({
                'count': len(self.memory),
                'average_weight': sum(weights) / len(weights),
                'max_weight': max(weights),
                'min_weight': min(weights),
                'oldest': min(timestamps),
                'newest': max(timestamps)
            })
            return base_stats
