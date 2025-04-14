from dataclasses import dataclass
from typing import List, Optional, Callable, Union, Dict, Any, Tuple
import torch
import torch.nn.functional as F
import time
import uuid
import math
import os
import threading
import random
from collections import deque
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from transformers import AutoModelForCausalLM

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
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
    enable_gestation: bool = True
    enable_sleep_training: bool = True
    enable_lifecycle_weighting: bool = True
    lifecycle_capacity_factor: float = 0.01
    lifecycle_curve: str = "sigmoid_linear"
    sleep_conf_threshold: float = 0.7
    sleep_log_min: int = 10
    exposure_gain_eager: int = 3
    exposure_gain_default: int = 2
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
    confidence_history_maxlen: int = 5
    temperament_history_maxlen: int = 5
    dry_run: bool = False
    dry_run_params: Dict[str, Any] = None
    memory_threshold: float = 0.85
    memory_decay_rate: float = 0.95
    use_scaffold_memory: bool = True
    use_token_map_memory: bool = True
    scaffold_weight: float = 1.0
    enable_curiosity: bool = True
    curiosity_weight_ignorance: float = 0.7
    curiosity_weight_novelty: float = 0.3
    curiosity_pressure_threshold: float = 0.7
    curiosity_pressure_drop: float = 0.3
    curiosity_novelty_threshold_spontaneous: float = 0.9
    curiosity_novelty_threshold_response: float = 0.8
    curiosity_silence_threshold: float = 20.0
    curiosity_question_cooldown: float = 60.0
    curiosity_queue_maxlen: int = 10
    curiosity_max_new_tokens: int = 8
    curiosity_base_temperature: float = 1.1
    curiosity_temperament_influence: float = 0.4
    curiosity_top_k: int = 30

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
        assert 0.5 <= self.curiosity_novelty_threshold_spontaneous <= 1.0, "Spontaneous threshold must be in [0.5, 1.0]"
        assert 0.5 <= self.curiosity_novelty_threshold_response <= 1.0, "Response threshold must be in [0.5, 1.0]"
        assert 0.5 <= self.curiosity_pressure_threshold <= 0.9, "Pressure threshold must be in [0.5, 0.9]"
        assert 0.1 <= self.curiosity_pressure_drop <= 0.5, "Pressure drop must be in [0.1, 0.5]"
        assert 5.0 <= self.curiosity_silence_threshold <= 60.0, "Silence threshold must be in [5.0, 60.0]"
        assert 30.0 <= self.curiosity_question_cooldown <= 120.0, "Question cooldown must be in [30.0, 120.0]"
        assert 5 <= self.curiosity_queue_maxlen <= 20, "Queue maxlen must be in [5, 20]"
        assert 0.0 <= self.curiosity_weight_ignorance <= 1.0, "Ignorance weight must be in [0.0, 1.0]"
        assert 0.0 <= self.curiosity_weight_novelty <= 1.0, "Novelty weight must be in [0.0, 1.0]"
        assert 5 <= self.curiosity_max_new_tokens <= 12, "Max new tokens must be in [5, 12]"
        assert 0.5 <= self.curiosity_base_temperature <= 1.5, "Base temperature must be in [0.5, 1.5]"
        assert 0.1 <= self.curiosity_temperament_influence <= 0.6, "Temperament influence must be in [0.1, 0.6]"
        assert 10 <= self.curiosity_top_k <= 50, "Top k must be in [10, 50]"

@dataclass
class DreamMemoryConfig:
    """Configuration for dream memory behavior."""
    max_memories: int = 100
    novelty_boost: float = 0.03
    base_weight: float = 0.1
    max_weight: float = 1.5
    decay_rate: float = 0.95
    prune_threshold: float = 0.1
    noise_scale: float = 0.05
    melancholy_noise: float = 0.02

class DreamMemory:
    """Thread-safe dream memory management system."""
    def __init__(self, config: DreamMemoryConfig, device: torch.device):
        self.memory = deque(maxlen=config.max_memories)
        self.config = config
        self.device = device
        self.lock = threading.Lock()

    def add_memory(self, prompt: str, hidden_state: torch.Tensor, is_novel: bool, temperament: float = 0.0) -> None:
        """Add a memory with automatic pruning."""
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
        weight = self.config.base_weight + (self.config.novelty_boost if is_novel else 0.0)
        return min(weight, self.config.max_weight)

    def _apply_noise(self, hidden_state: torch.Tensor, temperament: float) -> torch.Tensor:
        """Apply temperament-adjusted noise to hidden state."""
        noise_level = self.config.noise_scale + (self.config.melancholy_noise if temperament < -0.5 else 0.0)
        noise = torch.randn_like(hidden_state) * noise_level
        return (hidden_state + noise).detach().cpu()

    def _maintain_memory(self) -> None:
        """Apply decay and prune weak memories."""
        updated_memory = deque(maxlen=self.memory.maxlen)
        for m in self.memory:
            new_weight = m["weight"] * self.config.decay_rate
            if new_weight > self.config.prune_threshold:
                m["weight"] = new_weight
                updated_memory.append(m)
        self.memory = updated_memory

    def get_memories(self, n: int = 5) -> List[dict]:
        """Get top-n memories by weight."""
        with self.lock:
            return sorted(self.memory, key=lambda x: -x["weight"])[:n]

    def __len__(self) -> int:
        """Return current number of memories."""
        with self.lock:
            return len(self.memory)

def collate_batch(batch: List[dict], pad_token_id: int, max_seq_length: int, tokenizer: Any) -> Dict[str, torch.Tensor]:
    """Collate batch of prompt-completion pairs into tensors."""
    prompts = [item["prompt"] for item in batch]
    completions = [item["completion"] for item in batch]
    full_texts = [p + c for p, c in zip(prompts, completions)]

    encodings = tokenizer(
        full_texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_seq_length
    )
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    labels = input_ids.clone()
    labels[labels == pad_token_id] = -100
    prompt_encodings = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_seq_length
    )
    prompt_mask = prompt_encodings["attention_mask"]
    labels = torch.where(prompt_mask.bool(), -100, input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt": prompts
    }

class LifecycleManager:
    """Manages model lifecycle and exposure tracking."""
    def __init__(self, config: TrainingConfig, model: torch.nn.Module, state: Optional[Any]):
        self.config = config
        self.state = state
        self.data_exposure = 0
        self.lora_capacity = sum(p.numel() for p in model.parameters() if p.requires_grad) * config.lifecycle_capacity_factor

    def get_life_curve_weight(self) -> float:
        """Calculate lifecycle weight based on data exposure."""
        x = self.data_exposure / self.lora_capacity if self.lora_capacity > 0 else 0
        if self.config.lifecycle_curve == "sigmoid_linear":
            return min(1.0, 1 / (1 + math.exp(-self.config.sigmoid_scale * (x - self.config.sigmoid_shift))))
        return min(1.0, 1 - math.exp(-x))

    def update_exposure(self, prompts: List[str], temperament_score: float) -> None:
        """Update data exposure based on prompts and temperament."""
        exposure_gain = self.config.exposure_gain_eager if temperament_score > 0.5 else self.config.exposure_gain_default
        if self.state:
            for prompt in prompts:
                if prompt not in self.state.seen_prompts:
                    self.state.add_seen_prompt(prompt)
                    self.data_exposure += exposure_gain

class DreamManager:
    """Handles dreaming functionality and memory integration."""
    def __init__(self, config: TrainingConfig, model: torch.nn.Module, tokenizer: Any, device: torch.device, state: Optional[Any], logger: Callable):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.state = state
        self.logger = logger
        self.dream_memory = DreamMemory(
            DreamMemoryConfig(
                max_memories=100,
                novelty_boost=config.dream_novelty_boost,
                base_weight=config.dream_memory_weight,
                decay_rate=config.dream_memory_decay,
                prune_threshold=config.dream_prune_threshold,
                noise_scale=config.dream_noise_scale,
                melancholy_noise=config.temp_melancholy_noise
            ),
            device
        )

    def should_dream(self) -> bool:
        """Determine if dreaming should occur."""
        if not self.state or not self.config.dream_temperament_on:
            return False
        swing_dream = (
            len(self.state.confidence_history) >= self.config.confidence_history_maxlen and
            torch.var(torch.tensor(list(self.state.confidence_history))).item() > self.config.dream_swing_var
        )
        lifecycle_dream = (
            abs(self.state.temperament_score - self.state.last_temperament_score) > self.config.dream_lifecycle_delta
        )
        history_dream = (
            len(self.state.temperament_history) >= self.config.temperament_history_maxlen and
            abs(torch.tensor(list(self.state.temperament_history)).mean().item() - self.state.temperament_history[0]) > 0.3
        )
        should_dream = swing_dream or lifecycle_dream or history_dream
        self.logger({
            "event": "dream_check",
            "swing_dream": swing_dream,
            "lifecycle_dream": lifecycle_dream,
            "history_dream": history_dream,
            "should_dream": should_dream,
            "timestamp": time.time(),
            "conversation_id": self.state.history.conversation_id if self.state and hasattr(self.state, "history") else str(uuid.uuid4())
        })
        return should_dream

    def dream(self, log_entries: List[dict]) -> None:
        """Execute a dream cycle."""
        if not self.config.enable_dreaming or not log_entries:
            return
        dream_prompt = self._select_dream_prompt(log_entries)
        hidden_state = self._process_dream_prompt(dream_prompt)
        is_novel, similarity = self._check_novelty(dream_prompt, hidden_state)
        self.dream_memory.add_memory(
            prompt=dream_prompt,
            hidden_state=hidden_state,
            is_novel=is_novel,
            temperament=getattr(self.state, "temperament_score", 0.0)
        )
        self.logger({
            "event": "dream_cycle",
            "prompt": dream_prompt,
            "memory_count": len(self.dream_memory),
            "top_weight": self.dream_memory.get_memories()[0]["weight"] if self.dream_memory.get_memories() else 0,
            "timestamp": time.time(),
            "conversation_id": str(uuid.uuid4())
        })

    def _select_dream_prompt(self, log_entries: List[dict]) -> str:
        """Select prompt for dreaming."""
        if not self.config.enable_prompt_driven_dreams:
            return random.choice(log_entries)["prompt"]
        reference_prompt = self.state.history.messages[-1]["prompt"] if self.state and self.state.history.messages else random.choice(log_entries)["prompt"]
        return self._select_by_similarity(reference_prompt, log_entries)

    def _select_by_similarity(self, reference: str, log_entries: List[dict]) -> str:
        """Select prompt with temperature-adjusted sampling."""
        valid_entries = [e for e in log_entries if "prompt" in e]
        if not valid_entries:
            raise ValueError("No valid prompts found")
        
        similarities = []
        for entry in valid_entries:
            inputs = self.tokenizer(
                [reference, entry["prompt"]],
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                hidden = self.model(**inputs).hidden_states[-1].mean(dim=1)
                sim = F.cosine_similarity(hidden[0], hidden[1]).item()
            similarities.append(sim)
        
        recency_weights = [i/len(valid_entries) for i in range(len(valid_entries))]
        combined = [
            (self.config.dream_prompt_weight * sim) + ((1 - self.config.dream_prompt_weight) * recency)
            for sim, recency in zip(similarities, recency_weights)
        ]
        scaled_weights = torch.softmax(torch.tensor(combined) / 0.5, dim=0).tolist()
        return random.choices([e["prompt"] for e in valid_entries], weights=scaled_weights, k=1)[0]

    def _process_dream_prompt(self, prompt: str) -> torch.Tensor:
        """Process prompt through model."""
        with torch.no_grad():
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.config.max_seq_length,
                truncation=True
            ).to(self.device)
            return self.model(**inputs).hidden_states[-1].mean(dim=1)

    def _check_novelty(self, prompt: str, hidden_state: torch.Tensor) -> Tuple[bool, float]:
        """Check prompt novelty."""
        if prompt not in getattr(self.state, "seen_prompts", set()) or len(self.dream_memory) == 0:
            return True, 0.0
        similarities = [
            F.cosine_similarity(hidden_state.flatten(), m['vector'].to(self.device).flatten(), dim=0).item()
            for m in self.dream_memory.memory
        ]
        max_similarity = max(similarities) if similarities else 0.0
        return max_similarity < 0.7, max_similarity

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get dream memory statistics."""
        with self.dream_memory.lock:
            memories = list(self.dream_memory.memory)
            weights = [m['weight'] for m in memories]
            timestamps = [m['timestamp'] for m in memories]
            return {
                'status': 'active' if memories else 'empty',
                'count': len(memories),
                'average_weight': sum(weights) / len(weights) if weights else 0.0,
                'max_weight': max(weights) if weights else 0.0,
                'min_weight': min(weights) if weights else 0.0,
                'oldest': min(timestamps) if timestamps else None,
                'newest': max(timestamps) if timestamps else None,
                'config': {
                    'max_memories': self.dream_memory.memory.maxlen,
                    'decay_rate': self.dream_memory.config.decay_rate,
                    'prune_threshold': self.dream_memory.config.prune_threshold
                }
            }

class TrainingManager:
    """Manages core training operations."""
    def __init__(self, config: TrainingConfig, model: torch.nn.Module, device: torch.device, loss_fn: Callable, tokenizer: Any):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.loss_fn = loss_fn
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = self._init_scheduler()
        self.global_step = 0
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp and device.type == "cuda" else None

    def _init_scheduler(self) -> Optional[Any]:
        """Initialize learning rate scheduler."""
        warmup_steps = self.config.warmup_steps or int(self.config.warmup_ratio * self.config.total_steps)
        if self.config.scheduler_type == "linear":
            return get_linear_schedule_with_warmup(self.optimizer, warmup_steps, self.config.total_steps)
        elif self.config.scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.total_steps - warmup_steps,
                eta_min=self.config.cosine_min_lr
            )
        return None

    def train_step(self, batch: Dict[str, torch.Tensor], scaffold_context: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        """Execute a single training step."""
        self.model.train()
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if scaffold_context:
            scaffold_context = {k: v.to(self.device) for k, v in scaffold_context.items()}

        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], scaffold_context=scaffold_context)
            loss = self.loss_fn(outputs.logits, batch["labels"]) / self.config.grad_accum_steps

        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if (self.global_step + 1) % self.config.grad_accum_steps == 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()

        self.global_step += 1
        return {
            "loss": loss.item() * self.config.grad_accum_steps,
            "learning_rate": self.optimizer.param_groups[0]["lr"]
        }

    def validate(self, data: Union[List[dict], Any], scaffold_provider: Optional[Callable] = None) -> Tuple[float, Dict[str, float]]:
        """Validate model performance."""
        self.model.eval()
        total_loss, total_correct, total_tokens, batches = 0.0, 0, 0, 0
        metrics = {metric: 0.0 for metric in self.config.metrics_to_track}
        data_iter = data if not isinstance(data, (list, tuple)) else [
            data[i:i + self.config.batch_size] for i in range(0, len(data), self.config.batch_size)
        ]

        with torch.no_grad():
            for batch in data_iter:
                if isinstance(batch, (list, tuple)):
                    batch = collate_batch(batch, self.tokenizer.pad_token_id, self.config.max_seq_length, self.tokenizer)
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                scaffold_context = scaffold_provider(batch) if scaffold_provider else None
                if scaffold_context:
                    scaffold_context = scaffold_context.to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.config.use_amp else torch.bfloat16):
                    outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                    loss = self.loss_fn(outputs.logits, batch["labels"])

                total_loss += loss.item()
                batches += 1

                if "accuracy" in metrics:
                    preds = outputs.logits.argmax(dim=-1)
                    mask = batch["labels"] != -100
                    total_correct += (preds[mask] == batch["labels"][mask]).sum().item()
                    total_tokens += mask.sum().item()
                    metrics["accuracy"] = total_correct / total_tokens if total_tokens else 0.0
                if "perplexity" in metrics:
                    metrics["perplexity"] = torch.exp(loss).item()

        avg_loss = total_loss / batches if batches else 0.0
        metrics["loss"] = avg_loss
        return avg_loss, metrics

class TrainingEventHandler:
    """Handles training-related events and updates system state."""
    
    def __init__(self, logger: Logger, state: TrainingState):
        self.logger = logger
        self.state = state

    def handle_training_complete(self, epoch: int, avg_loss: float, data_exposure: float) -> None:
        """Handle training completion event."""
        self.state.update_data_exposure(data_exposure)
        self.logger.record({
            "event": "training_complete",
            "epoch": epoch,
            "avg_loss": avg_loss,
            "data_exposure": data_exposure,
            "timestamp": time.time(),
            "state_hash": self.state.get_state_hash()
        })

    def handle_gestation_complete(self, batch_size: int, avg_loss: float) -> None:
        """Handle gestation completion event."""
        self.state.update_gestation_metrics(batch_size, avg_loss)
        self.logger.record({
            "event": "gestation_complete",
            "batch_size": batch_size,
            "avg_loss": avg_loss,
            "timestamp": time.time(),
            "state_hash": self.state.get_state_hash()
        })

    def handle_dream_complete(self, dream_prompt: str, is_novel: bool, memory_count: int) -> None:
        """Handle dream completion event."""
        self.state.update_dream_metrics(dream_prompt, is_novel, memory_count)
        self.logger.record({
            "event": "dream_complete",
            "dream_prompt": dream_prompt,
            "is_novel": is_novel,
            "memory_count": memory_count,
            "timestamp": time.time(),
            "state_hash": self.state.get_state_hash()
        })

    def handle_sleep_train_complete(self, batch_size: int, data_exposure: float) -> None:
        """Handle sleep training completion event."""
        self.state.update_sleep_metrics(batch_size, data_exposure)
        self.logger.record({
            "event": "sleep_train_complete",
            "batch_size": batch_size,
            "data_exposure": data_exposure,
            "timestamp": time.time(),
            "state_hash": self.state.get_state_hash()
        })

class TrainingWorkflowManager:
    """Manages training cycles, sleep training, and gestation/dream cycles."""
    
    def __init__(self, trainer: 'SOVLTrainer', event_handler: TrainingEventHandler):
        self.trainer = trainer
        self.event_handler = event_handler
        self.logger = trainer.logger
        self.state = trainer.state
        self.config = trainer.config

    def run_training_cycle(self, batch: List[Dict[str, Any]], scaffold_provider: Optional[ScaffoldProvider] = None) -> Tuple[float, Dict[str, Any]]:
        """Run a complete training cycle."""
        try:
            # Get batch size from memory manager
            batch_size = self.trainer.memory_manager.get_batch_size()
            
            # Run training step
            loss, metrics = self.trainer.train_step_with_scaffold(
                batch=batch,
                scaffold_provider=scaffold_provider,
                dry_run=False
            )
            
            # Update state and log event
            self.event_handler.handle_training_complete(
                epoch=self.state.epoch,
                avg_loss=loss,
                data_exposure=metrics.get("data_exposure", 0.0)
            )
            
            return loss, metrics
            
        except Exception as e:
            self.logger.error(f"Error in training cycle: {str(e)}")
            raise

    def run_sleep_training(self, batch: List[Dict[str, Any]]) -> None:
        """Run sleep training cycle."""
        try:
            # Get batch size from memory manager
            batch_size = self.trainer.memory_manager.get_batch_size()
            
            # Run sleep training step
            loss, metrics = self.trainer.train_step_with_scaffold(
                batch=batch,
                scaffold_provider=None,
                dry_run=False
            )
            
            # Update state and log event
            self.event_handler.handle_sleep_train_complete(
                batch_size=batch_size,
                data_exposure=metrics.get("data_exposure", 0.0)
            )
            
        except Exception as e:
            self.logger.error(f"Error in sleep training: {str(e)}")
            raise

    def run_gestation_cycle(self, batch: List[Dict[str, Any]]) -> None:
        """Run gestation cycle."""
        try:
            # Get batch size from memory manager
            batch_size = self.trainer.memory_manager.get_batch_size()
            
            # Run gestation step
            loss, metrics = self.trainer.train_step_with_scaffold(
                batch=batch,
                scaffold_provider=None,
                dry_run=False
            )
            
            # Update state and log event
            self.event_handler.handle_gestation_complete(
                batch_size=batch_size,
                avg_loss=loss
            )
            
        except Exception as e:
            self.logger.error(f"Error in gestation cycle: {str(e)}")
            raise

    def run_dream_cycle(self, dream_prompt: str, is_novel: bool, memory_count: int) -> None:
        """Run dream cycle."""
        try:
            # Update state and log event
            self.event_handler.handle_dream_complete(
                dream_prompt=dream_prompt,
                is_novel=is_novel,
                memory_count=memory_count
            )
            
        except Exception as e:
            self.logger.error(f"Error in dream cycle: {str(e)}")
            raise

class SOVLTrainer:
    """Main trainer class coordinating training, dreaming, and lifecycle management."""
    
    def __init__(self, config: TrainingConfig, state: SOVLState, logger: Logger):
        self.config = config
        self.state = state
        self.logger = logger
        
        # Initialize components
        self.memory_manager = MemoryManager(config, logger)
        self.event_handler = TrainingEventHandler(logger, state)
        self.workflow_manager = TrainingWorkflowManager(self, self.event_handler)
        
        # Initialize model and optimizer
        self.model = None
        self.optimizer = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the model and optimizer."""
        try:
            # Initialize model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Initialize optimizer
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate
            )
            
        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            raise
            
    def train_step_with_scaffold(
        self,
        batch: List[Dict[str, Any]],
        scaffold_provider: Optional[ScaffoldProvider] = None,
        dry_run: bool = False,
        dry_run_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Run a training step with optional scaffold context."""
        try:
            if dry_run:
                # Return mock metrics for dry run
                return 0.0, {
                    "data_exposure": dry_run_params.get("data_exposure", 0.0),
                    "confidence": dry_run_params.get("confidence", 0.0)
                }
                
            # Prepare batch
            input_ids, attention_mask, labels = collate_batch(batch)
            
            # Get scaffold context if provided
            scaffold_context = None
            if scaffold_provider:
                scaffold_context = scaffold_provider.get_context()
                
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                scaffold_context=scaffold_context
            )
            
            # Calculate loss
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Calculate metrics
            metrics = {
                "data_exposure": len(batch) / self.config.max_batch_size,
                "confidence": outputs.confidence.item()
            }
            
            return loss.item(), metrics
            
        except Exception as e:
            self.logger.error(f"Error in training step: {str(e)}")
            raise
            
    def run_training_cycle(self, batch: List[Dict[str, Any]], scaffold_provider: Optional[ScaffoldProvider] = None) -> Tuple[float, Dict[str, Any]]:
        """Run a complete training cycle."""
        return self.workflow_manager.run_training_cycle(batch, scaffold_provider)
        
    def run_sleep_training(self, batch: List[Dict[str, Any]]) -> None:
        """Run sleep training cycle."""
        self.workflow_manager.run_sleep_training(batch)
        
    def run_gestation_cycle(self, batch: List[Dict[str, Any]]) -> None:
        """Run gestation cycle."""
        self.workflow_manager.run_gestation_cycle(batch)
        
    def run_dream_cycle(self, dream_prompt: str, is_novel: bool, memory_count: int) -> None:
        """Run dream cycle."""
        self.workflow_manager.run_dream_cycle(dream_prompt, is_novel, memory_count)
