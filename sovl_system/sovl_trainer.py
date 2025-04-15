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
import traceback

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

    def train_step_with_scaffold(
        self,
        batch: List[Dict[str, Any]],
        scaffold_provider: Optional[ScaffoldProvider] = None,
        dry_run: bool = False,
        dry_run_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Execute a single training step with scaffold support."""
        try:
            start_time = time.time()
            
            # Prepare batch data
            batch_size = len(batch)
            input_ids = torch.stack([item["input_ids"] for item in batch])
            attention_mask = torch.stack([item["attention_mask"] for item in batch])
            labels = torch.stack([item["labels"] for item in batch])
            
            # Get scaffold context if available
            scaffold_context = None
            if scaffold_provider:
                scaffold_start_time = time.time()
                scaffold_context = scaffold_provider(batch)
                scaffold_time = time.time() - scaffold_start_time
            
            # Forward pass
            forward_start_time = time.time()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                scaffold_context=scaffold_context
            )
            forward_time = time.time() - forward_start_time
            
            # Calculate loss
            loss = outputs.loss
            if self.config.grad_accum_steps > 1:
                loss = loss / self.config.grad_accum_steps
            
            # Backward pass
            backward_start_time = time.time()
            loss.backward()
            backward_time = time.time() - backward_start_time
            
            # Optimizer step
            optimizer_start_time = time.time()
            if (self.global_step + 1) % self.config.grad_accum_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            optimizer_time = time.time() - optimizer_start_time
            
            # Calculate metrics
            metrics = {
                "loss": loss.item(),
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "batch_size": batch_size,
                "grad_norm": self._calculate_grad_norm(),
                "timing": {
                    "total": time.time() - start_time,
                    "scaffold": scaffold_time if scaffold_provider else None,
                    "forward": forward_time,
                    "backward": backward_time,
                    "optimizer": optimizer_time
                },
                "memory": {
                    "allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else None,
                    "reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else None
                }
            }
            
            # Log metrics
            self.logger.record({
                "event": "training_step",
                "step": self.global_step,
                "metrics": metrics,
                "timestamp": time.time()
            })
            
            self.global_step += 1
            return loss.item(), metrics
            
        except Exception as e:
            self.logger.record({
                "error": f"Training step failed: {str(e)}",
                "step": self.global_step,
                "batch_size": len(batch),
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    def _calculate_grad_norm(self) -> float:
        """Calculate the gradient norm across all parameters."""
        try:
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            return total_norm ** 0.5
        except Exception as e:
            self.logger.record({
                "warning": f"Failed to calculate gradient norm: {str(e)}",
                "timestamp": time.time()
            })
            return 0.0

    def run_training_cycle(self, batch: List[Dict[str, Any]], scaffold_provider: Optional[ScaffoldProvider] = None) -> Tuple[float, Dict[str, Any]]:
        """Run a complete training cycle."""
        try:
            start_time = time.time()
            loss, metrics = self.train_step_with_scaffold(batch, scaffold_provider)
            
            # Log cycle metrics
            self.logger.record({
                "event": "training_cycle_complete",
                "step": self.global_step,
                "loss": loss,
                "metrics": metrics,
                "cycle_time": time.time() - start_time,
                "timestamp": time.time()
            })
            
            return loss, metrics
            
        except Exception as e:
            self.logger.record({
                "error": f"Training cycle failed: {str(e)}",
                "step": self.global_step,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

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

class TrainingCycleManager:
    """Manages the orchestration of training cycles, including sleep training."""
    
    def __init__(self, trainer: 'SOVLTrainer', config_manager: ConfigManager, logger: Logger):
        self.trainer = trainer
        self.config_manager = config_manager
        self.logger = logger
        self.training_config = config_manager.get_section("training_config")
        self.controls_config = config_manager.get_section("controls_config")
        
    def _validate_data(
        self, train_data: List[Dict[str, Any]], valid_data: List[Dict[str, Any]], batch_size: int
    ) -> Dict[str, Any]:
        """Validate training and validation data sufficiency.

        Args:
            train_data: List of training data dictionaries.
            valid_data: List of validation data dictionaries.
            batch_size: Size of each training batch.

        Returns:
            Dict indicating validation status; contains 'status' key with 'insufficient_data'
            if validation fails, else None.
        """
        if len(train_data) < batch_size or not valid_data:
            self.logger.record({
                "warning": "Insufficient data for training",
                "train_data_size": len(train_data),
                "valid_data_size": len(valid_data),
                "batch_size": batch_size,
                "timestamp": time.time()
            })
            return {"status": "insufficient_data"}
        return {}

    def _log_cycle_start(
        self, epochs: int, batch_size: int, influence_weight: float
    ) -> None:
        """Log the start of a training cycle.

        Args:
            epochs: Number of training epochs.
            batch_size: Size of each training batch.
            influence_weight: Lifecycle influence weight for the cycle.
        """
        self.logger.record({
            "event": "training_cycle_start",
            "epochs": epochs,
            "batch_size": batch_size,
            "data_exposure": self.trainer.data_exposure,
            "scaffold_influence": influence_weight,
            "timestamp": time.time()
        })

    def _handle_dry_run(
        self,
        train_data: List[Dict[str, Any]],
        scaffold_provider: Optional[Callable],
        dry_run_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle dry run execution if configured.

        Args:
            train_data: List of training data dictionaries.
            scaffold_provider: Optional provider for scaffold context.
            dry_run_params: Parameters for dry run configuration.

        Returns:
            Dict containing dry run results, including status and loss.
        """
        if dry_run_params.get("skip_training", True):
            dry_batch = train_data[:dry_run_params.get("max_samples", 2)]
            loss, metrics = self.trainer.train_step_with_scaffold(
                batch=dry_batch,
                scaffold_provider=scaffold_provider,
                dry_run=True,
                dry_run_params=dry_run_params
            )
            self.logger.record({
                "event": "dry_run_training_complete",
                "loss": loss,
                "timestamp": time.time()
            })
            return {"status": "dry_run_complete", "loss": loss}
        return {}

    def _execute_training(
        self,
        train_data: List[Dict[str, Any]],
        valid_data: List[Dict[str, Any]],
        scaffold_provider: Optional[Callable],
        epochs: int
    ) -> Dict[str, Any]:
        """Execute the core training loop.

        Args:
            train_data: List of training data dictionaries.
            valid_data: List of validation data dictionaries.
            scaffold_provider: Optional provider for scaffold context.
            epochs: Number of training epochs.

        Returns:
            Dict containing training results, including history, best validation loss,
            final epoch, and early stopping status.
        """
        return self.trainer.run_training_cycle(
            train_data=train_data,
            validation_data=valid_data,
            scaffold_provider=scaffold_provider,
            max_epochs=epochs,
            early_stopping_patience=self.training_config.get("max_patience", 3)
        )

    def _process_training_results(self, training_results: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Process training results and update trainer state.

        Args:
            training_results: Dict containing training outcomes.
            start_time: Timestamp when training started.

        Returns:
            Updated training results dict.
        """
        self.trainer.last_weight = self.trainer.get_life_curve_weight()
        self.logger.record({
            "event": "training_cycle_complete",
            "duration": time.time() - start_time,
            "last_weight": self.trainer.last_weight,
            "training_history": training_results.get("training_history", []),
            "best_val_loss": training_results.get("best_val_loss", float("inf")),
            "final_epoch": training_results.get("final_epoch", 0),
            "early_stopped": training_results.get("early_stopped", False),
            "timestamp": time.time()
        })
        return training_results

    def run_training_cycle(
        self,
        train_data: List[Dict[str, Any]],
        valid_data: List[Dict[str, Any]],
        scaffold_provider: Optional[Callable] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run a complete training cycle with validation.

        Args:
            train_data: List of training data dictionaries.
            valid_data: List of validation data dictionaries.
            scaffold_provider: Optional provider for scaffold context.
            epochs: Number of training epochs, defaults to config value.
            batch_size: Size of each training batch, defaults to config value.

        Returns:
            Dict containing training results or status if cycle is skipped.

        Raises:
            Exception: If training fails due to unexpected errors.
        """
        try:
            epochs = epochs or self.training_config.get("train_epochs", 3)
            batch_size = batch_size or self.training_config.get("batch_size", 1)

            # Validate data
            validation_result = self._validate_data(train_data, valid_data, batch_size)
            if validation_result:
                return validation_result

            # Get lifecycle weight
            influence_weight = (
                self.trainer.get_life_curve_weight()
                if self.controls_config.get("enable_lifecycle_weighting", True)
                else self.trainer.last_weight
            )

            # Log cycle start
            self._log_cycle_start(epochs, batch_size, influence_weight)

            # Handle dry run
            if self.training_config.get("dry_run", False):
                dry_run_params = self.training_config.get("dry_run_params", {})
                dry_run_result = self._handle_dry_run(train_data, scaffold_provider, dry_run_params)
                if dry_run_result:
                    return dry_run_result

            # Run training
            start_time = time.time()
            training_results = self._execute_training(train_data, valid_data, scaffold_provider, epochs)

            # Process results
            return self._process_training_results(training_results, start_time)

        except Exception as e:
            self.logger.record({
                "error": f"Training cycle failed: {str(e)}",
                "timestamp": time.time()
            })
            raise

    def run_sleep_training(self, log_entries: List[Dict[str, Any]]) -> None:
        """Run sleep training on dream-generated content."""
        try:
            if not self.controls_config.get("enable_sleep_training", True):
                self.logger.record({
                    "event": "sleep_training_skipped",
                    "reason": "Sleep training disabled",
                    "timestamp": time.time()
                })
                return
                
            self.logger.record({
                "event": "sleep_training_start",
                "timestamp": time.time()
            })
            
            # Run sleep training
            self.trainer.sleep_train(log_entries)
            self.trainer.last_trained = time.time()
            self.trainer.last_weight = self.trainer.get_life_curve_weight()
            
            # Update temperament if enabled
            if self.controls_config.get("enable_temperament", True):
                self.trainer._update_temperament()
                self.trainer.last_temperament_score = self.trainer.temperament_system.score
                
            self.logger.record({
                "event": "sleep_training_complete",
                "timestamp": time.time()
            })
            
        except Exception as e:
            self.logger.record({
                "error": f"Sleep training failed: {str(e)}",
                "timestamp": time.time()
            })
            raise

class SOVLTrainer:
    """Main trainer class for SOVL system."""
    
    def __init__(
        self,
        config: TrainingConfig,
        state: SOVLState,
        logger: Logger,
        curiosity_manager: Optional[Any] = None
    ):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
            state: System state
            logger: Logger instance
            curiosity_manager: Optional curiosity manager instance
        """
        self.config = config
        self.state = state
        self.logger = logger
        self.curiosity_manager = curiosity_manager
        self._initialize_model()
        
    def _initialize_model(self) -> None:
        """Initialize model and optimizer."""
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
        """Execute a single training step with scaffold support."""
        try:
            # Prepare batch
            prepared_batch = self._prepare_batch(batch)
            
            # Get scaffold context if available
            scaffold_context = None
            if scaffold_provider:
                scaffold_context = scaffold_provider(prepared_batch)
            
            # Forward pass
            outputs = self._forward_pass(prepared_batch, scaffold_context)
            
            # Calculate loss
            loss = self._calculate_loss(outputs, prepared_batch)
            
            # Backward pass
            if not dry_run:
                loss.backward()
                self._optimizer_step()
            
            # Update metrics
            metrics = self._update_metrics(outputs, loss)
            
            # Update curiosity if available
            if self.curiosity_manager:
                self._update_curiosity(metrics)
            
            return loss.item(), metrics
            
        except Exception as e:
            self.logger.record({
                "error": f"Training step failed: {str(e)}",
                "stack_trace": traceback.format_exc()
            })
            raise
            
    def _update_curiosity(self, metrics: Dict[str, Any]) -> None:
        """Update curiosity metrics if curiosity manager is available."""
        if not self.curiosity_manager:
            return
            
        try:
            self.curiosity_manager.update_metrics(
                question=None,  # No specific question for training
                score=metrics.get('confidence', 0.5),
                spontaneous=False,
                answered=True,
                conversation_id=self.state.conversation_id,
                state_hash=self.state.get_state_hash()
            )
        except Exception as e:
            self.logger.record({
                "error": f"Curiosity update failed: {str(e)}",
                "stack_trace": traceback.format_exc()
            })
            
    def _prepare_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Prepare batch for training."""
        # ... existing batch preparation code ...
        
    def _forward_pass(
        self,
        batch: Dict[str, torch.Tensor],
        scaffold_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Execute forward pass."""
        # ... existing forward pass code ...
        
    def _calculate_loss(
        self,
        outputs: torch.Tensor,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calculate loss."""
        # ... existing loss calculation code ...
        
    def _optimizer_step(self) -> None:
        """Execute optimizer step."""
        # ... existing optimizer step code ...
        
    def _update_metrics(
        self,
        outputs: torch.Tensor,
        loss: torch.Tensor
    ) -> Dict[str, Any]:
        """Update training metrics."""
        # ... existing metrics update code ...
