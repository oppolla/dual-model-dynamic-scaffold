from dataclasses import dataclass
from typing import List, Optional, Callable, Union, Tuple
import torch
import torch.nn.functional as F
import time
import uuid 
import math
import os
import threading
import random
from collections import deque  # Added for dream memory
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup

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
    scheduler_type: str = "linear"  # Options: linear, cosine, constant
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
    accumulation_steps: int = 4
    exposure_gain_eager: int = 3
    exposure_gain_default: int = 2
    dream_memory_weight: float = 0.1
    enable_dreaming: bool = True
    repetition_n: int = 3
    sigmoid_scale: float = 0.5
    sigmoid_shift: float = 5.0
    # Dream-specific parameters
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

    def __post_init__(self):
        if self.metrics_to_track is None:
            self.metrics_to_track = ["loss", "accuracy", "confidence"]
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
    """Centralized configuration for dream memory behavior"""
    max_memories: int = 100
    novelty_boost: float = 0.03
    base_weight: float = 0.1
    max_weight: float = 1.5
    decay_rate: float = 0.95
    prune_threshold: float = 0.1
    noise_scale: float = 0.05
    melancholy_noise: float = 0.02

class DreamMemory:
    """Thread-safe dream memory management system"""
    def __init__(self, config: DreamMemoryConfig, device: torch.device):
        self.memory = deque(maxlen=config.max_memories)
        self.config = config
        self.device = device
        self.lock = threading.Lock()
    
    def add_memory(self, 
                 prompt: str, 
                 hidden_state: torch.Tensor, 
                 is_novel: bool,
                 temperament: float = 0.0) -> None:
        """Add and maintain dream memories with automatic pruning"""
        with self.lock:
            # 1. Apply decay and pruning to existing memories
            self._maintain_memory()
            
            # 2. Calculate memory weight
            weight = self._calculate_memory_weight(is_novel)
            
            # 3. Apply noise based on temperament
            noisy_state = self._apply_noise(hidden_state, temperament)
            
            # 4. Store the new memory
            self.memory.append({
                "vector": noisy_state,
                "weight": weight,
                "prompt": prompt,
                "timestamp": time.time()
            })
    
    def _calculate_memory_weight(self, is_novel: bool) -> float:
        """Calculate weight with novelty boost"""
        weight = self.config.base_weight
        if is_novel:
            weight += self.config.novelty_boost
        return min(weight, self.config.max_weight)
    
    def _apply_noise(self, hidden_state: torch.Tensor, temperament: float) -> torch.Tensor:
        """Apply temperament-adjusted noise to hidden state"""
        noise_level = self.config.noise_scale
        if temperament < -0.5:  # Melancholy state
            noise_level += self.config.melancholy_noise
        
        noise = torch.randn_like(hidden_state) * noise_level
        return (hidden_state + noise).detach().cpu()
    
    def _maintain_memory(self) -> None:
        """Apply decay and prune weak memories"""
        self.memory = deque(
            {**m, "weight": m["weight"] * self.config.decay_rate}
            for m in self.memory
            if m["weight"] * self.config.decay_rate > self.config.prune_threshold
        )
    
    def get_memories(self, n: int = 5) -> List[dict]:
        """Get top-n most relevant memories by weight"""
        with self.lock:
            return sorted(self.memory, key=lambda x: -x["weight"])[:n]
    
    def __len__(self) -> int:
        """Current number of memories"""
        with self.lock:
            return len(self.memory)        

def collate_batch(batch: List[dict], pad_token_id: int, max_seq_length: int, tokenizer) -> dict:
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

class SOVLTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainingConfig,
        device: torch.device,
        loss_fn: Callable,
        logger: Optional[Callable] = None,
        memory_lock: Optional[threading.Lock] = None,
        tokenizer: Optional[Callable] = None,
        state: Optional[object] = None
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.loss_fn = loss_fn
        self.logger = logger or (lambda x: None)
        self.memory_lock = memory_lock or threading.Lock()
        self.tokenizer = tokenizer
        self.state = state
        self.global_step = 0
        self.best_valid_loss = float("inf")
        self.patience = 0
        self.loss_weight_fn = None
        self.memory_check = None
        self.data_exposure = 0
        self.lora_capacity = sum(p.numel() for p in model.parameters() if p.requires_grad) * config.lifecycle_capacity_factor
        self.scaffold_context = None
        self.is_gestating = False
        self.gestation_progress = 0
        self.gestation_batch = []
        self.gestation_total_loss = 0.0
        self.gestation_steps = 0
        self.sleep_progress = 0
        self.sleep_batch = []
        self.sleep_total_loss = 0.0
        self.sleep_steps = 0

        # Callback system for trainer events
        self.callbacks = {
            "on_training_complete": None,
            "on_gestation_complete": None,
            "on_dream_complete": None,
            "on_sleep_train_complete": None
        }

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Scheduler
        if config.scheduler_type == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=int(config.warmup_steps or config.warmup_ratio * config.total_steps),
                num_training_steps=config.total_steps
            )
        elif config.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.total_steps - int(config.warmup_steps or config.warmup_ratio * config.total_steps),
                eta_min=config.cosine_min_lr
            )
        else:
            self.scheduler = None

        # Apply dropout if specified
        if config.dropout_rate > 0:
            for module in self.model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = config.dropout_rate

    def register_callback(self, event: str, callback: Callable):
        """Register a callback for a specific event."""
        if event in self.callbacks:
            self.callbacks[event] = callback
        else:
            raise ValueError(f"Unknown event: {event}")
            
    def on_training_complete(self, epoch: int, avg_loss: float, data_exposure: float):
        """Notify completion of training cycle."""
        if self.callbacks["on_training_complete"]:
            self.callbacks["on_training_complete"](epoch, avg_loss, data_exposure)
        return {"epoch": epoch, "avg_loss": avg_loss, "data_exposure": data_exposure}

    def on_gestation_complete(self, batch_size: int, avg_loss: float):
        """Notify completion of gestation."""
        if self.callbacks["on_gestation_complete"]:
            self.callbacks["on_gestation_complete"](batch_size, avg_loss)
        return {"batch_size": batch_size, "avg_loss": avg_loss}

    def on_dream_complete(self, prompt: str, novelty: bool, memory_count: int):
        """Notify completion of dreaming."""
        if self.callbacks["on_dream_complete"]:
            self.callbacks["on_dream_complete"](prompt, novelty, memory_count)
        return {"prompt": prompt, "novelty": novelty, "memory_count": memory_count}

    def on_sleep_train_complete(self, batch_size: int, data_exposure: float):
        """Notify completion of sleep training."""
        if self.callbacks["on_sleep_train_complete"]:
            self.callbacks["on_sleep_train_complete"](batch_size, data_exposure)
        return {"batch_size": batch_size, "data_exposure": data_exposure}        

    def get_life_curve_weight(self):
        """Calculate lifecycle weight based on data exposure."""
        x = self.data_exposure / self.lora_capacity if self.lora_capacity > 0 else 0
        if self.config.lifecycle_curve == "sigmoid_linear":
            weight = 1 / (1 + math.exp(-self.config.sigmoid_scale * (x - self.config.sigmoid_shift)))
        else:
            weight = 1 - math.exp(-x)
        return min(1.0, weight)

    def update_exposure(self, prompts: List[str], temperament_score: float):
        """Update data exposure based on prompts and temperament."""
        exposure_gain = (
            self.config.exposure_gain_eager
            if temperament_score > 0.5
            else self.config.exposure_gain_default
        )
        for prompt in prompts:
            if self.state and prompt not in self.state.seen_prompts:
                self.state.add_seen_prompt(prompt)
                self.data_exposure += exposure_gain

    def has_repetition(self, output_ids: torch.Tensor, special_ids: set) -> bool:
        """Check for repeated sequences in output_ids."""
        ids = output_ids.tolist()
        filtered = [i for i in ids if i not in special_ids]
        n = self.config.repetition_n
        for i in range(len(filtered) - 2 * n):
            if filtered[i:i + n] == filtered[i + n:i + 2 * n]:
                return True
        return False

    def get_loss_weight(self, batch: dict) -> float:
        """Calculate loss weight based on log entries."""
        log_entries = self.logger.read() if hasattr(self.logger, "read") else []
        prompts = batch['prompt']
        for prompt in prompts:
            if any(e["prompt"] == prompt and e.get("is_system_question", False) and e["response"] for e in log_entries):
                return 1.2
        return 1.0

    def train_step(
        self,
        batch: dict,
        scaffold_context: Optional[torch.Tensor] = None,
        grad_clip: bool = True,
        loss_weight_fn: Optional[Callable] = None,
        dry_run: bool = False,
        memory_check: Optional[Callable] = None
    ) -> Tuple[Optional[float], Optional[float]]:
        """Robust training step with error handling and validation."""
        # Input validation
        required_keys = {"input_ids", "attention_mask", "labels", "prompt"}
        if not all(k in batch for k in required_keys):
            self.logger({
                "event": "invalid_batch",
                "missing_keys": str(required_keys - set(batch.keys())),
                "timestamp": time.time()
            })
            return None, None

        try:
            # Memory check if provided
            if memory_check:
                memory_check()

            # Device setup
            device = self.device
            self.scaffold_context = scaffold_context.to(device) if scaffold_context is not None else None

            # Dry run validation
            if dry_run:
                with torch.no_grad():
                    self.model.eval()
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16 if self.config.use_amp else torch.bfloat16
                    ):
                        inputs = {
                            "input_ids": batch["input_ids"].to(device),
                            "attention_mask": batch["attention_mask"].to(device)
                        }
                        outputs = self.model(**inputs)
                        loss = self.loss_fn(outputs.logits, batch["labels"].to(device))
                    return float(loss.item()), None

            # Main training
            self.model.train()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if self.config.use_amp else torch.bfloat16
            ):
                # Forward pass
                inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device)
                }
                outputs = self.model(**inputs)

                # Loss calculation
                labels = batch["labels"].to(device)
                loss = self.loss_fn(outputs.logits, labels)

                # Loss validation
                if torch.isnan(loss) or torch.isinf(loss):
                    raise ValueError(f"Invalid loss value: {loss.item()}")

                # Weight application
                weight = float(loss_weight_fn(batch)) if loss_weight_fn else 1.0
                safe_weight = min(max(weight, 0.5), 2.0)  # Clamped to reasonable range
                loss *= safe_weight

                if self.config.enable_lifecycle_weighting:
                    lifecycle_weight = min(self.get_life_curve_weight(), 1.0)
                    loss *= lifecycle_weight

            # Backward pass
            (loss / self.config.grad_accum_steps).backward()

            # Gradient update
            if (self.global_step + 1) % self.config.grad_accum_steps == 0:
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.max_grad_norm
                    )
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

            # Metrics calculation
            confidence = None
            if not dry_run and ("accuracy" in self.config.metrics_to_track 
                               or "confidence" in self.config.metrics_to_track):
                with torch.no_grad():
                    probs = torch.softmax(outputs.logits, dim=-1)
                    confidence = float(torch.max(probs, dim=-1).values.mean())

            self.global_step += 1
            return float(loss.item()), confidence

        except Exception as e:
            self.logger({
                "event": "train_step_failed",
                "error": str(e),
                "step": self.global_step,
                "timestamp": time.time()
            })
            self.optimizer.zero_grad()
            return None, None

    def validate(self, data: Union[List[dict], 'DataLoader'], scaffold_provider: Optional[Callable] = None) -> Tuple[float, dict]:
        """Validate model on provided data, returning loss and metrics."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        batches = 0
        metrics = {metric: 0.0 for metric in self.config.metrics_to_track}

        if isinstance(data, (list, tuple)):
            data_iter = [data[i:i + self.config.batch_size] for i in range(0, len(data), self.config.batch_size)]
        else:
            data_iter = data

        with torch.no_grad():
            for batch in data_iter:
                if isinstance(batch, (list, tuple)):
                    batch = collate_batch(batch, self.tokenizer.pad_token_id, self.config.max_seq_length, self.tokenizer)
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                scaffold_context = scaffold_provider(batch) if scaffold_provider else self.scaffold_context
                if scaffold_context is not None:
                    scaffold_context = scaffold_context.to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.config.use_amp else torch.bfloat16):
                    outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                    loss = self.loss_fn(outputs.logits, batch["labels"])

                total_loss += loss.item()
                batches += 1

                if "accuracy" in self.config.metrics_to_track:
                    preds = outputs.logits.argmax(dim=-1)
                    mask = batch["labels"] != -100
                    correct = (preds[mask] == batch["labels"][mask]).sum().item()
                    total_correct += correct
                    total_tokens += mask.sum().item()
                    metrics["accuracy"] = total_correct / total_tokens if total_tokens > 0 else 0.0
                if "perplexity" in self.config.metrics_to_track:
                    perplexity = torch.exp(loss).item()
                    metrics["perplexity"] = perplexity

        avg_loss = total_loss / batches if batches > 0 else 0.0
        metrics["loss"] = avg_loss
        self.logger({
            "event": "validation",
            "loss": avg_loss,
            "metrics": metrics,
            "timestamp": time.time()
        })
        return avg_loss, metrics

    def save_checkpoint(self, step: int, suffix: Optional[str] = None):
        checkpoint_dir = self.config.checkpoint_path
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_name = f"checkpoint_{step}{f'_{suffix}' if suffix else ''}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

        # Prepare memory data for saving (thread-safe)
        memory_data = None
        if hasattr(self, 'dream_memory'):
            with self.dream_memory.lock:
                memory_data = {
                    'memory': list(self.dream_memory.memory),  # Convert deque to list
                    'config': {
                        'max_memories': self.dream_memory.memory.maxlen,
                        'novelty_boost': self.dream_memory.config.novelty_boost,
                        # Include other config params as needed
                    }
                }

        state_dict = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.global_step,
            "best_valid_loss": self.best_valid_loss,
            "patience": self.patience,
            "data_exposure": self.data_exposure,
            "dream_memory": memory_data  # Add serialized memory
        }

        torch.save(state_dict, checkpoint_path)
        self.logger({
            "event": "checkpoint_saved",
            "path": checkpoint_path,
            "step": step,
            "memory_count": len(self.dream_memory) if hasattr(self, 'dream_memory') else 0,
            "timestamp": time.time()
        })
        print(f"Checkpoint saved: {checkpoint_path} (Memories: {len(self.dream_memory) if hasattr(self, 'dream_memory') else 0})")

    def load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            self.logger({
                "event": "checkpoint_load_failed",
                "path": checkpoint_path,
                "error": "File not found",
                "timestamp": time.time()
            })
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location=self.device)

        # Load core states
        self.model.load_state_dict(state_dict["model_state"])
        self.optimizer.load_state_dict(state_dict["optimizer_state"])
        if state_dict["scheduler_state"] and self.scheduler:
            self.scheduler.load_state_dict(state_dict["scheduler_state"])
        self.global_step = state_dict["global_step"]
        self.best_valid_loss = state_dict["best_valid_loss"]
        self.patience = state_dict["patience"]
        self.data_exposure = state_dict.get("data_exposure", 0)

        # Restore dream memory if exists in checkpoint
        if 'dream_memory' in state_dict and state_dict['dream_memory'] is not None:
            if not hasattr(self, 'dream_memory'):
                self._init_dream_memory()  # Initialize if not exists

            with self.dream_memory.lock:
                # Restore memory content and configuration
                self.dream_memory.memory = deque(
                    state_dict['dream_memory']['memory'],
                    maxlen=state_dict['dream_memory']['config']['max_memories']
                )
                # Update config if needed
                self.dream_memory.config.novelty_boost = state_dict['dream_memory']['config']['novelty_boost']

        self.logger({
            "event": "checkpoint_loaded",
            "path": checkpoint_path,
            "step": self.global_step,
            "memory_count": len(self.dream_memory) if hasattr(self, 'dream_memory') else 0,
            "timestamp": time.time()
        })
        print(f"Checkpoint loaded: {checkpoint_path} at step {self.global_step} (Memories: {len(self.dream_memory) if hasattr(self, 'dream_memory') else 0})")

    def should_stop(self) -> bool:
        """Check if training should stop based on early stopping criteria."""
        return self.patience >= self.config.max_patience

    def gestate(self, log_entries: List[dict], resume: bool = False) -> bool:
        """Perform gestation training on log entries."""
        if not self.config.enable_gestation:
            return False

        if not log_entries:
            print("No log data to gestate.")
            self._reset_gestation_state()
            return False

        if not resume and not self._should_gestate(log_entries):
            return False

        if not resume:
            self.is_gestating = True
            self.gestation_progress = 0
            self.gestation_batch = [
                {"prompt": entry["prompt"], "completion": entry["response"]}
                for entry in log_entries if "prompt" in entry and "response" in entry
            ]
            self.gestation_total_loss = 0.0
            self.gestation_steps = 0
            print("\nTrainer Gestating...")
            if self.config.enable_dreaming and self._should_dream():
                self._dream()

            self.data_exposure += sum(
                len(entry["prompt"]) + len(entry["response"])
                for entry in log_entries
                if "prompt" in entry and "response" in entry
            )

        if self.gestation_progress < len(self.gestation_batch):
            batch = [self.gestation_batch[self.gestation_progress]]
            formatted_batch = collate_batch(
                batch,
                self.tokenizer.pad_token_id,
                self.config.max_seq_length,
                self.tokenizer
            )
            loss, confidence = self.train_step(
                batch=formatted_batch,
                scaffold_context=self.scaffold_context,
                grad_clip=True,
                loss_weight_fn=self.loss_weight_fn,
                dry_run=False,
                memory_check=self.memory_check
            )

            self.gestation_total_loss += loss if loss is not None else 0.0
            self.gestation_steps += 1 if loss is not None else 0
            self.gestation_progress += 1
            if self.gestation_steps % 5 == 0 and self.gestation_steps > 0:
                print(f"Gestation progress: {self.gestation_progress}/{len(self.gestation_batch)}, loss: {self.gestation_total_loss / self.gestation_steps:.4f}")
            return True

        avg_loss = self.gestation_total_loss / self.gestation_steps if self.gestation_steps > 0 else 0
        print(f"\nGestation complete: {len(self.gestation_batch)}/{len(self.gestation_batch)}, loss: {avg_loss:.4f}")
        self._reset_gestation_state()
        return False

    def _should_gestate(self, log_entries: List[dict]) -> bool:
        """Determine if gestation should proceed."""
        if len(log_entries) < self.config.sleep_log_min:
            print(f"Gestation check: Log size {len(log_entries)} < {self.config.sleep_log_min}. No gestation.")
            return False
        avg_confidence = self.state.sleep_confidence_sum / self.state.sleep_confidence_count if self.state.sleep_confidence_count > 0 else 0.5
        should_gestate = (len(log_entries) >= self.config.sleep_log_min) and (self.state.sleep_confidence_count == 0 or avg_confidence > self.config.sleep_conf_threshold)
        print(f"Gestation check: Confidence {avg_confidence:.2f} > {self.config.sleep_conf_threshold}, Log {len(log_entries)} >= {self.config.sleep_log_min}, Gestate: {should_gestate}")
        return should_gestate

    def _reset_gestation_state(self):
        """Reset gestation-related state."""
        self.is_gestating = False
        self.gestation_progress = 0
        self.gestation_batch = []
        self.gestation_total_loss = 0.0
        self.gestation_steps = 0

    def _should_dream(self):
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
        history_dream = False
        if len(self.state.temperament_history) >= self.config.temperament_history_maxlen:
            trend = torch.tensor(list(self.state.temperament_history)).mean().item() - self.state.temperament_history[0]
            history_dream = abs(trend) > 0.3
        should_dream = swing_dream or lifecycle_dream or history_dream
        self.logger.write({
            "event": "dream_check",
            "swing_dream": swing_dream,
            "lifecycle_dream": lifecycle_dream,
            "history_dream": history_dream,
            "should_dream": should_dream,
            "timestamp": time.time(),
            "conversation_id": self.state.history.conversation_id if self.state and hasattr(self.state, "history") else str(uuid.uuid4())
        })
        return should_dream

    def _dream(self) -> None:
        """Execute a complete dream cycle with memory creation"""
        if not self.config.enable_dreaming:
            return

        # Initialize dream memory if needed
        if not hasattr(self, 'dream_memory'):
            self._init_dream_memory()

        # Get log entries for dreaming
        log_entries = self._get_dream_log_entries()
        if not log_entries:
            return

        # Select and process dream prompt
        dream_prompt = self._select_dream_prompt(log_entries)
        hidden_state = self._process_dream_prompt(dream_prompt)

        # Store the memory
        self._store_dream_memory(dream_prompt, hidden_state)

        # Log the dream event
        self._log_dream_event(dream_prompt)

    def _init_dream_memory(self) -> None:
        """Initialize the dream memory system"""
        self.dream_memory = DreamMemory(
            config=DreamMemoryConfig(
                max_memories=100,
                novelty_boost=self.config.dream_novelty_boost,
                base_weight=self.config.dream_memory_weight,
                decay_rate=self.config.dream_memory_decay,
                prune_threshold=self.config.dream_prune_threshold,
                noise_scale=self.config.dream_noise_scale,
                melancholy_noise=self.config.temp_melancholy_noise
            ),
            device=self.device
        )

    def _get_dream_log_entries(self) -> List[dict]:
        """Retrieve log entries for dreaming"""
        log_entries = self.logger.read() if hasattr(self.logger, "read") else []
        if not log_entries:
            self.logger({
                "event": "dream_skipped",
                "reason": "empty_logs",
                "timestamp": time.time()
            })
        return log_entries

    def _select_dream_prompt(self, log_entries: List[dict]) -> str:
        """Select prompt for dreaming based on configuration"""
        if not self.config.enable_prompt_driven_dreams:
            return random.choice(log_entries)["prompt"]

        # Get reference prompt (most recent interaction)
        reference_prompt = self._get_reference_prompt(log_entries)

        # Calculate weighted prompt selection
        return self._select_by_similarity(reference_prompt, log_entries)

    def _get_reference_prompt(self, log_entries: List[dict]) -> str:
        """Get reference prompt for similarity comparison"""
        if hasattr(self.state, "history") and self.state.history.messages:
            return self.state.history.messages[-1]["prompt"]
        return random.choice(log_entries)["prompt"]

    def _select_by_similarity(self, reference: str, log_entries: List[dict]) -> str:
        """Select prompt with temperature-adjusted sampling"""
        if not log_entries:
            raise ValueError("No log entries available for dreaming")
        
        # Calculate all similarities first
        similarities = []
        valid_entries = []
        
        for entry in log_entries:
            if "prompt" not in entry:
                continue
                
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
            valid_entries.append(entry)
        
        if not similarities:
            raise ValueError("No valid prompts found in log entries")
        
        # Combine similarity with recency using temperature
        recency_weights = [i/len(valid_entries) for i in range(len(valid_entries))]
        combined = [
            (self.config.dream_prompt_weight * sim) + 
            ((1 - self.config.dream_prompt_weight) * recency)
            for sim, recency in zip(similarities, recency_weights)
        ]
        
        # Apply temperature scaling (0.5 makes distribution more uniform)
        temperature = 0.5
        scaled_weights = torch.softmax(
            torch.tensor(combined) / temperature, 
            dim=0
        ).tolist()
        
        # Select based on final weights
        return random.choices(
            [e["prompt"] for e in valid_entries],
            weights=scaled_weights,
            k=1
        )[0]

    def _process_dream_prompt(self, prompt: str) -> torch.Tensor:
        """Process prompt through model to get hidden state"""
        with torch.no_grad():
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.config.max_seq_length,
                truncation=True
            ).to(self.device)
            return self.model(**inputs).hidden_states[-1].mean(dim=1)
        
    def _check_novelty(self, prompt: str, hidden_state: torch.Tensor) -> Tuple[bool, float]:
        """
        Enhanced novelty check with semantic similarity analysis
        Returns:
            Tuple of (is_novel: bool, similarity_score: float)
        """
        # 1. Basic novelty check
        basic_novel = prompt not in getattr(self.state, "seen_prompts", set())
        if basic_novel or not hasattr(self, 'dream_memory') or len(self.dream_memory) == 0:
            return (basic_novel, 0.0)

        # 2. Semantic similarity check with existing memories
        similarities = []
        with self.dream_memory.lock:
            for memory in self.dream_memory.memory:
                # Calculate cosine similarity between hidden states
                sim = F.cosine_similarity(
                    hidden_state.flatten(),
                    memory['vector'].to(self.device).flatten(),
                    dim=0
                ).item()
                similarities.append(sim)

        max_similarity = max(similarities) if similarities else 0.0
        is_semantically_novel = max_similarity < 0.7  # Threshold for considering novel

        return (is_semantically_novel, max_similarity)    

    def _store_dream_memory(self, prompt: str, hidden_state: torch.Tensor) -> None:
        """Store processed prompt in dream memory"""
        is_novel = prompt not in getattr(self.state, "seen_prompts", set())
        self.dream_memory.add_memory(
            prompt=prompt,
            hidden_state=hidden_state,
            is_novel=is_novel,
            temperament=getattr(self.state, "temperament_score", 0.0)
        )

    def _log_dream_event(self, prompt: str) -> None:
        """Log dream event details"""
        memories = self.dream_memory.get_memories()
        self.logger({
            "event": "dream_cycle",
            "prompt": prompt,
            "memory_count": len(self.dream_memory),
            "top_weight": memories[0]["weight"] if memories else 0,
            "timestamp": time.time(),
            "conversation_id": str(uuid.uuid4())  # Generate new ID if state not available
        })

    def sleep_train(self, log_entries: List[dict]):
        """Perform sleep training on log_entries."""
        if not self.config.enable_sleep_training or not self._should_gestate(log_entries):
            return
        print("\n--- Sleep Training Initiated ---")
        batch = [
            {"prompt": entry["prompt"], "completion": entry["response"]}
            for entry in log_entries if "prompt" in entry and "response" in entry
        ]
        if not batch:
            print("No valid training data in logs.")
            return

        if self.config.enable_dreaming and self._should_dream():
            self._dream()

        original_epochs = self.config.max_epochs
        self.config.max_epochs = 1
        self.train(
            train_data=batch,
            valid_data=None,
            scaffold_provider=None
        )
        self.config.max_epochs = original_epochs

        self.data_exposure += sum(
            len(entry["prompt"]) + len(entry["response"])
            for entry in batch
        )
        self._reset_sleep_state()
        print("Sleep Training Complete")

    def _reset_sleep_state(self):
        """Reset sleep-related state."""
        self.sleep_progress = 0
        self.sleep_batch = []
        self.sleep_total_loss = 0.0
        self.sleep_steps = 0

    def cleanup(self):
        """Clean up trainer resources."""
        try:
            self._reset_gestation_state()
            self._reset_sleep_state()
            self.optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger({
                "event": "trainer_cleanup",
                "timestamp": time.time(),
                "details": "Trainer resources cleared"
            })
            print("Trainer cleanup completed")
        except Exception as e:
            self.logger({
                "event": "trainer_cleanup_failed",
                "error": str(e),
                "timestamp": time.time()
            })
            print(f"Trainer cleanup failed: {e}")

    def train(
        self,
        train_data: Union[List[dict], 'DataLoader'],
        valid_data: Optional[Union[List[dict], 'DataLoader']] = None,
        scaffold_provider: Optional[Callable] = None,
        resume_checkpoint: Optional[str] = None
    ):
        """Run training loop over epochs."""
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)

        if getattr(self.state, 'dry_run', False) and self.state.dry_run_params.get('skip_training', False):
            print("\n=== DRY RUN TRAINING ===")
            dry_batch = train_data[:self.state.dry_run_params.get('max_samples', self.config.batch_size)]
            if isinstance(dry_batch, (list, tuple)):
                dry_batch = collate_batch(dry_batch, self.tokenizer.pad_token_id, self.config.max_seq_length, self.tokenizer)
            dry_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in dry_batch.items()}
            scaffold_context = scaffold_provider(dry_batch) if scaffold_provider else self.scaffold_context
            if scaffold_context is not None:
                scaffold_context = scaffold_context.to(self.device)
            loss, _ = self.train_step(
                dry_batch,
                scaffold_context=scaffold_context,
                dry_run=True,
                memory_check=self.memory_check
            )
            print(f"Dry run training complete: Loss = {loss}")
            return

        if isinstance(train_data, (list, tuple)):
            train_iter = [train_data[i:i + self.config.batch_size] for i in range(0, len(train_data), self.config.batch_size)]
        else:
            train_iter = train_data

        if valid_data and isinstance(valid_data, (list, tuple)):
            valid_iter = [valid_data[i:i + self.config.batch_size] for i in range(0, len(valid_data), self.config.batch_size)]
        else:
            valid_iter = valid_data

        for epoch in range(self.config.max_epochs):
            self.model.train()
            epoch_loss = 0.0
            steps_in_epoch = 0

            for batch in train_iter:
                if isinstance(batch, (list, tuple)):
                    batch = collate_batch(batch, self.tokenizer.pad_token_id, self.config.max_seq_length, self.tokenizer)
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                scaffold_context = scaffold_provider(batch) if scaffold_provider else self.scaffold_context
                if scaffold_context is not None:
                    scaffold_context = scaffold_context.to(self.device)

                loss, confidence = self.train_step(
                    batch,
                    scaffold_context=scaffold_context,
                    grad_clip=True,
                    loss_weight_fn=self.loss_weight_fn,
                    dry_run=False,
                    memory_check=self.memory_check
                )

                if loss is not None:
                    epoch_loss += loss
                    steps_in_epoch += 1
                    self.update_exposure(batch["prompt"], self.state.temperament_score if self.state else 0.0)
                    self.logger({
                        "event": "train_step",
                        "epoch": epoch + 1,
                        "step": self.global_step,
                        "loss": loss,
                        "confidence": confidence,
                        "data_exposure": self.data_exposure,
                        "timestamp": time.time()
                    })

                if valid_iter and self.config.validate_every_n_steps and self.global_step % self.config.validate_every_n_steps == 0:
                    valid_loss, metrics = self.validate(valid_iter, scaffold_provider)
                    self.logger({
                        "event": "validation",
                        "epoch": epoch + 1,
                        "step": self.global_step,
                        "loss": valid_loss,
                        "metrics": metrics,
                        "timestamp": time.time()
                    })
                    if valid_loss < self.best_valid_loss:
                        self.best_valid_loss = valid_loss
                        self.patience = 0
                        self.save_checkpoint(self.global_step, suffix="best")
                    else:
                        self.patience += 1
                    if self.should_stop():
                        self.logger({
                            "event": "early_stopping",
                            "epoch": epoch + 1,
                            "step": self.global_step,
                            "valid_loss": valid_loss,
                            "timestamp": time.time()
                        })
                        print(f"Early stopping at epoch {epoch + 1}, step {self.global_step}")
                        self.on_training_complete(epoch + 1, valid_loss, self.data_exposure)
                        return

            # Log epoch end
            avg_epoch_loss = epoch_loss / steps_in_epoch if steps_in_epoch > 0 else 0.0
            self.logger({
                "event": "epoch_end",
                "epoch": epoch + 1,
                "avg_loss": avg_epoch_loss,
                "data_exposure": self.data_exposure,
                "timestamp": time.time()
            })
            print(f"Epoch {epoch + 1}/{self.config.max_epochs}: Avg Loss = {avg_epoch_loss:.4f}")

        # Trigger callback after training completes
        self.on_training_complete(self.config.max_epochs, avg_epoch_loss, self.data_exposure)

    def get_memory_stats(self) -> dict:
        """Get detailed statistics about dream memory usage"""
        base_stats = {
            'status': 'dream_memory_not_initialized',
            'count': 0,
            'average_weight': 0.0,
            'max_weight': 0.0,
            'min_weight': 0.0,
            'oldest': None,
            'newest': None,
            'config': {
                'max_memories': 0,
                'decay_rate': 0.0,
                'prune_threshold': 0.0
            }
        }
        
        if not hasattr(self, 'dream_memory'):
            return base_stats

        with self.dream_memory.lock:
            memories = list(self.dream_memory.memory)
            if not memories:
                base_stats.update({
                    'status': 'empty',
                    'config': {
                        'max_memories': self.dream_memory.memory.maxlen,
                        'decay_rate': self.dream_memory.config.decay_rate,
                        'prune_threshold': self.dream_memory.config.prune_threshold
                    }
                })
                return base_stats

            weights = [m['weight'] for m in memories]
            timestamps = [m['timestamp'] for m in memories]
            
            base_stats.update({
                'status': 'active',
                'count': len(memories),
                'average_weight': sum(weights) / len(weights),
                'max_weight': max(weights),
                'min_weight': min(weights),
                'oldest': min(timestamps),
                'newest': max(timestamps),
                'config': {
                    'max_memories': self.dream_memory.memory.maxlen,
                    'decay_rate': self.dream_memory.config.decay_rate,
                    'prune_threshold': self.dream_memory.config.prune_threshold
                }
            })
            return base_stats

    def __post_init__(self):
        if self.metrics_to_track is None:
            self.metrics_to_track = ["loss", "accuracy", "confidence"]
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

def collate_batch(batch: List[dict], pad_token_id: int, max_seq_length: int, tokenizer) -> dict:
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
