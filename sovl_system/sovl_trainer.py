from dataclasses import dataclass
from typing import List, Optional, Callable, Union, Tuple, Dict
import torch
import torch.nn.functional as F
import time
import uuid
import math
import os
import threading
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from sovl_utils import get_life_curve_weight, has_repetition, update_exposure, calculate_confidence
from sovl_dream_memory import DreamMemory, DreamMemoryConfig

@dataclass
class CoreTrainingConfig:
    """Core training parameters."""
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
    accumulation_steps: int = 4
    exposure_gain_eager: int = 3
    exposure_gain_default: int = 2
    sigmoid_scale: float = 0.5
    sigmoid_shift: float = 5.0

    def __post_init__(self):
        if self.metrics_to_track is None:
            self.metrics_to_track = ["loss", "accuracy", "confidence"]
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.grad_accum_steps >= 1, "Gradient accumulation steps must be at least 1"
        assert self.max_grad_norm > 0, "Max gradient norm must be positive"
        assert self.scheduler_type in ["linear", "cosine", "constant"], "Invalid scheduler type"
        assert self.lifecycle_curve in ["sigmoid_linear", "exponential"], "Invalid lifecycle curve"

@dataclass
class CuriosityConfig:
    """Curiosity-related parameters."""
    weight_ignorance: float = 0.0
    weight_novelty: float = 1.0
    pressure_threshold: float = 0.9
    pressure_drop: float = 0.5
    novelty_threshold_spontaneous: float = 0.5
    novelty_threshold_response: float = 1.0
    silence_threshold: float = 5.0
    question_cooldown: float = 30.0
    queue_maxlen: int = 5
    max_new_tokens: int = 12
    base_temperature: float = 0.5
    temperament_influence: float = 0.6
    top_k: int = 10

    def __post_init__(self):
        assert 0.5 <= self.novelty_threshold_spontaneous <= 1.0, "Spontaneous threshold must be in [0.5, 1.0]"
        assert 0.5 <= self.novelty_threshold_response <= 1.0, "Response threshold must be in [0.5, 1.0]"
        assert 0.5 <= self.pressure_threshold <= 0.9, "Pressure threshold must be in [0.5, 0.9]"
        assert 0.1 <= self.pressure_drop <= 0.5, "Pressure drop must be in [0.1, 0.5]"
        assert 5.0 <= self.silence_threshold <= 60.0, "Silence threshold must be in [5.0, 60.0]"
        assert 30.0 <= self.question_cooldown <= 120.0, "Question cooldown must be in [30.0, 120.0]"
        assert 5 <= self.queue_maxlen <= 20, "Queue maxlen must be in [5, 20]"
        assert 0.0 <= self.weight_ignorance <= 1.0, "Ignorance weight must be in [0.0, 1.0]"
        assert 0.0 <= self.weight_novelty <= 1.0, "Novelty weight must be in [0.0, 1.0]"
        assert 5 <= self.max_new_tokens <= 12, "Max new tokens must be in [5, 12]"
        assert 0.5 <= self.base_temperature <= 1.5, "Base temperature must be in [0.5, 1.5]"
        assert 0.1 <= self.temperament_influence <= 0.6, "Temperament influence must be in [0.1, 0.6]"
        assert 10 <= self.top_k <= 50, "Top k must be in [10, 50]"

@dataclass
class TrainingConfig:
    """Aggregated training configuration."""
    core: CoreTrainingConfig
    dream: DreamMemoryConfig
    curiosity: CuriosityConfig

class DataProcessor:
    """Handles data collation and preprocessing for training."""
    def __init__(self, tokenizer, max_seq_length: int, pad_token_id: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self._prompt_cache = {}

    def collate_batch(self, batch: List[dict]) -> dict:
        """Collate batch of prompt-completion pairs into tensors.

        Args:
            batch: List of dictionaries with 'prompt' and 'completion' keys.

        Returns:
            Dictionary with input_ids, attention_mask, labels, and prompts.
        """
        prompts = [item["prompt"] for item in batch]
        completions = [item["completion"] for item in batch]
        full_texts = [p + c for p, c in zip(prompts, completions)]

        if tuple(full_texts) in self._prompt_cache:
            encodings = self._prompt_cache[tuple(full_texts)]
        else:
            encodings = self.tokenizer(
                full_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_length
            )
            self._prompt_cache[tuple(full_texts)] = encodings

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100

        prompt_encodings = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length
        )
        prompt_mask = prompt_encodings["attention_mask"]
        labels = torch.where(prompt_mask.bool(), -100, input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt": prompts
        }

class GestationManager:
    """Manages gestation and sleep training processes."""
    def __init__(self, trainer: 'SOVLTrainer', config: CoreTrainingConfig, logger):
        self.trainer = trainer
        self.config = config
        self.logger = logger
        self.is_gestating = False
        self.gestation_progress = 0
        self.gestation_batch = []
        self.gestation_total_loss = 0.0
        self.gestation_steps = 0
        self.sleep_progress = 0
        self.sleep_batch = []
        self.sleep_total_loss = 0.0
        self.sleep_steps = 0

    def gestate(self, log_entries: List[dict], resume: bool = False) -> bool:
        """Perform gestation training on log entries.

        Args:
            log_entries: List of log entries with prompt and response.
            resume: Whether to resume an ongoing gestation.

        Returns:
            bool: True if gestation is ongoing, False if complete or skipped.
        """
        if not self.config.enable_gestation:
            return False

        if not log_entries:
            self.logger({
                "event": "gestation_skipped",
                "reason": "empty_logs",
                "timestamp": time.time()
            })
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
            if self.config.enable_dreaming and self.trainer._should_dream():
                self.trainer._dream()

            self.trainer.data_exposure += sum(
                len(entry["prompt"]) + len(entry["response"])
                for entry in log_entries
                if "prompt" in entry and "response" in entry
            )

        if self.gestation_progress < len(self.gestation_batch):
            batch = [self.gestation_batch[self.gestation_progress]]
            formatted_batch = self.trainer.data_processor.collate_batch(batch)
            metrics = self.trainer.train_step(formatted_batch, scaffold_context=self.trainer.scaffold_context)

            self.gestation_total_loss += metrics["loss"]
            self.gestation_steps += 1
            self.gestation_progress += 1
            if self.gestation_steps % 5 == 0 and self.gestation_steps > 0:
                print(f"Gestation progress: {self.gestation_progress}/{len(self.gestation_batch)}, loss: {self.gestation_total_loss / self.gestation_steps:.4f}")
            return True

        avg_loss = self.gestation_total_loss / self.gestation_steps if self.gestation_steps > 0 else 0
        print(f"\nGestation complete: {len(self.gestation_batch)}/{len(self.gestation_batch)}, loss: {avg_loss:.4f}")
        self.trainer.on_gestation_complete(len(self.gestation_batch), avg_loss)
        self._reset_gestation_state()
        return False

    def sleep_train(self, log_entries: List[dict]) -> None:
        """Perform sleep training on log entries.

        Args:
            log_entries: List of log entries with prompt and response.
        """
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

        if self.config.enable_dreaming and self.trainer._should_dream():
            self.trainer._dream()

        original_epochs = self.config.max_epochs
        self.config.max_epochs = 1
        self.trainer.train(
            train_data=batch,
            valid_data=None,
            scaffold_provider=None
        )
        self.config.max_epochs = original_epochs

        self.trainer.data_exposure += sum(
            len(entry["prompt"]) + len(entry["response"])
            for entry in batch
        )
        self.trainer.on_sleep_train_complete(len(batch), self.trainer.data_exposure)
        self._reset_sleep_state()
        print("Sleep Training Complete")

    def _should_gestate(self, log_entries: List[dict]) -> bool:
        """Determine if gestation should proceed."""
        if len(log_entries) < self.config.sleep_log_min:
            self.logger({
                "event": "gestation_check",
                "log_size": len(log_entries),
                "min_required": self.config.sleep_log_min,
                "result": False,
                "timestamp": time.time()
            })
            return False
        avg_confidence = self.trainer.state.sleep_confidence_sum / self.trainer.state.sleep_confidence_count if self.trainer.state.sleep_confidence_count > 0 else 0.5
        should_gestate = (len(log_entries) >= self.config.sleep_log_min) and (self.trainer.state.sleep_confidence_count == 0 or avg_confidence > self.config.sleep_conf_threshold)
        self.logger({
            "event": "gestation_check",
            "confidence": avg_confidence,
            "threshold": self.config.sleep_conf_threshold,
            "log_size": len(log_entries),
            "min_required": self.config.sleep_log_min,
            "result": should_gestate,
            "timestamp": time.time()
        })
        return should_gestate

    def _reset_gestation_state(self):
        """Reset gestation-related state."""
        self.is_gestating = False
        self.gestation_progress = 0
        self.gestation_batch = []
        self.gestation_total_loss = 0.0
        self.gestation_steps = 0

    def _reset_sleep_state(self):
        """Reset sleep-related state."""
        self.sleep_progress = 0
        self.sleep_batch = []
        self.sleep_total_loss = 0.0
        self.sleep_steps = 0

class DreamManager:
    """Manages dreaming functionality for training."""
    def __init__(self, trainer: 'SOVLTrainer', config: DreamMemoryConfig, logger, device: torch.device):
        self.trainer = trainer
        self.dream_memory = DreamMemory(config, device)
        self.logger = logger

    def dream(self) -> None:
        """Execute a complete dream cycle with memory creation."""
        log_entries = self._get_dream_log_entries()
        if not log_entries:
            return

        dream_prompt = self._select_dream_prompt(log_entries)
        hidden_state = self._process_dream_prompt(dream_prompt)
        self._store_dream_memory(dream_prompt, hidden_state)
        self._log_dream_event(dream_prompt)

    def _get_dream_log_entries(self) -> List[dict]:
        """Retrieve log entries for dreaming."""
        log_entries = self.logger.read() if hasattr(self.logger, "read") else []
        if not log_entries:
            self.logger({
                "event": "dream_skipped",
                "reason": "empty_logs",
                "timestamp": time.time()
            })
        return log_entries

    def _select_dream_prompt(self, log_entries: List[dict]) -> str:
        """Select prompt for dreaming based on configuration."""
        if not self.trainer.config.core.enable_dreaming:
            return random.choice(log_entries)["prompt"]

        reference_prompt = self._get_reference_prompt(log_entries)
        return self._select_by_similarity(reference_prompt, log_entries)

    def _get_reference_prompt(self, log_entries: List[dict]) -> str:
        """Get reference prompt for similarity comparison."""
        if hasattr(self.trainer.state, "history") and self.trainer.state.history.messages:
            return self.trainer.state.history.messages[-1]["prompt"]
        return random.choice(log_entries)["prompt"]

    def _select_by_similarity(self, reference: str, log_entries: List[dict]) -> str:
        """Select prompt with temperature-adjusted sampling."""
        if not log_entries:
            raise ValueError("No log entries available for dreaming")

        valid_entries = [e for e in log_entries if "prompt" in e]
        if not valid_entries:
            raise ValueError("No valid prompts found in log entries")

        batch_texts = [reference] + [e["prompt"] for e in valid_entries]
        inputs = self.trainer.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=self.trainer.config.core.max_seq_length,
            return_tensors="pt"
        ).to(self.trainer.device)

        with torch.no_grad():
            hidden = self.trainer.model(**inputs).hidden_states[-1].mean(dim=1)
            similarities = F.cosine_similarity(hidden[0:1], hidden[1:], dim=-1).cpu().tolist()

        recency_weights = [i / len(valid_entries) for i in range(len(valid_entries))]
        combined = [
            (self.trainer.config.dream.prompt_weight * sim) + ((1 - self.trainer.config.dream.prompt_weight) * recency)
            for sim, recency in zip(similarities, recency_weights)
        ]

        temperature = 0.5
        scaled_weights = torch.softmax(torch.tensor(combined) / temperature, dim=0).tolist()
        return random.choices([e["prompt"] for e in valid_entries], weights=scaled_weights, k=1)[0]

    def _process_dream_prompt(self, prompt: str) -> torch.Tensor:
        """Process prompt through model to get hidden state."""
        with torch.no_grad():
            inputs = self.trainer.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.trainer.config.core.max_seq_length,
                truncation=True
            ).to(self.trainer.device)
            return self.trainer.model(**inputs).hidden_states[-1].mean(dim=1)

    def _check_novelty(self, prompt: str, hidden_state: torch.Tensor) -> Tuple[bool, float]:
        """Check novelty with semantic similarity analysis.

        Args:
            prompt: Input prompt to check.
            hidden_state: Hidden state of the prompt.

        Returns:
            Tuple of (is_novel: bool, similarity_score: float).
        """
        basic_novel = prompt not in getattr(self.trainer.state, "seen_prompts", set())
        if basic_novel or len(self.dream_memory) == 0:
            return basic_novel, 0.0

        similarities = []
        with self.dream_memory.lock:
            for memory in self.dream_memory.memory:
                sim = F.cosine_similarity(
                    hidden_state.flatten(),
                    memory['vector'].to(self.trainer.device).flatten(),
                    dim=0
                ).item()
                similarities.append(sim)

        max_similarity = max(similarities) if similarities else 0.0
        is_semantically_novel = max_similarity < 0.7
        return is_semantically_novel, max_similarity

    def _store_dream_memory(self, prompt: str, hidden_state: torch.Tensor) -> None:
        """Store processed prompt in dream memory."""
        is_novel, _ = self._check_novelty(prompt, hidden_state)
        self.dream_memory.add_memory(
            prompt=prompt,
            hidden_state=hidden_state,
            is_novel=is_novel,
            temperament=getattr(self.trainer.state, "temperament_score", 0.0)
        )

    def _log_dream_event(self, prompt: str) -> None:
        """Log dream event details."""
        memories = self.dream_memory.get_memories()
        self.logger({
            "event": "dream_cycle",
            "prompt": prompt,
            "memory_count": len(self.dream_memory),
            "top_weight": memories[0]["weight"] if memories else 0,
            "timestamp": time.time(),
            "conversation_id": self.trainer.state.history.conversation_id if self.trainer.state and hasattr(self.trainer.state, "history") else str(uuid.uuid4())
        })

class SOVLTrainer:
    """Manages model training with lifecycle weighting and scaffold integration.

    Args:
        model: PyTorch model to train.
        config: Training configuration.
        device: Device to run training on.
        loss_fn: Loss function for training.
        logger: Logger instance for events.
        memory_lock: Threading lock for memory operations.
        tokenizer: Tokenizer for data processing.
        state: System state object.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainingConfig,
        device: torch.device,
        loss_fn: Callable,
        logger,
        memory_lock: Optional[threading.Lock] = None,
        tokenizer=None,
        state=None
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.loss_fn = loss_fn
        self.logger = logger
        self.memory_lock = memory_lock or threading.Lock()
        self.tokenizer = tokenizer
        self.state = state
        self.global_step = 0
        self.best_valid_loss = float("inf")
        self.patience = 0
        self.data_exposure = 0
        self.lora_capacity = sum(p.numel() for p in model.parameters() if p.requires_grad) * config.core.lifecycle_capacity_factor
        self.scaffold_context = None
        self.data_processor = DataProcessor(tokenizer, config.core.max_seq_length, tokenizer.pad_token_id if tokenizer else 0)

        self.gestation_manager = GestationManager(self, config.core, logger)
        self.dream_manager = DreamManager(self, config.dream, logger, device)

        self.callbacks = {
            "on_training_complete": [],
            "on_gestation_complete": [],
            "on_dream_complete": [],
            "on_sleep_train_complete": []
        }

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.core.learning_rate,
            weight_decay=config.core.weight_decay
        )

        if config.core.scheduler_type == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=int(config.core.warmup_steps or config.core.warmup_ratio * config.core.total_steps),
                num_training_steps=config.core.total_steps
            )
        elif config.core.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.core.total_steps - int(config.core.warmup_steps or config.core.warmup_ratio * config.core.total_steps),
                eta_min=config.core.cosine_min_lr
            )
        else:
            self.scheduler = None

        if config.core.dropout_rate > 0:
            for module in self.model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = config.core.dropout_rate

    def register_callback(self, event: str, callback: Callable):
        """Register a callback for a specific event.

        Args:
            event: Event name (e.g., 'on_training_complete').
            callback: Function to call when the event occurs.
        """
        if event in self.callbacks:
            self.callbacks[event].append(callback)
        else:
            self.logger({
                "event": "callback_registration_failed",
                "error": f"Unknown event: {event}",
                "timestamp": time.time()
            })

    def on_training_complete(self, epoch: int, avg_loss: float, data_exposure: float):
        """Notify completion of training cycle."""
        result = {"epoch": epoch, "avg_loss": avg_loss, "data_exposure": data_exposure}
        for callback in self.callbacks["on_training_complete"]:
            callback(epoch, avg_loss, data_exposure)
        return result

    def on_gestation_complete(self, batch_size: int, avg_loss: float):
        """Notify completion of gestation."""
        result = {"batch_size": batch_size, "avg_loss": avg_loss}
        for callback in self.callbacks["on_gestation_complete"]:
            callback(batch_size, avg_loss)
        return result

    def on_dream_complete(self, prompt: str, novelty: bool, memory_count: int):
        """Notify completion of dreaming."""
        result = {"prompt": prompt, "novelty": novelty, "memory_count": memory_count}
        for callback in self.callbacks["on_dream_complete"]:
            callback(prompt, novelty, memory_count)
        return result

    def on_sleep_train_complete(self, batch_size: int, data_exposure: float):
        """Notify completion of sleep training."""
        result = {"batch_size": batch_size, "data_exposure": data_exposure}
        for callback in self.callbacks["on_sleep_train_complete"]:
            callback(batch_size, data_exposure)
        return result

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
        batch: Dict[str, torch.Tensor],
        scaffold_context: Optional[Dict[str, torch.Tensor]] = None,
        accumulation_steps: int = 1
    ) -> Dict[str, float]:
        """Execute a single training step with optional scaffold context.

        Args:
            batch: Dictionary containing input tensors.
            scaffold_context: Optional scaffold context for the training step.
            accumulation_steps: Number of gradient accumulation steps.

        Returns:
            Dictionary containing loss and other metrics.
        """
        try:
            self.model.train()
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            if scaffold_context:
                scaffold_context = {k: v.to(self.device) for k, v in scaffold_context.items()}

            with torch.cuda.amp.autocast(enabled=self.config.core.use_amp):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    scaffold_context=scaffold_context
                )
                loss = self.loss_fn(outputs.logits, batch["labels"])

                if self.config.core.enable_lifecycle_weighting:
                    weight = get_life_curve_weight(
                        self.data_exposure,
                        self.lora_capacity,
                        self.config.core.sigmoid_scale,
                        self.config.core.sigmoid_shift,
                        self.config.core.lifecycle_curve
                    )
                    loss = loss * weight

            loss = loss / accumulation_steps
            loss.backward()

            if (self.global_step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.core.max_grad_norm)
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

            self.global_step += 1
            metrics = {
                "loss": loss.item() * accumulation_steps,
                "learning_rate": self.optimizer.param_groups[0]["lr"]
            }

            if "confidence" in self.config.core.metrics_to_track:
                confidence = calculate_confidence(outputs.logits, batch["labels"])
                metrics["confidence"] = confidence
                if self.state:
                    with self.memory_lock:
                        self.state.sleep_confidence_sum += confidence
                        self.state.sleep_confidence_count += 1
                        self.state.confidence_history.append(confidence)

            self.logger({
                "event": "train_step",
                "step": self.global_step,
                "metrics": metrics,
                "timestamp": time.time()
            })

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return metrics

        except torch.cuda.OutOfMemoryError as oom:
            self.logger({
                "event": "train_step_failed",
                "error": "CUDA out of memory",
                "timestamp": time.time(),
                "memory_stats": {
                    "allocated": torch.cuda.memory_allocated(),
                    "reserved": torch.cuda.memory_reserved()
                }
            })
            torch.cuda.empty_cache()
            raise
        except Exception as e:
            self.logger({
                "event": "train_step_failed",
                "error": str(e),
                "timestamp": time.time()
            })
            raise

    def validate(self, data: Union[List[dict], 'DataLoader'], scaffold_provider: Optional[Callable] = None) -> Tuple[float, dict]:
        """Validate model on provided data, returning loss and metrics.

        Args:
            data: Validation data (list or DataLoader).
            scaffold_provider: Optional function to provide scaffold context.

        Returns:
            Tuple of average loss and metrics dictionary.
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        batches = 0
        metrics = {metric: 0.0 for metric in self.config.core.metrics_to_track}

        if isinstance(data, (list, tuple)):
            data_iter = [data[i:i + self.config.core.batch_size] for i in range(0, len(data), self.config.core.batch_size)]
        else:
            data_iter = data

        with torch.no_grad():
            for batch in data_iter:
                if isinstance(batch, (list, tuple)):
                    batch = self.data_processor.collate_batch(batch)
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                scaffold_context = scaffold_provider(batch) if scaffold_provider else self.scaffold_context
                if scaffold_context is not None:
                    scaffold_context = scaffold_context.to(self.device)

                try:
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.config.core.use_amp else torch.bfloat16):
                        outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                        loss = self.loss_fn(outputs.logits, batch["labels"])

                    total_loss += loss.item()
                    batches += 1

                    if "accuracy" in self.config.core.metrics_to_track:
                        preds = outputs.logits.argmax(dim=-1)
                        mask = batch["labels"] != -100
                        correct = (preds[mask] == batch["labels"][mask]).sum().item()
                        total_correct += correct
                        total_tokens += mask.sum().item()
                        metrics["accuracy"] = total_correct / total_tokens if total_tokens > 0 else 0.0
                    if "perplexity" in self.config.core.metrics_to_track:
                        perplexity = torch.exp(loss).item()
                        metrics["perplexity"] = perplexity
                    if "confidence" in self.config.core.metrics_to_track:
                        metrics["confidence"] = calculate_confidence(outputs.logits, batch["labels"])

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except torch.cuda.OutOfMemoryError as oom:
                    self.logger({
                        "event": "validate_failed",
                        "error": "CUDA out of memory",
                        "timestamp": time.time(),
                        "memory_stats": {
                            "allocated": torch.cuda.memory_allocated(),
                            "reserved": torch.cuda.memory_reserved()
                        }
                    })
                    torch.cuda.empty_cache()
                    raise
                except Exception as e:
                    self.logger({
                        "event": "validate_failed",
                        "error": str(e),
                        "timestamp": time.time()
                    })
                    raise

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
        """Save model checkpoint.

        Args:
            step: Current training step.
            suffix: Optional suffix for checkpoint filename.
        """
        checkpoint_dir = self.config.core.checkpoint_path
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_name = f"checkpoint_{step}{f'_{suffix}' if suffix else ''}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

        memory_data = self.dream_manager.dream_memory.get_state() if self.config.core.enable_dreaming else None

        state_dict = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.global_step,
            "best_valid_loss": self.best_valid_loss,
            "patience": self.patience,
            "data_exposure": self.data_exposure,
            "dream_memory": memory_data
        }

        torch.save(state_dict, checkpoint_path)
        self.logger({
            "event": "checkpoint_saved",
            "path": checkpoint_path,
            "step": step,
            "memory_count": len(self.dream_manager.dream_memory) if self.config.core.enable_dreaming else 0,
            "timestamp": time.time()
        })
        print(f"Checkpoint saved: {checkpoint_path} (Memories: {len(self.dream_manager.dream_memory) if self.config.core.enable_dreaming else 0})")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        if not os.path.exists(checkpoint_path):
            self.logger({
                "event": "checkpoint_load_failed",
                "path": checkpoint_path,
                "error": "File not found",
                "timestamp": time.time()
            })
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict["model_state"])
        self.optimizer.load_state_dict(state_dict["optimizer_state"])
        if state_dict["scheduler_state"] and self.scheduler:
            self.scheduler.load_state_dict(state_dict["scheduler_state"])
        self.global_step = state_dict["global_step"]
        self.best_valid_loss = state_dict["best_valid_loss"]
        self.patience = state_dict["patience"]
        self.data_exposure = state_dict.get("data_exposure", 0)

        if 'dream_memory' in state_dict and state_dict['dream_memory'] is not None and self.config.core.enable_dreaming:
            self.dream_manager.dream_memory.load_state(state_dict['dream_memory'])

        self.logger({
            "event": "checkpoint_loaded",
            "path": checkpoint_path,
            "step": self.global_step,
            "memory_count": len(self.dream_manager.dream_memory) if self.config.core.enable_dreaming else 0,
            "timestamp": time.time()
        })
        print(f"Checkpoint loaded: {checkpoint_path} at step {self.global_step} (Memories: {len(self.dream_manager.dream_memory) if self.config.core.enable_dreaming else 0})")

    def should_stop(self) -> bool:
        """Check if training should stop based on early stopping criteria."""
        return self.patience >= self.config.core.max_patience

    def _should_dream(self) -> bool:
        """Determine if dreaming should occur."""
        if not self.state or not self.config.dream.temperament_on:
            return False
        swing_dream = (
            len(self.state.confidence_history) >= self.config.dream.confidence_history_maxlen and
            torch.var(torch.tensor(list(self.state.confidence_history))).item() > self.config.dream.swing_var
        )
        lifecycle_dream = (
            abs(self.state.temperament_score - self.state.last_temperament_score) > self.config.dream.lifecycle_delta
        )
        history_dream = False
        if len(self.state.temperament_history) >= self.config.dream.temperament_history_maxlen:
            trend = torch.tensor(list(self.state.temperament_history)).mean().item() - self.state.temperament_history[0]
            history_dream = abs(trend) > 0.3
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

    def _dream(self) -> None:
        """Execute a dream cycle."""
        if not self.config.core.enable_dreaming:
            return
        self.dream_manager.dream()

    def cleanup(self):
        """Clean up trainer resources."""
        try:
            self.gestation_manager._reset_gestation_state()
            self.gestation_manager._reset_sleep_state()
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
        """Run training loop over epochs.

        Args:
            train_data: Training data (list or DataLoader).
            valid_data: Optional validation data.
            scaffold_provider: Optional function to provide scaffold context.
            resume_checkpoint: Optional path to resume from checkpoint.
        """
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)

        if getattr(self.state, 'dry_run', False) and self.state.dry_run_params.get('skip_training', False):
            print("\n=== DRY RUN TRAINING ===")
            dry_batch = train_data[:self.state.dry_run_params.get('max_samples', self.config.core.batch_size)]
            if isinstance(dry_batch, (list, tuple)):
                dry_batch = self.data_processor.collate_batch(dry_batch)
            dry_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in dry_batch.items()}
            scaffold_context = scaffold_provider(dry_batch) if scaffold_provider else self.scaffold_context
            if scaffold_context is not None:
                scaffold_context = scaffold_context.to(self.device)
            metrics = self.train_step(dry_batch, scaffold_context=scaffold_context)
            print(f"Dry run training complete: Loss = {metrics['loss']}")
            return

        if isinstance(train_data, (list, tuple)):
            train_iter = [train_data[i:i + self.config.core.batch_size] for i in range(0, len(train_data), self.config.core.batch_size)]
        else:
            train_iter = train_data

        if valid_data and isinstance(valid_data, (list, tuple)):
            valid_iter = [valid_data[i:i + self.config.core.batch_size] for i in range(0, len(valid_data), self.config.core.batch_size)]
        else:
            valid_iter = valid_data

        for epoch in range(self.config.core.max_epochs):
            self.model.train()
            epoch_loss = 0.0
            steps_in_epoch = 0

            for batch in train_iter:
                if isinstance(batch, (list, tuple)):
                    batch = self.data_processor.collate_batch(batch)
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                scaffold_context = scaffold_provider(batch) if scaffold_provider else self.scaffold_context
                if scaffold_context is not None:
                    scaffold_context = scaffold_context.to(self.device)

                metrics = self.train_step(batch, scaffold_context=scaffold_context)

                epoch_loss += metrics["loss"]
                steps_in_epoch += 1
                update_exposure(self.state, batch["prompt"], self.state.temperament_score if self.state else 0.0, self.config.core)

                self.logger({
                    "event": "train_step",
                    "epoch": epoch + 1,
                    "step": self.global_step,
                    "loss": metrics["loss"],
                    "confidence": metrics.get("confidence", 0.0),
                    "data_exposure": self.data_exposure,
                    "timestamp": time.time()
                })

                if valid_iter and self.config.core.validate_every_n_steps and self.global_step % self.config.core.validate_every_n_steps == 0:
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

            avg_epoch_loss = epoch_loss / steps_in_epoch if steps_in_epoch > 0 else 0.0
            self.logger({
                "event": "epoch_end",
                "epoch": epoch + 1,
                "avg_loss": avg_epoch_loss,
                "data_exposure": self.data_exposure,
                "timestamp": time.time()
            })
            print(f"Epoch {epoch + 1}/{self.config.core.max_epochs}: Avg Loss = {avg_epoch_loss:.4f}")

            if self.global_step % self.config.core.checkpoint_interval == 0:
                self.save_checkpoint(self.global_step)

        self.on_training_complete(self.config.core.max_epochs, avg_epoch_loss, self.data_exposure)

    def run_training_cycle(
        self,
        train_data: List[Dict[str, torch.Tensor]],
        validation_data: Optional[List[Dict[str, torch.Tensor]]] = None,
        scaffold_provider: Optional[Callable] = None,
        max_epochs: Optional[int] = None,
        early_stopping_patience: int = 3
    ) -> Dict[str, Any]:
        """Run a complete training cycle with optional validation and early stopping.

        Args:
            train_data: List of training batches.
            validation_data: Optional list of validation batches.
            scaffold_provider: Optional function to provide scaffold context.
            max_epochs: Optional override for maximum number of epochs.
            early_stopping_patience: Number of epochs to wait before early stopping.

        Returns:
            Dictionary containing training results and metrics.
        """
        try:
            best_val_loss = float('inf')
            epochs_without_improvement = 0
            training_history = []

            num_epochs = max_epochs or self.config.core.max_epochs

            for epoch in range(num_epochs):
                epoch_metrics = {
                    "train_loss": 0.0,
                    "val_loss": float('inf'),
                    "learning_rate": self.optimizer.param_groups[0]["lr"]
                }

                self.model.train()
                for batch in train_data:
                    scaffold_context = scaffold_provider(batch) if scaffold_provider else None
                    step_metrics = self.train_step(
                        batch=batch,
                        scaffold_context=scaffold_context,
                        accumulation_steps=self.config.core.grad_accum_steps
                    )
                    epoch_metrics["train_loss"] += step_metrics["loss"]

                epoch_metrics["train_loss"] /= len(train_data)

                if validation_data:
                    val_loss, metrics = self.validate(validation_data, scaffold_provider)
                    epoch_metrics["val_loss"] = val_loss

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_without_improvement = 0
                        self.save_checkpoint(self.global_step, suffix="best")
                    else:
                        epochs_without_improvement += 1
                        if epochs_without_improvement >= early_stopping_patience:
                            self.logger({
                                "event": "early_stopping",
                                "epoch": epoch + 1,
                                "step": self.global_step,
                                "val_loss": val_loss,
                                "timestamp": time.time()
                            })
                            break

                self.logger({
                    "event": "epoch_end",
                    "epoch": epoch + 1,
                    "metrics": epoch_metrics,
                    "timestamp": time.time()
                })
                print(f"Epoch {epoch + 1}/{num_epochs}: "
                      f"train_loss={epoch_metrics['train_loss']:.4f}, "
                      f"val_loss={epoch_metrics['val_loss']:.4f}, "
                      f"lr={epoch_metrics['learning_rate']:.6f}")

                training_history.append(epoch_metrics)

                if self.global_step % self.config.core.checkpoint_interval == 0:
                    self.save_checkpoint(self.global_step)

            result = {
                "training_history": training_history,
                "best_val_loss": best_val_loss,
                "final_epoch": len(training_history),
                "early_stopped": epochs_without_improvement >= early_stopping_patience
            }
            self.on_training_complete(len(training_history), epoch_metrics["train_loss"], self.data_exposure)
            return result

        except Exception as e:
            self.logger({
                "event": "training_cycle_failed",
                "error": str(e),
                "timestamp": time.time()
            })
            raise

    def get_memory_stats(self) -> dict:
        """Get detailed statistics about dream memory usage."""
        return self.dream_manager.dream_memory.get_stats()
