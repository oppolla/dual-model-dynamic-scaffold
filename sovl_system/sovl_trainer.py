from dataclasses import dataclass
from typing import Optional, Callable, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
import threading
import os

@dataclass
class TrainingConfig:
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
    metrics_to_track: List[str] = None  # e.g., ["accuracy", "perplexity"]

    def __post_init__(self):
        if self.metrics_to_track is None:
            self.metrics_to_track = ["loss", "accuracy"]
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.grad_accum_steps >= 1, "Gradient accumulation steps must be at least 1"
        assert self.max_grad_norm > 0, "Max gradient norm must be positive"
        assert self.scheduler_type in ["linear", "cosine", "constant"], "Invalid scheduler type"

def collate_batch(batch: List[dict], pad_token_id: int, max_seq_length: int, tokenizer) -> dict:
    """Collate batch of prompt-completion pairs into tensors."""
    prompts = [item["prompt"] for item in batch]
    completions = [item["completion"] for item in batch]
    full_texts = [p + c for p, c in zip(prompts, completions)]

    # Tokenize full text
    encodings = tokenizer(
        full_texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_seq_length
    )
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    # Create labels with prompt masking
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
        tokenizer: Optional[Callable] = None
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.loss_fn = loss_fn
        self.logger = logger or (lambda x: None)
        self.memory_lock = memory_lock or threading.Lock()
        self.tokenizer = tokenizer
        self.global_step = 0
        self.best_valid_loss = float("inf")
        self.patience = 0
        self.loss_weight_fn = None
        self.memory_check = None

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
        else:  # constant
            self.scheduler = None

        # Apply dropout if specified
        if config.dropout_rate > 0:
            for module in self.model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = config.dropout_rate

    def train_step(
        self,
        batch: dict,
        scaffold_context: Optional[torch.Tensor] = None,
        grad_clip: bool = True,
        loss_weight_fn: Optional[Callable] = None,
        dry_run: bool = False,
        memory_check: Optional[Callable] = None
    ) -> tuple[Optional[float], Optional[float]]:
        """Execute a single training step."""
        if memory_check:
            memory_check()

        if dry_run:
            self.model.eval()
            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.config.use_amp else torch.bfloat16):
                    outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                    loss = self.loss_fn(outputs.logits, batch["labels"])
            return loss.item(), None

        self.model.train()
        with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.config.use_amp else torch.bfloat16):
            outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = self.loss_fn(outputs.logits, batch["labels"])
            if loss_weight_fn:
                weight = loss_weight_fn(batch)
                loss *= weight

        if torch.isnan(loss) or torch.isinf(loss):
            self.logger({"event": "invalid_loss", "loss": str(loss.item()), "timestamp": time.time()})
            return None, None

        scaled_loss = loss / self.config.grad_accum_steps
        scaled_loss.backward()

        if (self.global_step + 1) % self.config.grad_accum_steps == 0:
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()

        self.global_step += 1

        # Compute confidence
        confidence = None
        if self.config.metrics_to_track and ("accuracy" in self.config.metrics_to_track or "confidence" in self.config.metrics_to_track):
            with torch.no_grad():
                probs = torch.softmax(outputs.logits, dim=-1)
                confidence = torch.max(probs, dim=-1).values.mean().item()

        # Save checkpoint if needed
        if self.config.checkpoint_interval and self.global_step % self.config.checkpoint_interval == 0:
            self.save_checkpoint(self.global_step)

        return loss.item(), confidence

    def validate(self, data: Union[List[dict], 'DataLoader'], scaffold_provider: Optional[Callable] = None) -> tuple[float, dict]:
        """Validate model on provided data, returning loss and metrics."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        batches = 0
        metrics = {metric: 0.0 for metric in self.config.metrics_to_track}

        # Handle DataLoader or list of batches
        if isinstance(data, (list, tuple)):
            data_iter = [data[i:i + self.config.batch_size] for i in range(0, len(data), self.config.batch_size)]
        else:
            data_iter = data

        with torch.no_grad():
            for batch in data_iter:
                if isinstance(batch, (list, tuple)):
                    batch = collate_batch(batch, self.tokenizer.pad_token_id, self.config.max_seq_length, self.tokenizer)
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                scaffold_context = scaffold_provider(batch) if scaffold_provider else None
                if scaffold_context is not None:
                    scaffold_context = scaffold_context.to(self.device)

                with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.config.use_amp else torch.bfloat16):
                    outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                    loss = self.loss_fn(outputs.logits, batch["labels"])

                total_loss += loss.item()
                batches += 1

                # Compute metrics
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
        self.logger.write({
            "event": "validation",
            "loss": avg_loss,
            "metrics": metrics,
            "timestamp": time.time()
        })
        return avg_loss, metrics

    def save_checkpoint(self, step: int, suffix: Optional[str] = None):
        """Save trainer state to checkpoint."""
        checkpoint_dir = self.config.checkpoint_path
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_name = f"checkpoint_{step}{f'_{suffix}' if suffix else ''}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

        state_dict = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.global_step,
            "best_valid_loss": self.best_valid_loss,
            "patience": self.patience
        }
        torch.save(state_dict, checkpoint_path)
        self.logger.write({
            "event": "checkpoint_saved",
            "path": checkpoint_path,
            "step": step,
            "timestamp": time.time()
        })
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load trainer state from checkpoint."""
        if not os.path.exists(checkpoint_path):
            self.logger.write({
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
        self.logger.write({
            "event": "checkpoint_loaded",
            "path": checkpoint_path,
            "step": self.global_step,
            "timestamp": time.time()
        })
        print(f"Checkpoint loaded: {checkpoint_path} at step {self.global_step}")

    def should_stop(self) -> bool:
        """Check if training should stop based on early stopping criteria."""
        return self.patience >= self.config.max_patience

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

        # Prepare data
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

                scaffold_context = scaffold_provider(batch) if scaffold_provider else None
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
                    self.logger({
                        "event": "train_step",
                        "epoch": epoch + 1,
                        "step": self.global_step,
                        "loss": loss,
                        "confidence": confidence,
                        "timestamp": time.time()
                    })

                # Validate periodically
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
                        return

            avg_epoch_loss = epoch_loss / steps_in_epoch if steps_in_epoch > 0 else 0.0
            self.logger({
                "event": "epoch_end",
                "epoch": epoch + 1,
                "avg_loss": avg_epoch_loss,
                "timestamp": time.time()
            })
            print(f"Epoch {epoch + 1}/{self.config.max_epochs}: Avg Loss = {avg_epoch_loss:.4f}")
