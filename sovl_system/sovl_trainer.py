"""
sovl_trainer.py - Comprehensive training module for SOVL system
Handles optimization, scheduling, gradient accumulation, and validation.
"""

import warnings
from typing import Dict, List, Tuple, Optional, Callable, Union
import torch
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.cuda.amp import autocast, GradScaler
from threading import Lock
from collections import deque
import math
import time

class Logger:
    """Simple logger for recording training events"""
    def __init__(self, log_file: str):
        self.log_file = log_file

    def record(self, entry: Dict):
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"Logging failed: {e}")

class TrainingConfig:
    """Configuration container for training parameters"""
    def __init__(
        self,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        total_steps: int = 100000,
        grad_accum_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
        max_patience: int = 2,
        **kwargs
    ):
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if grad_accum_steps < 1:
            raise ValueError("Gradient accumulation steps must be at least 1")
        if max_grad_norm <= 0:
            raise ValueError("Max gradient norm must be positive")
        if warmup_steps < 0:
            raise ValueError("Warmup steps cannot be negative")
        if total_steps < warmup_steps:
            raise ValueError("Total steps must be at least warmup steps")
        if max_patience < 0:
            raise ValueError("Max patience cannot be negative")

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.max_patience = max_patience
        self.extra_args = kwargs

class SOVLTrainer:
    """
    Complete training manager handling:
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Loss computation
    - Validation metrics
    - Confidence tracking
    - Early stopping
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainingConfig,
        device: torch.device,
        loss_fn: Optional[_Loss] = None,
        custom_optimizer: Optional[Optimizer] = None,
        custom_scheduler: Optional[LambdaLR] = None,
        logger: Optional[Logger] = None,
        memory_lock: Optional[Lock] = None
    ):
        """
        Initialize trainer with model and configuration.
        
        Args:
            model: Model to be trained
            config: Training configuration
            device: Target device (cuda/cpu)
            loss_fn: Custom loss function (default: CrossEntropyLoss)
            custom_optimizer: Pre-configured optimizer
            custom_scheduler: Pre-configured scheduler
            logger: Logger for training events
            memory_lock: Lock for thread-safe memory operations
        """
        self.model = model
        self.config = config
        self.device = device
        self.scaler = GradScaler(enabled=config.use_amp)
        self.logger = logger or Logger("trainer_logs.jsonl")
        self.memory_lock = memory_lock or Lock()
        
        # Initialize training state
        self.global_step = 0
        self.total_loss = 0.0
        self.current_lr = 0.0
        self.best_valid_loss = float('inf')
        self.patience = 0
        
        # Setup components
        self.loss_fn = loss_fn or F.cross_entropy
        self.optimizer = custom_optimizer or self._configure_optimizer()
        self.scheduler = custom_scheduler or self._configure_scheduler()
        
    def _configure_optimizer(self) -> Optimizer:
        """Configure optimizer with parameter groups"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            **self.config.extra_args.get("optimizer", {})
        )
    
    def _configure_scheduler(self) -> LambdaLR:
        """Create learning rate schedule with warmup"""
        def lr_lambda(current_step: int):
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(1, self.config.warmup_steps))
            progress = float(current_step - self.config.warmup_steps) / float(
                max(1, self.config.total_steps - self.config.warmup_steps)
            )
            return max(0.0, 1.0 - progress)
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def _calculate_confidence_score(self, logits, target_ids) -> float:
        """Calculate confidence score based on logits and target IDs"""
        try:
            if not logits or len(logits) == 0 or len(logits) != len(target_ids):
                self.logger.record({
                    "warning": f"Invalid logits length {len(logits) if logits else 'N/A'} vs target_ids {len(target_ids)}",
                    "global_step": self.global_step,
                    "timestamp": time.time()
                })
                return 0.5
            probs = torch.softmax(torch.stack(logits), dim=-1)
            max_probs = torch.max(probs, dim=-1).values
            if max_probs.var().item() < 1e-5:
                return 0.2
            return max_probs.mean().item()
        except Exception as e:
            self.logger.record({
                "error": f"Confidence score failed: {str(e)}",
                "global_step": self.global_step,
                "timestamp": time.time()
            })
            return 0.5
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        scaffold_context: Optional[torch.Tensor] = None,
        grad_clip: bool = True,
        loss_weight_fn: Optional[Callable[[Dict], float]] = None,
        dry_run: bool = False,
        memory_check: Optional[Callable] = None
    ) -> Tuple[float, Optional[float]]:
        """
        Execute single training step with optional gradient accumulation.
        
        Args:
            batch: Input batch dictionary
            scaffold_context: Optional context for cross-attention
            grad_clip: Whether to clip gradients
            loss_weight_fn: Function to compute loss weight
            dry_run: If True, skip gradient updates
            memory_check: Optional memory monitoring function
            
        Returns:
            step_loss: Unscaled loss value
            confidence: Confidence score or None
        """
        if memory_check:
            memory_check()

        with self.memory_lock:
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            if dry_run:
                self.model.eval()
                with torch.no_grad(), autocast(enabled=self.config.use_amp):
                    outputs = self.model(**inputs, scaffold_context=scaffold_context)
                    loss = self.loss_fn(
                        outputs.logits.view(-1, outputs.logits.size(-1)),
                        inputs["labels"].view(-1),
                        ignore_index=-100
                    )
                confidence = self._calculate_confidence_score(outputs.logits, inputs["labels"]) if "labels" in inputs else None
                return loss.item(), confidence

            self.model.train()
            with autocast(enabled=self.config.use_amp):
                outputs = self.model(**inputs, scaffold_context=scaffold_context)
                loss = self.loss_fn(
                    outputs.logits.view(-1, outputs.logits.size(-1)),
                    inputs["labels"].view(-1),
                    ignore_index=-100
                )
                if loss_weight_fn:
                    weight = loss_weight_fn(batch)
                    loss = loss * weight
                loss = loss / self.config.grad_accum_steps

            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.record({
                    "error": "Invalid loss detected",
                    "loss_value": loss.item(),
                    "global_step": self.global_step,
                    "timestamp": time.time()
                })
                return 0.0, None

            self.scaler.scale(loss).backward()

            grad_norm = 0.0
            if grad_clip and (self.global_step + 1) % self.config.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn_utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                ).item()
                if grad_norm > self.config.max_grad_norm * 10:
                    self.logger.record({
                        "warning": "Excessive gradient norm",
                        "grad_norm": grad_norm,
                        "global_step": self.global_step,
                        "timestamp": time.time()
                    })

            if (self.global_step + 1) % self.config.grad_accum_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.current_lr = self.scheduler.get_last_lr()[0]

            self.global_step += 1
            self.total_loss += loss.item()

            confidence = self._calculate_confidence_score(outputs.logits, inputs["labels"]) if "labels" in inputs else None
            return loss.item() * self.config.grad_accum_steps, confidence
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: torch.utils.data.DataLoader,
        scaffold_provider: Optional[Callable] = None,
        metric_fns: Optional[Dict[str, Callable]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Run full validation pass.
        
        Args:
            val_loader: Validation data loader
            scaffold_provider: Function to generate scaffold context
            metric_fns: Additional metrics to compute
            
        Returns:
            avg_loss: Average validation loss
            metrics: Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        metrics = {"accuracy": 0.0, "perplexity": 0.0}

        with self.memory_lock:
            for batch in val_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                scaffold_ctx = scaffold_provider(batch) if scaffold_provider else None

                with autocast(enabled=self.config.use_amp):
                    outputs = self.model(**inputs, scaffold_context=scaffold_ctx)
                    loss = self.loss_fn(
                        outputs.logits.view(-1, outputs.logits.size(-1)),
                        inputs["labels"].view(-1),
                        ignore_index=-100
                    )

                batch_size = inputs["input_ids"].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # Default metrics
                if "labels" in inputs:
                    preds = torch.argmax(outputs.logits, dim=-1)
                    mask = inputs["labels"] != -100
                    accuracy = (preds[mask] == inputs["labels"][mask]).float().mean().item()
                    metrics["accuracy"] += accuracy * batch_size
                metrics["perplexity"] += math.exp(loss.item()) * batch_size

                # Custom metrics
                if metric_fns:
                    for name, fn in metric_fns.items():
                        metrics[name] = metrics.get(name, 0) + (
                            fn(outputs, inputs) * batch_size
                        )

        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        metrics = {k: v / total_samples for k, v in metrics.items() if total_samples > 0}
        metrics["perplexity"] = math.exp(avg_loss) if avg_loss != float('inf') else float('inf')

        # Update early stopping
        if avg_loss < self.best_valid_loss:
            self.best_valid_loss = avg_loss
            self.patience = 0
        else:
            self.patience += 1

        return avg_loss, metrics
    
    def should_stop(self) -> bool:
        """Check if training should stop based on patience"""
        return self.patience >= self.config.max_patience
    
    def get_learning_rate(self) -> float:
        """Get current learning rate"""
        return self.current_lr if hasattr(self, "current_lr") else 0.0
    
    def state_dict(self) -> Dict:
        """Return trainer state dictionary"""
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "total_loss": self.total_loss,
            "best_valid_loss": self.best_valid_loss,
            "patience": self.patience
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load trainer state"""
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.scaler.load_state_dict(state_dict["scaler"])
        self.global_step = state_dict["global_step"]
        self.total_loss = state_dict["total_loss"]
        self.best_valid_loss = state_dict.get("best_valid_loss", float('inf'))
        self.patience = state_dict.get("patience", 0)

def collate_batch(
    batch: List[Dict[str, Union[List[int], torch.Tensor]]],
    pad_token_id: int,
    max_seq_length: Optional[int] = None,
    label_pad_id: int = -100
) -> Dict[str, torch.Tensor]:
    """
    Efficient batch collation with padding and truncation.
    
    Args:
        batch: List of tokenized examples
        pad_token_id: Padding token ID
        max_seq_length: Optional maximum sequence length
        label_pad_id: Padding ID for labels
        
    Returns:
        Dictionary of batched tensors:
        {
            "input_ids": torch.Tensor,
            "attention_mask": torch.Tensor,
            "labels": torch.Tensor (optional)
        }
    """
    def _pad_sequence(sequences, padding_value):
        return torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(s, dtype=torch.long) for s in sequences],
            batch_first=True,
            padding_value=padding_value
        )
    
    # Process input_ids
    input_ids = [item["input_ids"] for item in batch]
    if max_seq_length:
        input_ids = [ids[:max_seq_length] for ids in input_ids]
    
    padded_inputs = _pad_sequence(input_ids, pad_token_id)
    attention_mask = (padded_inputs != pad_token_id).long()
    
    result = {
        "input_ids": padded_inputs,
        "attention_mask": attention_mask
    }
    
    # Process labels if present
    if "labels" in batch[0]:
        labels = [item["labels"] for item in batch]
        if max_seq_length:
            labels = [l[:max_seq_length] for l in labels]
        padded_labels = _pad_sequence(labels, label_pad_id)
        result["labels"] = padded_labels
    
    # Include prompt for loss weighting if present
    if "prompt" in batch[0]:
        result["prompt"] = [item["prompt"] for item in batch]
    
    return result

def configure_optimizer(
    model: torch.nn.Module,
    config: Union[TrainingConfig, Dict]
) -> Optimizer:
    """
    Configure optimizer from model and config.
    
    Args:
        model: Model containing parameters to optimize
        config: Either TrainingConfig or dictionary with params
        
    Returns:
        Configured optimizer instance
    """
    if isinstance(config, dict):
        config = TrainingConfig(**config)
    
    trainer = SOVLTrainer(model, config, device=torch.device("cpu"))
    return trainer.optimizer

# Example metric functions
def accuracy_metric(
    outputs: torch.Tensor,
    inputs: Dict[str, torch.Tensor],
    ignore_index: int = -100
) -> float:
    """Compute accuracy ignoring padding tokens"""
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    preds = torch.argmax(logits, dim=-1)
    labels = inputs["labels"]
    mask = labels != ignore_index
    return (preds[mask] == labels[mask]).float().mean().item()

def perplexity_metric(loss: float) -> float:
    """Compute perplexity from loss"""
    return math.exp(loss) if loss != float('inf') else float('inf')
