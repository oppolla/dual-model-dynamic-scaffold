import time
from typing import Any, Dict, List, Optional, Deque, Tuple
from collections import deque, defaultdict
import traceback
import json
import threading
import math

import torch
from torch import nn

from sovl_error import ErrorHandler


class Curiosity:
    """Computes curiosity scores based on ignorance and novelty."""
    
    def __init__(
        self,
        weight_ignorance: float = 0.7,
        weight_novelty: float = 0.3,
        metrics_maxlen: int = 1000,
        logger: Optional[Any] = None
    ):
        self._validate_weights(weight_ignorance, weight_novelty)
        self.weight_ignorance = weight_ignorance
        self.weight_novelty = weight_novelty
        self.metrics_maxlen = metrics_maxlen
        self.logger = logger
        self.cosine_similarity = nn.CosineSimilarity(dim=-1, eps=1e-8)
        self.metrics = deque(maxlen=metrics_maxlen)

    def _validate_weights(self, ignorance: float, novelty: float) -> None:
        """Validate weight parameters."""
        if not (0.0 <= ignorance <= 1.0 and 0.0 <= novelty <= 1.0):
            raise ValueError("Weights must be between 0.0 and 1.0")
        if abs(ignorance + novelty - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")

    def compute_curiosity(
        self,
        base_conf: float,
        scaf_conf: float,
        memory_embeddings: List[torch.Tensor],
        query_embedding: torch.Tensor,
        device: torch.device
    ) -> float:
        """Compute curiosity score based on confidence and embeddings."""
        try:
            ignorance = self._compute_ignorance_score(base_conf, scaf_conf)
            novelty = (
                self._compute_novelty_score(memory_embeddings, query_embedding, device)
                if memory_embeddings and query_embedding is not None
                else 0.0
            )
            score = self.weight_ignorance * ignorance + self.weight_novelty * novelty
            return self._clamp_score(score)
        except Exception as e:
            self._log_error(f"Curiosity computation failed: {str(e)}")
            return 0.5

    def _compute_ignorance_score(self, base_conf: float, scaf_conf: float) -> float:
        """Compute ignorance component of curiosity score."""
        return self._clamp_score(1.0 - (base_conf * 0.5 + scaf_conf * 0.5))

    def _compute_novelty_score(
        self,
        memory_embeddings: List[torch.Tensor],
        query_embedding: torch.Tensor,
        device: torch.device
    ) -> float:
        """Compute novelty component of curiosity score."""
        query_embedding = query_embedding.to(device)
        similarities = [
            self.cosine_similarity(query_embedding, emb.to(device)).item()
            for emb in memory_embeddings
        ]
        return self._clamp_score(1.0 - max(similarities, default=0.0))

    def _clamp_score(self, score: float) -> float:
        """Clamp score between 0.0 and 1.0."""
        return max(0.0, min(1.0, score))

    def _log_error(self, message: str) -> None:
        """Log error if logger is available."""
        if self.logger:
            self.logger.log_error(
                error_msg=message,
                error_type="curiosity_error",
                stack_trace=traceback.format_exc()
            )


class CuriosityPressure:
    """Manages curiosity pressure accumulation and eruption."""
    
    def __init__(self):
        self.value: float = 0.0
        self.last_update: float = time.time()

    def update(
        self,
        temperament: float,
        confidence: float,
        silence: float,
        silence_threshold: float
    ) -> None:
        """Update pressure based on temperament, confidence, and silence."""
        time_delta = time.time() - self.last_update
        self.last_update = time.time()

        temperament_effect = 0.1 * max(0.0, temperament)
        confidence_effect = 0.05 * (1.0 - confidence)
        silence_effect = 0.2 * (silence / silence_threshold) if silence > silence_threshold else 0.0

        self.value = self._clamp_pressure(
            self.value + time_delta * (temperament_effect + confidence_effect + silence_effect)
        )

    def should_erupt(self, threshold: float) -> bool:
        """Check if pressure exceeds threshold."""
        return self.value >= threshold

    def drop_pressure(self, amount: float) -> None:
        """Reduce pressure by a specified amount."""
        self.value = self._clamp_pressure(self.value - amount)

    def _clamp_pressure(self, value: float) -> float:
        """Clamp pressure between 0.0 and 1.0."""
        return max(0.0, min(1.0, value))


class CuriosityCallbacks:
    """Handles curiosity-related callbacks."""
    
    def __init__(self, logger: Optional[Any] = None):
        self.callbacks: Dict[str, List[callable]] = {}
        self.logger = logger

    def register_callback(self, event: str, callback: callable) -> None:
        """Register a callback for an event."""
        self.callbacks.setdefault(event, []).append(callback)

    def trigger_callback(self, event: str, **kwargs) -> None:
        """Trigger callbacks for an event."""
        for callback in self.callbacks.get(event, []):
            try:
                callback(**kwargs)
            except Exception as e:
                self._log_error(f"Callback {event} failed: {str(e)}")

    def _log_error(self, message: str) -> None:
        """Log error if logger is available."""
        if self.logger:
            self.logger.log_error(
                error_msg=message,
                error_type="curiosity_error",
                stack_trace=traceback.format_exc()
            )


class CuriosityError(Exception):
    """Custom error for curiosity-related failures."""
    pass

class CuriosityManager:
    """Manages curiosity-driven exploration and learning."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        error_manager: ErrorHandler,
        device: torch.device,
        state_manager=None
    ):
        self.config_manager = config_manager
        self.logger = logger
        self.error_manager = error_manager
        self.state_manager = state_manager
        self.device = device
        self.metrics = defaultdict(list)
        self.curiosity_queue = deque(maxlen=config_manager.get("curiosity_queue_maxlen"))
        self._initialized = False
        self.state = None  # Will be set by SOVLState
        self.pressure = 0.0
        self.last_update = time.time()
        self.conversation_id = None
        self.state_hash = None
        self.pressure_mgr = CuriosityPressure()
        self.lock = threading.Lock()
        self._pressure_history = deque(maxlen=100)  # Track pressure changes
        self._last_pressure_change = 0.0
        self._pressure_change_cooldown = 1.0  # Minimum seconds between pressure changes
        self._min_pressure = 0.1  # Minimum pressure to maintain
        self._max_pressure = 0.9  # Maximum pressure to allow
        self._pressure_decay_rate = 0.95  # Rate at which pressure naturally decays
        self._initialize_components()
        
        # Log device initialization
        self._log_event(
            "device_initialized",
            device=str(self.device),
            device_type=self.device.type
        )
        
        # Add memory tracking
        self.total_memory_mb = 0.0
        self.max_memory_mb = config_manager.get("max_dream_memory_mb", 512.0)

    def _initialize_components(self) -> None:
        """Initialize curiosity components."""
        self.curiosity = Curiosity(
            weight_ignorance=self.config_manager.config.get("weight_ignorance", 0.7),
            weight_novelty=self.config_manager.config.get("weight_novelty", 0.3),
            metrics_maxlen=self.config_manager.config.get("metrics_maxlen", 1000),
            logger=self.logger
        )
        self.callbacks = CuriosityCallbacks(logger=self.logger)
        self.last_question_time: float = time.time()
        
    def set_state(self, state: SOVLState) -> None:
        """Set the SOVL state and synchronize with CuriosityState."""
        with self.lock:
            try:
                if not isinstance(state, SOVLState):
                    raise ValueError("State must be an instance of SOVLState")
                    
                # Validate device consistency
                if state.device != self.device:
                    self._log_warning(f"State device ({state.device}) differs from manager device ({self.device})")
                    
                self.state = state
                
                # Validate and synchronize curiosity state
                if not hasattr(state, 'curiosity'):
                    self._log_error("Missing curiosity state in SOVLState")
                    state.curiosity = CuriosityState(self.config_manager, self.logger, self.device)
                    
                # Validate required attributes
                required_attrs = [
                    'pressure', 'novelty_threshold_spontaneous',
                    'novelty_threshold_response', 'pressure_threshold',
                    'pressure_drop', 'silence_threshold',
                    'question_cooldown', 'queue_maxlen'
                ]
                
                missing_attrs = [attr for attr in required_attrs 
                               if not hasattr(state.curiosity, attr)]
                
                if missing_attrs:
                    self._log_error(f"Missing required attributes in CuriosityState: {missing_attrs}")
                    return
                
                # Validate and clamp parameter ranges
                if not (0 <= state.curiosity.pressure <= 1):
                    self._log_warning(f"Invalid pressure value: {state.curiosity.pressure}, clamping to [0,1]")
                    state.curiosity.pressure = max(0, min(1, state.curiosity.pressure))
                
                if not (0 <= state.curiosity.novelty_threshold_spontaneous <= 1):
                    self._log_warning(f"Invalid novelty_threshold_spontaneous: {state.curiosity.novelty_threshold_spontaneous}, clamping to [0,1]")
                    state.curiosity.novelty_threshold_spontaneous = max(0, min(1, state.curiosity.novelty_threshold_spontaneous))
                
                if not (0 <= state.curiosity.novelty_threshold_response <= 1):
                    self._log_warning(f"Invalid novelty_threshold_response: {state.curiosity.novelty_threshold_response}, clamping to [0,1]")
                    state.curiosity.novelty_threshold_response = max(0, min(1, state.curiosity.novelty_threshold_response))
                
                # Synchronize pressure
                self.pressure = state.curiosity.pressure
                
                # Log successful synchronization
                self._log_event("state_sync", {
                    "pressure": self.pressure,
                    "novelty_threshold_spontaneous": state.curiosity.novelty_threshold_spontaneous,
                    "novelty_threshold_response": state.curiosity.novelty_threshold_response,
                    "conversation_id": getattr(state, 'history', {}).get('conversation_id', None),
                    "state_hash": getattr(state, 'state_hash', None),
                    "device": str(self.device)
                })
                
            except Exception as e:
                self._log_error(f"Error synchronizing with SOVLState: {str(e)}")
                raise

    def _generate_query_embedding(self, state: Any, tokenizer: Any, model: Any, query: Optional[str]) -> Optional[torch.Tensor]:
        """Generate query embedding with device validation."""
        try:
            if not query:
                return None
                
            # Ensure model is on correct device
            model = model.to(self.device)
            
            # Tokenize and move to device
            inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embedding
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1)
                
            return embedding
            
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "query_embedding",
                "query": query
            })
            return None
            
    def _validate_device(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """Validate tensor device and move if necessary."""
        try:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"{name} must be a torch.Tensor")
                
            if tensor.device != self.device:
                self._log_warning(f"Moving {name} from {tensor.device} to {self.device}")
                tensor = tensor.to(self.device)
                
            return tensor
            
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "device_validation",
                "tensor_name": name,
                "tensor_shape": list(tensor.shape) if isinstance(tensor, torch.Tensor) else None
            })
            raise

    def update_pressure(self, new_pressure: float) -> None:
        """Update curiosity pressure and ensure state synchronization."""
        try:
            # Validate pressure value
            if not isinstance(new_pressure, (int, float)):
                raise ValueError("Pressure must be numeric")
            if not 0.0 <= new_pressure <= 1.0:
                raise ValueError("Pressure must be between 0.0 and 1.0")

            # Update local pressure
            self.pressure = new_pressure
            self.last_update = time.time()

            # Update state if available
            if self.state is not None:
                self.state.curiosity.pressure = new_pressure
                self.state_hash = self.state.state_hash
                self._log_event(
                    "pressure_updated",
                    pressure=new_pressure,
                    conversation_id=self.conversation_id,
                    state_hash=self.state_hash
                )
            else:
                self._log_warning(
                    "pressure_update_no_state",
                    pressure=new_pressure
                )

        except Exception as e:
            self._log_error(
                "pressure_update_failed",
                error=str(e),
                pressure=new_pressure
            )
            raise

    def _log_event(self, event_type: str, **kwargs) -> None:
        """Log an event with standardized metadata."""
        try:
            metadata = {
                "event_type": event_type,
                "timestamp": time.time(),
                "device": str(self.device),
                "conversation_id": self.conversation_id,
                "state_hash": self.state_hash,
                **kwargs
            }
            self.logger.info(f"Curiosity event: {json.dumps(metadata)}")
        except Exception as e:
            self._log_error(f"Failed to log event: {str(e)}", event_type=event_type, **kwargs)

    def _log_warning(self, event_type: str, **kwargs) -> None:
        """Log a warning with standardized metadata."""
        try:
            metadata = {
                "event_type": event_type,
                "timestamp": time.time(),
                "device": str(self.device),
                "conversation_id": self.conversation_id,
                "state_hash": self.state_hash,
                **kwargs
            }
            self.logger.warning(f"Curiosity warning: {json.dumps(metadata)}")
        except Exception as e:
            self.logger.error(f"Failed to log curiosity warning: {str(e)}")

    def _log_error(self, message: str, error_type: str = "curiosity_error", **kwargs) -> None:
        """Log an error with standardized metadata."""
        try:
            self.logger.log_error(
                error_msg=message,
                error_type=error_type,
                stack_trace=traceback.format_exc(),
                additional_info={
                    "device": str(self.device),
                    "conversation_id": self.conversation_id,
                    "state_hash": self.state_hash,
                    **kwargs
                }
            )
        except Exception as e:
            # Fallback logging if primary logging fails
            print(f"Failed to log error: {str(e)}")
            print(f"Original error: {message}")
            print(f"Stack trace: {traceback.format_exc()}")

    def update_metrics(self, metric_name: str, value: float) -> bool:
        """Update curiosity metrics."""
        try:
            maxlen = self.config_manager.get("metrics_maxlen")
            self.metrics[metric_name].append(value)
            if len(self.metrics[metric_name]) > maxlen:
                self.metrics[metric_name].pop(0)
            return True
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "metrics_update",
                "metric_name": metric_name,
                "value": value
            })
            return False
            
    def calculate_curiosity_score(self, prompt: str) -> float:
        """Calculate curiosity score for a prompt."""
        try:
            if not self._initialized:
                raise RuntimeError("CuriosityManager not initialized")
                
            novelty_score = self._calculate_novelty(prompt)
            ignorance_score = self._calculate_ignorance(prompt)
            
            weight_novelty = self.config_manager.get("weight_novelty")
            weight_ignorance = self.config_manager.get("weight_ignorance")
            
            return (weight_novelty * novelty_score + 
                   weight_ignorance * ignorance_score)
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "curiosity_score_calculation",
                "prompt": prompt
            })
            return 0.0
            
    def should_explore(self, prompt: str) -> bool:
        """Determine if exploration should be triggered."""
        try:
            if not self._initialized:
                return False
                
            curiosity_score = self.calculate_curiosity_score(prompt)
            threshold = self.config_manager.get("novelty_threshold_spontaneous")
            
            return curiosity_score > threshold
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "exploration_check",
                "prompt": prompt
            })
            return False
            
    def queue_exploration(self, prompt: str) -> bool:
        """Queue a prompt for exploration."""
        try:
            if not self._initialized:
                return False
                
            self.curiosity_queue.append({
                "prompt": prompt,
                "timestamp": time.time(),
                "score": self.calculate_curiosity_score(prompt)
            })
            return True
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "exploration_queue",
                "prompt": prompt
            })
            return False
            
    def get_next_exploration(self) -> Optional[Dict]:
        """Get next prompt for exploration."""
        try:
            if not self.curiosity_queue:
                return None
                
            timeout = self.config_manager.get("curiosity_question_timeout")
            current_time = time.time()
            
            while self.curiosity_queue:
                item = self.curiosity_queue[0]
                if current_time - item["timestamp"] > timeout:
                    self.curiosity_queue.popleft()
                else:
                    return item
                    
            return None
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "get_next_exploration"
            })
            return None
            
    def _calculate_novelty(self, prompt: str) -> float:
        """Calculate novelty score for a prompt."""
        try:
            if not self.state_manager:
                return 0.0
                
            seen_prompts = self.state_manager.get_seen_prompts()
            if not seen_prompts:
                return 1.0
                
            similarities = [
                cosine_similarity(
                    self.state_manager.get_prompt_embedding(prompt),
                    self.state_manager.get_prompt_embedding(seen)
                )
                for seen in seen_prompts
            ]
            
            return 1.0 - max(similarities)
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "novelty_calculation",
                "prompt": prompt
            })
            return 0.0
            
    def _calculate_ignorance(self, prompt: str) -> float:
        """Calculate ignorance score for a prompt."""
        try:
            if not self.state_manager:
                return 0.0
                
            confidence = self.state_manager.get_confidence()
            if confidence is None:
                return 1.0
                
            decay_rate = self.config_manager.get("curiosity_decay_rate")
            return math.exp(-decay_rate * confidence)
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "ignorance_calculation",
                "prompt": prompt
            })
            return 0.0

    def generate_question(
        self,
        state: Any,
        tokenizer: Any,
        model: Any,
        max_length: int = 512
    ) -> Optional[str]:
        """Generate a curiosity-driven question."""
        try:
            if not self._initialized:
                raise RuntimeError("CuriosityManager not initialized")
                
            # Get next exploration item
            item = self.get_next_exploration()
            if not item:
                return None
                
            # Process the prompt through the model
            inputs = tokenizer(
                item["prompt"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
                
            question = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return question
            
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "question_generation",
                "max_length": max_length
            })
            return None
            
    def set_state(self, state: Any) -> bool:
        """Set the state for the CuriosityManager."""
        try:
            if not state:
                raise ValueError("State cannot be None")
                
            self.state = state
            self._initialized = True
            return True
            
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "state_set",
                "state_hash": getattr(state, "state_hash", None)
            })
            return False
            
    def reset(self) -> bool:
        """Reset the CuriosityManager state."""
        try:
            self.metrics.clear()
            self.curiosity_queue.clear()
            self._initialized = False
            self.state = None
            return True
            
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "manager_reset"
            })
            return False

    def tune(self, **params) -> None:
        """Update curiosity parameters with validation and logging.
        
        Args:
            **params: Key-value pairs of parameters to update
        """
        try:
            with self.lock:
                for key, value in params.items():
                    # Validate parameter exists and is valid
                    if not hasattr(self, key):
                        self._log_warning(
                            "invalid_parameter",
                            message=f"Invalid curiosity parameter: {key}",
                            parameter=key,
                            value=value
                        )
                        continue
                        
                    # Validate value type and range
                    if key in ["pressure", "weight_ignorance", "weight_novelty"]:
                        if not isinstance(value, (int, float)):
                            self._log_warning(
                                "invalid_value_type",
                                message=f"Invalid type for {key}: {type(value)}",
                                parameter=key,
                                value=value
                            )
                            continue
                        if not 0.0 <= value <= 1.0:
                            self._log_warning(
                                "invalid_value_range",
                                message=f"Value out of range for {key}: {value}",
                                parameter=key,
                                value=value
                            )
                            continue
                            
                    # Update parameter
                    setattr(self, key, value)
                    self._log_event(
                        "parameter_updated",
                        message=f"Updated {key} to {value}",
                        parameter=key,
                        value=value
                    )
                    
                    # Special handling for queue_maxlen
                    if key == "queue_maxlen":
                        self.curiosity_queue = deque(maxlen=value)
                        
        except Exception as e:
            self._log_error(
                "tune_failed",
                message=f"Failed to tune parameters: {str(e)}",
                parameters=params,
                error=str(e)
            )
            raise

    def get_metrics_summary(self) -> Dict[str, float]:
        """Get summary statistics of tracked metrics."""
        try:
            if not self._initialized:
                return {}
                
            summary = {}
            for metric_name, values in self.metrics.items():
                if values:
                    summary[f"{metric_name}_mean"] = sum(values) / len(values)
                    summary[f"{metric_name}_max"] = max(values)
                    summary[f"{metric_name}_min"] = min(values)
                    
            return summary
            
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "metrics_summary"
            })
            return {}
            
    def get_exploration_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about the exploration queue."""
        try:
            if not self._initialized:
                return {}
                
            current_time = time.time()
            stats = {
                "queue_length": len(self.curiosity_queue),
                "avg_score": 0.0,
                "oldest_item_age": 0.0,
                "newest_item_age": 0.0
            }
            
            if self.curiosity_queue:
                scores = [item["score"] for item in self.curiosity_queue]
                stats["avg_score"] = sum(scores) / len(scores)
                stats["oldest_item_age"] = current_time - self.curiosity_queue[0]["timestamp"]
                stats["newest_item_age"] = current_time - self.curiosity_queue[-1]["timestamp"]
                
            return stats
            
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "queue_stats"
            })
            return {}

    def _get_valid_memory_embeddings(self, state: SOVLState) -> List[torch.Tensor]:
        """
        Get valid memory embeddings while respecting memory limits.
        
        Args:
            state: Current SOVLState instance
            
        Returns:
            List of valid memory embeddings within memory limits
        """
        try:
            memory_embeddings = []
            hidden_size = self.config_manager.get("hidden_size")
            total_size = 0.0
            
            if not hasattr(state, 'dream_memory') or not state.dream_memory:
                return memory_embeddings
                
            for entry in state.dream_memory:
                tensor = entry["tensor"]
                # Calculate memory size in MB
                memory_size = tensor.element_size() * tensor.nelement() / (1024 * 1024)
                
                # Check memory limit
                if total_size + memory_size > self.max_memory_mb:
                    self._log_event("memory_limit_reached", {
                        "total_size_mb": total_size,
                        "skipped_size_mb": memory_size,
                        "max_memory_mb": self.max_memory_mb
                    })
                    break
                    
                # Validate tensor shape
                if tensor.shape[-1] == hidden_size:
                    try:
                        # Validate and move tensor to correct device
                        tensor = self._validate_device(tensor, f"dream_memory_tensor_{len(memory_embeddings)}")
                        memory_embeddings.append(tensor)
                        total_size += memory_size
                    except Exception as e:
                        self._log_warning("tensor_validation_failed", {
                            "error": str(e),
                            "tensor_shape": list(tensor.shape)
                        })
                        continue
                else:
                    self._log_warning("invalid_tensor_shape", {
                        "expected_size": hidden_size,
                        "actual_size": tensor.shape[-1]
                    })
                    
            # Update total memory tracking
            self.total_memory_mb = total_size
            
            return memory_embeddings
            
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "get_valid_memory_embeddings",
                "hidden_size": hidden_size if 'hidden_size' in locals() else None,
                "total_size_mb": total_size if 'total_size' in locals() else None
            })
            return []
            
    def compute_curiosity(
        self,
        base_conf: float,
        scaf_conf: float,
        state: SOVLState,
        query_embedding: torch.Tensor,
        device: torch.device
    ) -> float:
        """Compute curiosity score based on confidence and embeddings."""
        try:
            ignorance = self._compute_ignorance_score(base_conf, scaf_conf)
            
            # Get memory embeddings with memory limits
            memory_embeddings = self._get_valid_memory_embeddings(state)
            
            novelty = (
                self._compute_novelty_score(memory_embeddings, query_embedding, device)
                if memory_embeddings and query_embedding is not None
                else 0.0
            )
            
            score = self.weight_ignorance * ignorance + self.weight_novelty * novelty
            return self._clamp_score(score)
            
        except Exception as e:
            self.error_manager.handle_curiosity_error(e, {
                "operation": "compute_curiosity",
                "base_conf": base_conf,
                "scaf_conf": scaf_conf,
                "memory_count": len(memory_embeddings) if 'memory_embeddings' in locals() else 0
            })
            return 0.5

    def get_pressure(self) -> float:
        """Get current pressure with validation and decay."""
        try:
            with self.lock:
                # Apply natural decay
                time_delta = time.time() - self.last_update
                if time_delta > 0:
                    self.pressure *= (self._pressure_decay_rate ** time_delta)
                    self.last_update = time.time()
                
                # Ensure pressure stays within bounds
                self.pressure = max(self._min_pressure, min(self._max_pressure, self.pressure))
                
                # Record pressure in history
                self._pressure_history.append((time.time(), self.pressure))
                
                return self.pressure
                
        except Exception as e:
            self._log_error(f"Failed to get pressure: {str(e)}")
            return self._min_pressure  # Return minimum pressure on error

    def reduce_pressure(self, amount: float) -> None:
        """Reduce pressure with validation and cooldown."""
        try:
            with self.lock:
                current_time = time.time()
                
                # Check cooldown
                if current_time - self._last_pressure_change < self._pressure_change_cooldown:
                    self._log_warning(
                        "pressure_change_cooldown",
                        message="Pressure change too frequent",
                        last_change=self._last_pressure_change,
                        current_time=current_time
                    )
                    return
                
                # Validate amount
                if not isinstance(amount, (int, float)) or amount <= 0:
                    self._log_warning(
                        "invalid_pressure_reduction",
                        message=f"Invalid pressure reduction amount: {amount}",
                        amount=amount
                    )
                    return
                
                # Calculate new pressure
                new_pressure = self.pressure - amount
                
                # Ensure pressure stays above minimum
                if new_pressure < self._min_pressure:
                    self._log_warning(
                        "pressure_min_reached",
                        message=f"Pressure reduction would go below minimum: {new_pressure}",
                        current_pressure=self.pressure,
                        reduction=amount
                    )
                    new_pressure = self._min_pressure
                
                # Update pressure
                self.pressure = new_pressure
                self._last_pressure_change = current_time
                self.last_update = current_time
                
                # Log pressure change
                self._log_event(
                    "pressure_reduced",
                    message=f"Pressure reduced by {amount}",
                    old_pressure=self.pressure + amount,
                    new_pressure=self.pressure,
                    reduction=amount
                )
                
        except Exception as e:
            self._log_error(f"Failed to reduce pressure: {str(e)}")

    def get_pressure_stats(self) -> Dict[str, Any]:
        """Get statistics about pressure changes."""
        try:
            with self.lock:
                if not self._pressure_history:
                    return {
                        "current_pressure": self.pressure,
                        "min_pressure": self._min_pressure,
                        "max_pressure": self._max_pressure,
                        "history_size": 0
                    }
                
                pressures = [p for _, p in self._pressure_history]
                return {
                    "current_pressure": self.pressure,
                    "min_pressure": self._min_pressure,
                    "max_pressure": self._max_pressure,
                    "avg_pressure": sum(pressures) / len(pressures),
                    "min_observed": min(pressures),
                    "max_observed": max(pressures),
                    "history_size": len(pressures),
                    "last_change": self._last_pressure_change
                }
                
        except Exception as e:
            self._log_error(f"Failed to get pressure stats: {str(e)}")
            return {}
