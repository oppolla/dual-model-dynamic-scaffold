import time
from typing import Any, Dict, List, Optional, Deque, Tuple
from collections import deque
import traceback
import json

import torch
from torch import nn


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
        device: torch.device
    ):
        self.config_manager = config_manager
        self.logger = logger
        self.device = device
        self.state = None  # Will be set by SOVLState
        self.pressure = 0.0
        self.last_update = time.time()
        self.conversation_id = None
        self.state_hash = None
        self.pressure_mgr = CuriosityPressure()
        self.metrics: Deque[Dict[str, Any]] = deque(maxlen=config_manager.config.get("metrics_maxlen", 1000))
        self._initialize_components()
        
        # Log device initialization
        self._log_event(
            "device_initialized",
            device=str(self.device),
            device_type=self.device.type
        )

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
        """Set the SOVLState reference and validate synchronization."""
        if not isinstance(state, SOVLState):
            raise ValueError("State must be an instance of SOVLState")
            
        self.state = state
        self.conversation_id = state.history.conversation_id
        self.state_hash = state.state_hash
        
        # Validate state synchronization
        self._validate_state_sync()
        self._log_event(
            "state_sync_initialized",
            conversation_id=self.conversation_id,
            state_hash=self.state_hash,
            pressure=self.pressure
        )

    def _validate_state_sync(self) -> None:
        """Validate synchronization between CuriosityManager and SOVLState."""
        if self.state is None:
            raise StateError("SOVLState not set")
            
        if not hasattr(self.state, 'curiosity'):
            raise StateError("SOVLState missing curiosity attribute")
            
        # Validate pressure synchronization
        if abs(self.pressure - self.state.curiosity.pressure) > 0.01:
            self._log_warning(
                "pressure_mismatch",
                manager_pressure=self.pressure,
                state_pressure=self.state.curiosity.pressure
            )
            # Align pressures
            self.pressure = self.state.curiosity.pressure

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

    def update_metrics(
        self,
        question: Optional[str] = None,
        score: float = 0.5,
        spontaneous: bool = False,
        answered: bool = False,
        conversation_id: Optional[str] = None,
        state_hash: Optional[str] = None
    ) -> None:
        """
        Update curiosity metrics.
        
        Args:
            question: Optional question text
            score: Curiosity score
            spontaneous: Whether the question was spontaneous
            answered: Whether the question was answered
            conversation_id: Optional conversation ID
            state_hash: Optional state hash
        """
        try:
            metric = {
                "question": question,
                "score": score,
                "spontaneous": spontaneous,
                "answered": answered,
                "timestamp": time.time(),
                "conversation_id": conversation_id,
                "state_hash": state_hash
            }
            self.metrics.append(metric)
            self._log_metrics_update(metric)
            self.callbacks.trigger_callback("metrics_updated", **metric)
            
        except Exception as e:
            self.logger.record({
                "error": f"Failed to update metrics: {str(e)}",
                "stack_trace": traceback.format_exc()
            })
            
    def _log_metrics_update(self, metric: Dict[str, Any]) -> None:
        """Log metrics update event."""
        try:
            self.logger.record_event(
                event_type="curiosity_metrics_update",
                message="Curiosity metrics updated",
                level="info",
                additional_info={
                    "metric": metric,
                    "timestamp": time.time(),
                    "conversation_id": self.conversation_id,
                    "state_hash": self.state_hash,
                    "device": str(self.device)
                }
            )
        except Exception as e:
            self.logger.record_event(
                event_type="logging_failure",
                message=f"Failed to log metrics update: {str(e)}",
                level="warning",
                additional_info={
                    "error": str(e),
                    "stack_trace": traceback.format_exc(),
                    "conversation_id": self.conversation_id,
                    "state_hash": self.state_hash
                }
            )
            
    def get_metrics(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get metrics with optional limit.
        
        Args:
            limit: Optional limit on number of metrics to return
            
        Returns:
            List of metrics
        """
        metrics = list(self.metrics)
        return metrics[-limit:] if limit is not None else metrics
        
    def clear_metrics(self) -> None:
        """Clear all metrics."""
        self.metrics.clear()
        self.callbacks.trigger_callback("metrics_cleared")
        
    def should_erupt(self, state: Any) -> bool:
        """
        Check if curiosity pressure exceeds threshold.
        
        Args:
            state: Current system state
            
        Returns:
            bool: Whether to generate a question
        """
        try:
            if not hasattr(state, 'curiosity'):
                return False
                
            # Use state's curiosity pressure for consistency
            pressure = state.curiosity.pressure
            threshold = self.config_manager.config.get("pressure_threshold", 0.7)
            
            self._log_event("eruption_check", {
                "pressure": pressure,
                "threshold": threshold,
                "conversation_id": getattr(state, 'history', {}).get('conversation_id', None),
                "state_hash": getattr(state, 'state_hash', None)
            })
            
            return pressure >= threshold
            
        except Exception as e:
            self._log_error(f"Failed to check eruption: {str(e)}")
            return False

    def reduce_pressure(self, amount: float, state: Any) -> None:
        """
        Reduce curiosity pressure by specified amount.
        
        Args:
            amount: Amount to reduce pressure by
            state: Current system state
        """
        try:
            # Reduce pressure in CuriosityPressure
            self.pressure_mgr.drop_pressure(amount)
            
            # Sync with CuriosityState
            if hasattr(state, 'curiosity'):
                state.curiosity.pressure = self.pressure_mgr.value
                
                self._log_event("pressure_reduced", {
                    "amount": amount,
                    "new_pressure": state.curiosity.pressure,
                    "conversation_id": getattr(state, 'history', {}).get('conversation_id', None),
                    "state_hash": getattr(state, 'state_hash', None)
                })
                
        except Exception as e:
            self._log_error(f"Failed to reduce pressure: {str(e)}")
            raise

    def _validate_device(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """Validate and move tensor to correct device with enhanced logging."""
        try:
            if tensor.device != self.device:
                self._log_warning(
                    "device_mismatch",
                    tensor_name=name,
                    source_device=str(tensor.device),
                    target_device=str(self.device),
                    tensor_shape=list(tensor.shape),
                    tensor_dtype=str(tensor.dtype)
                )
                tensor = tensor.to(self.device)
            return tensor
        except Exception as e:
            self._log_error(
                "device_validation_failed",
                error=str(e),
                tensor_name=name,
                tensor_shape=list(tensor.shape) if isinstance(tensor, torch.Tensor) else None,
                tensor_dtype=str(tensor.dtype) if isinstance(tensor, torch.Tensor) else None
            )
            raise
        
    def _get_valid_memory_embeddings(self, state: Any) -> List[torch.Tensor]:
        """Extract valid memory embeddings from state with enhanced device validation."""
        memory_embeddings = []
        hidden_size = self.config_manager.config["default_hidden_size"]
        
        if not hasattr(state, 'dream_memory') or not state.dream_memory:
            return memory_embeddings

        try:
            for i, entry in enumerate(state.dream_memory):
                tensor = entry["tensor"]
                if tensor.shape[-1] == hidden_size:
                    # Validate and move tensor to correct device
                    tensor = self._validate_device(tensor, f"dream_memory_tensor_{i}")
                    memory_embeddings.append(tensor)
                else:
                    self._log_warning(
                        "invalid_memory_tensor_shape",
                        tensor_index=i,
                        tensor_shape=list(tensor.shape),
                        expected_hidden_size=hidden_size
                    )
        except Exception as e:
            self._log_error(
                "memory_embedding_extraction_failed",
                error=str(e),
                memory_length=len(state.dream_memory) if hasattr(state, 'dream_memory') else 0
            )
        return memory_embeddings

    def _generate_query_embedding(
        self, state: Any, tokenizer: Any, model: Any, query: Optional[str]
    ) -> Optional[torch.Tensor]:
        """Generate query embedding with device validation."""
        try:
            # Validate model device
            if model is not None:
                model_device = next(model.parameters()).device
                if model_device != self.device:
                    self._log_warning(
                        "model_device_mismatch",
                        model_device=str(model_device),
                        target_device=str(self.device)
                    )
                    model.to(self.device)

            if not (query and tokenizer and model):
                if hasattr(state, 'last_prompt_embedding') and state.last_prompt_embedding is not None:
                    return self._validate_device(state.last_prompt_embedding, "last_prompt_embedding")
                return torch.zeros(self.config_manager.config["default_hidden_size"], device=self.device)

            inputs = tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = (
                    outputs.hidden_states[-1] or
                    getattr(getattr(outputs, 'base_model_output', None), 'hidden_states', [None])[-1]
                )
                if hidden_states is not None:
                    embedding = hidden_states[:, -1, :].squeeze()
                    return self._validate_device(embedding, "query_embedding")
                self._log_warning("model_output_lacks_hidden_states")
        except Exception as e:
            self._log_error(
                "query_embedding_generation_failed",
                error=str(e),
                query=query[:100] if query else None,
                model_device=str(model.device) if model is not None else None
            )
        return None

    def compute_curiosity(
        self,
        state: Any,
        tokenizer: Any,
        model: Any,
        query: Optional[str] = None
    ) -> float:
        """Compute curiosity score with enhanced device validation."""
        try:
            # Store state metadata for logging
            self.conversation_id = getattr(state, 'conversation_id', None)
            self.state_hash = getattr(state, 'state_hash', None)
            
            # Validate state
            self._validate_state(state)
            
            # Get embeddings with device validation
            query_embedding = self._generate_query_embedding(state, tokenizer, model, query)
            memory_embeddings = self._get_valid_memory_embeddings(state)
            
            # Log device information
            self._log_event(
                "curiosity_computation_started",
                query_embedding_device=str(query_embedding.device) if query_embedding is not None else None,
                memory_embeddings_devices=[str(emb.device) for emb in memory_embeddings],
                target_device=str(self.device),
                memory_embeddings_count=len(memory_embeddings)
            )

            base_conf = self._get_base_confidence(state)
            scaf_conf = self._get_scaffold_confidence(state)

            score = self.curiosity.compute_curiosity(
                base_conf=base_conf,
                scaf_conf=scaf_conf,
                memory_embeddings=memory_embeddings,
                query_embedding=query_embedding,
                device=self.device
            )

            self._update_state_novelty(state, score)
            self.callbacks.trigger_callback("curiosity_computed", score=score)
            return score
            
        except Exception as e:
            error_msg = f"Curiosity computation failed: {str(e)}"
            self._log_error(error_msg)
            raise CuriosityError(error_msg) from e
            
    def _validate_state(self, state: Any) -> None:
        """Validate state attributes required for curiosity computation."""
        if state is None:
            raise StateError("State is None")
            
        required_attributes = [
            'confidence_history',
            'sleep_confidence_sum',
            'sleep_confidence_count',
            'dream_memory',
            'last_prompt_embedding'
        ]
        
        missing_attributes = [attr for attr in required_attributes 
                            if not hasattr(state, attr)]
        if missing_attributes:
            raise StateError(f"Missing required state attributes: {missing_attributes}")
            
        # Validate data structures
        if not isinstance(state.confidence_history, deque):
            raise StateError("State confidence_history must be a deque")
            
        if not isinstance(state.dream_memory, deque):
            raise StateError("State dream_memory must be a deque")
            
        # Validate values
        if not isinstance(state.sleep_confidence_sum, (int, float)):
            raise StateError("State sleep_confidence_sum must be numeric")
            
        if not isinstance(state.sleep_confidence_count, int):
            raise StateError("State sleep_confidence_count must be an integer")
            
    def _get_base_confidence(self, state: Any) -> float:
        """Extract base confidence from state."""
        return state.confidence_history[-1] if state.confidence_history else 0.5

    def _get_scaffold_confidence(self, state: Any) -> float:
        """Extract scaffold confidence from state."""
        return (
            state.sleep_confidence_sum / state.sleep_confidence_count
            if state.sleep_confidence_count > 0 else 0.5
        )

    def generate_question(
        self,
        state: Any,
        tokenizer: Any,
        model: Any,
        prompt: Optional[str] = None,
        spontaneous: bool = False
    ) -> Optional[str]:
        """Generate a curiosity-driven question if conditions are met."""
        try:
            # Store state metadata for logging
            self.conversation_id = getattr(state, 'conversation_id', None)
            self.state_hash = getattr(state, 'state_hash', None)
            
            # Validate state
            self._validate_state(state)
            
            if not self._can_generate_question(tokenizer, model):
                return None

            current_time = time.time()
            if not self._check_cooldown(current_time):
                return None

            curiosity_score = self.compute_curiosity(state, tokenizer, model, prompt)
            if not self._should_generate_question(curiosity_score, prompt, spontaneous):
                return None

            question = self._generate_question_text(state, tokenizer, model, prompt)
            if question:
                self._update_question_state(state, question, curiosity_score, current_time)
            return question
            
        except StateError as e:
            self._log_error(f"State validation failed: {str(e)}", error_type="state_error")
            return None
        except Exception as e:
            self._log_error(f"Question generation failed: {str(e)}")
            return None

    def _can_generate_question(self, tokenizer: Any, model: Any) -> bool:
        """Check if question generation is possible."""
        if not (tokenizer and model):
            self._log_warning("Missing tokenizer or model for question generation")
            return False
        return True

    def _check_cooldown(self, current_time: float) -> bool:
        """Check if question generation cooldown has elapsed."""
        return (current_time - self.last_question_time) >= self.config_manager.config["question_cooldown"]

    def _should_generate_question(self, curiosity_score: float, prompt: Optional[str], spontaneous: bool) -> bool:
        """Determine if a question should be generated."""
        threshold = (
            self.config_manager.config["novelty_threshold_spontaneous"] if spontaneous
            else self.config_manager.config["novelty_threshold_response"]
        )
        return (
            (spontaneous and curiosity_score >= threshold) or
            (not spontaneous and prompt and curiosity_score >= threshold) or
            self.should_erupt(state)
        )

    def _generate_question_text(
        self, state: Any, tokenizer: Any, model: Any, prompt: Optional[str]
    ) -> Optional[str]:
        """Generate question text using model and tokenizer."""
        try:
            base_prompt = self._get_base_prompt(state, prompt)
            inputs = tokenizer(
                base_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config_manager.config["max_new_tokens"],
                    temperature=max(0.1, self.config_manager.config["base_temperature"] + 
                                  self.config_manager.config["temperament_influence"] * getattr(state, 'temperament_score', 0.5)),
                    top_k=self.config_manager.config["top_k"],
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            question = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            if not question:
                # Fallback to default question if generation fails
                question = "Can you tell me more about that?"
                self._log_warning("Question generation failed, using fallback")
            
            return question
            
        except Exception as e:
            error_msg = f"Failed to generate question text: {str(e)}"
            self._log_error(error_msg)
            raise CuriosityError(error_msg) from e

    def _get_base_prompt(self, state: Any, prompt: Optional[str]) -> str:
        """Select appropriate base prompt for question generation."""
        return (
            prompt if prompt and isinstance(prompt, str)
            else getattr(getattr(state, 'history', None), 'messages', [{}])[-1].get('prompt', "What is this about?")
        )

    def _update_question_state(self, state: Any, question: str, curiosity_score: float, current_time: float) -> None:
        """Update state after question generation using CuriosityState methods."""
        try:
            if not hasattr(state, 'curiosity'):
                raise StateError("State missing curiosity component")
                
            # Use CuriosityState's add_question method for thread-safe updates
            state.curiosity.add_question(
                question=question,
                score=curiosity_score,
                context_vector=None  # Optional: add context vector if available
            )
            
            self.last_question_time = current_time
            self.pressure_mgr.value = state.curiosity.pressure
            
            # Log question addition with standardized metadata
            self._log_event("question_added", {
                "question": question,
                "score": curiosity_score,
                "pressure": self.pressure_mgr.value,
                "queue_length": len(state.curiosity.unanswered_questions),
                "conversation_id": getattr(state, 'history', {}).get('conversation_id', None),
                "state_hash": getattr(state, 'state_hash', None)
            })
            
            # Trigger callbacks
            self.callbacks.trigger_callback("question_generated", question=question, score=curiosity_score)
            
        except Exception as e:
            self._log_error(f"Failed to update question state: {str(e)}")
            raise
            
    def _update_state_novelty(self, state: Any, score: float) -> None:
        """Update state novelty scores if available."""
        if hasattr(state, 'curiosity') and hasattr(state.curiosity, 'novelty_scores'):
            state.curiosity.novelty_scores.append(score)

    def tune(
        self,
        enable: Optional[bool] = None,
        spontaneous_threshold: Optional[float] = None,
        response_threshold: Optional[float] = None,
        pressure_threshold: Optional[float] = None,
        pressure_drop: Optional[float] = None,
        silence_threshold: Optional[float] = None,
        question_cooldown: Optional[float] = None,
        queue_maxlen: Optional[int] = None,
        weight_ignorance: Optional[float] = None,
        weight_novelty: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        base_temperature: Optional[float] = None,
        temperament_influence: Optional[float] = None,
        metrics_maxlen: Optional[int] = None,
        top_k: Optional[int] = None
    ) -> None:
        """Tune curiosity parameters."""
        updates = self._validate_tune_params(locals())
        self._apply_tune_updates(updates)

    def _validate_tune_params(self, params: Dict) -> Dict:
        """Validate tuning parameters."""
        updates = {}
        param_constraints = {
            "enable": lambda x: isinstance(x, bool),
            "spontaneous_threshold": lambda x: 0.5 <= x <= 1.0,
            "response_threshold": lambda x: 0.5 <= x <= 1.0,
            "pressure_threshold": lambda x: 0.5 <= x <= 0.9,
            "pressure_drop": lambda x: 0.1 <= x <= 0.5,
            "silence_threshold": lambda x: 5.0 <= x <= 60.0,
            "question_cooldown": lambda x: 30.0 <= x <= 120.0,
            "queue_maxlen": lambda x: 5 <= x <= 20,
            "weight_ignorance": lambda x: 0.0 <= x <= 1.0,
            "weight_novelty": lambda x: 0.0 <= x <= 1.0,
            "max_new_tokens": lambda x: 5 <= x <= 12,
            "base_temperature": lambda x: 0.5 <= x <= 1.5,
            "temperament_influence": lambda x: 0.1 <= x <= 0.6,
            "metrics_maxlen": lambda x: 100 <= x <= 10000,
            "top_k": lambda x: 10 <= x <= 50
        }

        for key, validator in param_constraints.items():
            value = params.get(key)
            if value is not None and validator(value):
                updates[key] = value
        return updates

    def _apply_tune_updates(self, updates: Dict) -> None:
        """Apply validated tuning updates."""
        try:
            if "queue_maxlen" in updates:
                self.last_question_time = time.time()
            if "metrics_maxlen" in updates:
                self.metrics = deque(self.metrics, maxlen=updates["metrics_maxlen"])
            if "weight_ignorance" in updates:
                self.curiosity.weight_ignorance = updates["weight_ignorance"]
            if "weight_novelty" in updates:
                self.curiosity.weight_novelty = updates["weight_novelty"]

            self.config_manager.config.update(updates)
            if updates:
                self._log_event("curiosity_tune", {
                    "params": updates,
                    "timestamp": time.time()
                })
                
        except Exception as e:
            self._log_error(f"Failed to apply tune updates: {str(e)}")
            raise

    def generate_curiosity_question(
        self, state: Any, tokenizer: Any, model: Any, context: Optional[str] = None, spontaneous: bool = False
    ) -> Optional[str]:
        """Generate a curiosity-driven question."""
        try:
            if not self.config_manager.config.get("enable_curiosity", True):
                return None
                
            # Validate state and curiosity component
            self._validate_state(state)
            if not hasattr(state, 'curiosity'):
                raise StateError("State missing curiosity component")
                
            # Generate question using state's curiosity component
            question = self.generate_question(state, tokenizer, model, context, spontaneous)
            
            if question:
                # Log question generation
                self._log_event("curiosity_question", {
                    "question": question,
                    "spontaneous": spontaneous,
                    "pressure": state.curiosity.pressure,
                    "queue_length": len(state.curiosity.unanswered_questions)
                })
                
            return question
            
        except Exception as e:
            self._log_error(f"Failed to generate curiosity question: {str(e)}")
            return None
