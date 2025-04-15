from typing import Optional, Dict, Any, List, Protocol
import time
from sovl_config import ConfigManager
from sovl_logger import Logger
from sovl_utils import NumericalGuard
from sovl_curiosity import CuriosityManager
from sovl_trainer import SOVLTrainer
from sovl_scaffold import CrossAttentionInjector
from sovl_validation import ValidationSchema
import traceback
from collections import defaultdict, deque
from sovl_error import ErrorManager

class ICuriosityManager(Protocol):
    """Interface for curiosity management."""
    def get_pressure(self) -> float: ...
    def reduce_pressure(self, amount: float) -> None: ...

class ITrainer(Protocol):
    """Interface for model training."""
    def train_step(self, batch: Dict[str, Any]) -> float: ...
    def get_current_parameters(self) -> Dict[str, Any]: ...

class ICrossAttentionInjector(Protocol):
    """Interface for cross attention injection."""
    def inject_cross_attention(self, model: Any, scaffold_model: Any, **kwargs) -> None: ...

class SOVLTuner:
    """Centralized module for tuning SOVL system parameters dynamically."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        error_manager: Optional[ErrorManager] = None,
        curiosity_manager: Optional[ICuriosityManager] = None,
        trainer: Optional[ITrainer] = None,
        cross_attention_injector: Optional[ICrossAttentionInjector] = None
    ):
        self.config_manager = config_manager
        self.logger = logger
        self.error_manager = error_manager
        self.curiosity_manager = curiosity_manager
        self.trainer = trainer
        self.cross_attention_injector = cross_attention_injector
        self._guard = NumericalGuard(logger)
        
        # Cache configuration sections
        self.core_config = config_manager.get_section("core_config")
        self.controls_config = config_manager.get_section("controls_config")
        self.training_config = config_manager.get_section("training_config")
        self.curiosity_config = config_manager.get_section("curiosity_config")
        self.cross_attn_config = config_manager.get_section("cross_attn_config")
        self.lora_config = config_manager.get_section("lora_config")
        
        # Error handling state
        self._last_error_time = 0.0
        self._error_cooldown = 1.0  # seconds
        self._error_counts = defaultdict(int)
        
        # Confidence monitoring
        self._confidence_history = deque(maxlen=100)
        self._last_confidence_check = 0.0
        self._confidence_check_interval = 60.0  # seconds
        
    def _handle_error(self, error: Exception, error_type: str, context: Dict[str, Any]) -> None:
        """Handle errors with coordination between ErrorManager and local handling."""
        try:
            current_time = time.time()
            error_key = f"{error_type}:{type(error).__name__}"
            
            # Check for duplicate errors within cooldown period
            if current_time - self._last_error_time < self._error_cooldown:
                self.logger.record_event(
                    event_type="duplicate_error",
                    message=f"Duplicate error detected within cooldown period: {error_key}",
                    level="warning",
                    additional_info={
                        "error_type": error_type,
                        "error_key": error_key,
                        "context": context
                    }
                )
                return
                
            self._last_error_time = current_time
            self._error_counts[error_key] += 1
            
            # Log error locally
            self.logger.record_event(
                event_type="tuner_error",
                message=f"Error in {error_type}: {str(error)}",
                level="error",
                additional_info={
                    "error_type": error_type,
                    "error_key": error_key,
                    "error_count": self._error_counts[error_key],
                    "context": context
                }
            )
            
            # Delegate to ErrorManager if available
            if self.error_manager:
                if error_type == "training":
                    self.error_manager.handle_training_error(error, context.get("batch_size", 1))
                elif error_type == "curiosity":
                    self.error_manager.handle_curiosity_error(error, context.get("event_type", "unknown"))
                elif error_type == "memory":
                    self.error_manager.handle_memory_error(error, context.get("model_size", 0))
                else:
                    self.error_manager.handle_generation_error(error, context.get("prompt", ""))
                    
        except Exception as e:
            # Fallback logging if error handling fails
            self.logger.record_event(
                event_type="error_handling_failed",
                message=f"Failed to handle error: {str(e)}",
                level="critical",
                additional_info={
                    "original_error": str(error),
                    "error_type": error_type,
                    "context": context
                }
            )
            
    def validate_param(self, param_name: str, value: Any) -> bool:
        """Validate a parameter against its allowed range or type."""
        try:
            # Extract section and key from param_name
            section, key = param_name.split(".", 1)
            
            # Use shared validation schema
            is_valid, error_msg = ValidationSchema.validate_value(
                section, key, value, self.logger
            )
            
            if not is_valid:
                self.logger.record_event(
                    event_type="param_validation_error",
                    message=f"Invalid parameter value: {error_msg}",
                    level="error",
                    additional_info={
                        "param_name": param_name,
                        "value": value
                    }
                )
            
            return is_valid
            
        except ValueError as e:
            self.logger.record_event(
                event_type="param_validation_error",
                message=f"Failed to validate parameter: {str(e)}",
                level="error",
                additional_info={
                    "param_name": param_name,
                    "value": value
                }
            )
            return False
    
    def _monitor_confidence(self, confidence: float) -> None:
        """Monitor confidence scores and adjust parameters if needed."""
        try:
            current_time = time.time()
            if current_time - self._last_confidence_check < self._confidence_check_interval:
                return
                
            self._last_confidence_check = current_time
            self._confidence_history.append(confidence)
            
            if len(self._confidence_history) < 10:
                return
                
            # Calculate confidence statistics
            confidences = list(self._confidence_history)
            mean_conf = sum(confidences) / len(confidences)
            variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
            
            # Check for confidence issues
            if variance > 0.1:  # High variance in confidence
                self.logger.record_event(
                    event_type="confidence_variance_high",
                    message="High variance detected in confidence scores",
                    level="warning",
                    additional_info={
                        "variance": variance,
                        "mean_confidence": mean_conf,
                        "history_length": len(confidences)
                    }
                )
                
                # Adjust temperament influence to stabilize confidence
                current_influence = self.curiosity_config.get("temperament_influence", 0.3)
                new_influence = min(current_influence + 0.1, 0.6)
                
                if new_influence != current_influence:
                    self.tune_curiosity(temperament_influence=new_influence)
                    self.logger.record_event(
                        event_type="temperament_influence_adjusted",
                        message="Adjusted temperament influence to stabilize confidence",
                        level="info",
                        additional_info={
                            "old_influence": current_influence,
                            "new_influence": new_influence,
                            "variance": variance
                        }
                    )
                    
            elif mean_conf < 0.3:  # Consistently low confidence
                self.logger.record_event(
                    event_type="confidence_low",
                    message="Consistently low confidence detected",
                    level="warning",
                    additional_info={
                        "mean_confidence": mean_conf,
                        "history_length": len(confidences)
                    }
                )
                
                # Reduce curiosity pressure to allow for more stable generation
                if self.curiosity_manager:
                    current_pressure = self.curiosity_manager.get_pressure()
                    if current_pressure > 0.3:
                        self.curiosity_manager.reduce_pressure(0.1)
                        self.logger.record_event(
                            event_type="curiosity_pressure_reduced",
                            message="Reduced curiosity pressure to improve confidence",
                            level="info",
                            additional_info={
                                "old_pressure": current_pressure,
                                "new_pressure": self.curiosity_manager.get_pressure(),
                                "mean_confidence": mean_conf
                            }
                        )
                        
        except Exception as e:
            self.logger.record_event(
                event_type="confidence_monitoring_error",
                message=f"Error in confidence monitoring: {str(e)}",
                level="error",
                additional_info={"error": str(e)}
            )
            
    def tune_parameters(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Tune system parameters based on performance metrics."""
        try:
            # Get current parameters
            current_params = self.trainer.get_current_parameters() if self.trainer else {}
            
            # Monitor confidence if available
            if "confidence" in metrics:
                self._monitor_confidence(metrics["confidence"])
            
            # Adjust parameters based on metrics
            tuned_params = self._adjust_parameters(current_params, metrics)
            
            # Apply parameter changes
            if self.trainer:
                self.trainer.update_parameters(tuned_params)
            
            # Log tuning results
            self.logger.record({
                "event": "parameters_tuned",
                "previous_params": current_params,
                "new_params": tuned_params,
                "metrics": metrics,
                "timestamp": time.time()
            })
            
            return tuned_params
            
        except Exception as e:
            self._handle_error(
                error=e,
                error_type="training",
                context={
                    "metrics": metrics,
                    "batch_size": metrics.get("batch_size", 1)
                }
            )
            raise
            
    def _adjust_parameters(self, current_params: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust parameters based on metrics with error handling."""
        try:
            tuned_params = current_params.copy()
            
            # Adjust learning rate based on loss
            if "loss" in metrics:
                loss = metrics["loss"]
                if loss > self.config_manager.get("loss_threshold", 2.0):
                    tuned_params["learning_rate"] *= 0.5
                elif loss < self.config_manager.get("loss_target", 0.5):
                    tuned_params["learning_rate"] *= 1.1
                    
            # Adjust curiosity pressure based on performance
            if self.curiosity_manager and "performance" in metrics:
                performance = metrics["performance"]
                if performance < self.config_manager.get("performance_threshold", 0.7):
                    # Get current pressure stats
                    pressure_stats = self.curiosity_manager.get_pressure_stats()
                    
                    # Calculate reduction amount based on performance gap
                    performance_gap = self.config_manager.get("performance_threshold", 0.7) - performance
                    reduction_amount = min(0.1, performance_gap * 0.2)  # Cap reduction at 0.1
                    
                    # Only reduce pressure if it's above minimum
                    if pressure_stats["current_pressure"] > pressure_stats["min_pressure"]:
                        self.curiosity_manager.reduce_pressure(reduction_amount)
                        
                        # Log pressure adjustment
                        self.logger.record_event(
                            event_type="pressure_adjusted",
                            message="Adjusted curiosity pressure based on performance",
                            level="info",
                            additional_info={
                                "performance": performance,
                                "performance_threshold": self.config_manager.get("performance_threshold", 0.7),
                                "reduction_amount": reduction_amount,
                                "pressure_stats": pressure_stats
                            }
                        )
                    
            return tuned_params
            
        except Exception as e:
            self._handle_error(
                error=e,
                error_type="parameter_adjustment",
                context={
                    "current_params": current_params,
                    "metrics": metrics
                }
            )
            raise
    
    def tune_curiosity(
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
        top_k: Optional[int] = None,
        attention_weight: Optional[float] = None,
        question_timeout: Optional[float] = None
    ) -> bool:
        """Tune curiosity-related parameters with transactional safety."""
        updates = {}
        prefix = "curiosity_config."
        params = {
            "enable_curiosity": enable,
            "novelty_threshold_spontaneous": spontaneous_threshold,
            "novelty_threshold_response": response_threshold,
            "pressure_threshold": pressure_threshold,
            "pressure_drop": pressure_drop,
            "silence_threshold": silence_threshold,
            "question_cooldown": question_cooldown,
            "queue_maxlen": queue_maxlen,
            "weight_ignorance": weight_ignorance,
            "weight_novelty": weight_novelty,
            "max_new_tokens": max_new_tokens,
            "base_temperature": base_temperature,
            "temperament_influence": temperament_influence,
            "top_k": top_k,
            "attention_weight": attention_weight,
            "question_timeout": question_timeout
        }
        
        # Validate all parameters before attempting updates
        for param, value in params.items():
            if value is not None:
                full_key = f"{prefix}{param}"
                if not self.validate_param(full_key, value):
                    self.logger.record_event(
                        event_type="curiosity_tuning_validation_failed",
                        message=f"Validation failed for parameter: {param}",
                        level="error",
                        additional_info={
                            "param": param,
                            "value": value
                        }
                    )
                    return False
                updates[full_key] = value
        
        if not updates:
            return True
            
        # Attempt batch update with rollback
        success = self.config_manager.update_batch(updates, rollback_on_failure=True)
        
        if success:
            # Update cached config
            for key, value in updates.items():
                param = key.split(".")[-1]
                self.curiosity_config[param] = value
            
            # Notify curiosity manager with error handling
            if self.curiosity_manager:
                try:
                    # Convert config keys to manager parameter names
                    manager_params = {
                        k.split(".")[-1]: v for k, v in updates.items()
                    }
                    self.curiosity_manager.tune(**manager_params)
                except Exception as e:
                    self.logger.record_event(
                        event_type="curiosity_manager_tune_failed",
                        message=f"Failed to update CuriosityManager: {str(e)}",
                        level="error",
                        additional_info={
                            "error": str(e),
                            "stack_trace": traceback.format_exc()
                        }
                    )
                    # Don't return False here as config update was successful
                    # Just log the error and continue
            
            self.logger.record_event(
                event_type="curiosity_tuning_success",
                message="Successfully tuned curiosity parameters",
                level="info",
                additional_info={
                    "updated_params": list(updates.keys())
                }
            )
        else:
            self.logger.record_event(
                event_type="curiosity_tuning_failed",
                message="Failed to tune curiosity parameters",
                level="error",
                additional_info={
                    "attempted_params": list(updates.keys())
                }
            )
            
        return success
    
    def adjust_temperament(
        self,
        eager_threshold: Optional[float] = None,
        sluggish_threshold: Optional[float] = None,
        mood_influence: Optional[float] = None,
        curiosity_boost: Optional[float] = None,
        restless_drop: Optional[float] = None,
        melancholy_noise: Optional[float] = None,
        conf_feedback_strength: Optional[float] = None,
        temp_smoothing_factor: Optional[float] = None,
        decay_rate: Optional[float] = None
    ) -> bool:
        """Tune temperament-related parameters with safe ranges and validation."""
        try:
            # Define safe parameter ranges
            safe_ranges = {
                "temp_smoothing_factor": (0.1, 1.0),
                "temp_eager_threshold": (0.5, 0.9),
                "temp_sluggish_threshold": (0.1, 0.5),
                "temp_mood_influence": (0.1, 0.9),
                "temp_curiosity_boost": (0.1, 0.5),
                "temp_restless_drop": (0.1, 0.5),
                "temp_melancholy_noise": (0.0, 0.2),
                "conf_feedback_strength": (0.1, 0.9),
                "temperament_decay_rate": (0.1, 0.9)
            }
            
            # Prepare updates with validation
            updates = {}
            params = {
                "temp_eager_threshold": eager_threshold,
                "temp_sluggish_threshold": sluggish_threshold,
                "temp_mood_influence": mood_influence,
                "temp_curiosity_boost": curiosity_boost,
                "temp_restless_drop": restless_drop,
                "temp_melancholy_noise": melancholy_noise,
                "conf_feedback_strength": conf_feedback_strength,
                "temp_smoothing_factor": temp_smoothing_factor,
                "temperament_decay_rate": decay_rate
            }
            
            # Validate and prepare updates
            for key, value in params.items():
                if value is not None:
                    min_val, max_val = safe_ranges[key]
                    if not (min_val <= value <= max_val):
                        self.logger.record_event(
                            event_type="temperament_parameter_warning",
                            message=f"Parameter {key} out of safe range, clamping to bounds",
                            level="warning",
                            additional_info={
                                "parameter": key,
                                "value": value,
                                "min": min_val,
                                "max": max_val
                            }
                        )
                        value = max(min_val, min(value, max_val))
                    updates[f"controls_config.{key}"] = value
            
            if not updates:
                return True
                
            # Apply updates with rollback
            success = self.config_manager.update_batch(updates, rollback_on_failure=True)
            
            if success:
                # Update cached config
                for key, value in updates.items():
                    param = key.split(".")[-1]
                    self.controls_config[param] = value
                
                # Reset temperament history if trainer is available
                if self.trainer and hasattr(self.trainer, 'state'):
                    with self.trainer.state.lock:
                        self.trainer.state.temperament_history.clear()
                        self.trainer.state.temperament_score = 0.5  # Reset to neutral
                        self.logger.record_event(
                            event_type="temperament_history_reset",
                            message="Temperament history reset after parameter update",
                            level="info",
                            additional_info={
                                "reason": "temperament parameter update",
                                "updated_params": list(updates.keys())
                            }
                        )
                
                # Log successful update
                self.logger.record_event(
                    event_type="temperament_parameters_updated",
                    message="Temperament parameters updated successfully",
                    level="info",
                    additional_info={
                        "updates": updates,
                        "history_reset": self.trainer is not None
                    }
                )
            else:
                self.logger.record_event(
                    event_type="temperament_update_failed",
                    message="Failed to update temperament parameters",
                    level="error",
                    additional_info=updates
                )
                
            return success
            
        except Exception as e:
            self.logger.record_event(
                event_type="temperament_adjustment_error",
                message=f"Failed to adjust temperament: {str(e)}",
                level="error",
                additional_info={"error": str(e)}
            )
            return False
    
    def tune_dream(
        self,
        swing_var: Optional[float] = None,
        lifecycle_delta: Optional[float] = None,
        temperament_on: Optional[bool] = None,
        noise_scale: Optional[float] = None,
        memory_weight: Optional[float] = None,
        memory_maxlen: Optional[int] = None,
        prompt_weight: Optional[float] = None,
        novelty_boost: Optional[float] = None,
        memory_decay: Optional[float] = None,
        prune_threshold: Optional[float] = None
    ) -> bool:
        """Tune dreaming-related parameters."""
        updates = {}
        prefix = "controls_config."
        params = {
            "dream_swing_var": swing_var,
            "dream_lifecycle_delta": lifecycle_delta,
            "dream_temperament_on": temperament_on,
            "dream_noise_scale": noise_scale,
            "dream_memory_weight": memory_weight,
            "dream_memory_maxlen": memory_maxlen,
            "dream_prompt_weight": prompt_weight,
            "dream_novelty_boost": novelty_boost,
            "dream_memory_decay": memory_decay,
            "dream_prune_threshold": prune_threshold
        }
        
        for param, value in params.items():
            if value is not None:
                full_key = f"{prefix}{param}"
                if self.validate_param(full_key, value):
                    updates[full_key] = value
                else:
                    return False
        
        if updates:
            success = self.config_manager.update_batch(updates)
            if success:
                # Update cached config
                for key, value in updates.items():
                    param = key.split(".")[-1]
                    self.controls_config[param] = value
                
                # Notify trainer if dreaming parameters are used
                if self.trainer and "controls_config.dream_memory_weight" in updates:
                    self.trainer.config.dream_memory_weight = updates["controls_config.dream_memory_weight"]
                
                self.config_manager.save_config()
                self.logger.record({
                    "event": "tune_dream",
                    "params": updates,
                    "success": success,
                    "timestamp": time.time()
                })
            return success
        return True
    
    def set_sleep_params(
        self,
        conf_threshold: Optional[float] = None,
        time_factor: Optional[float] = None,
        log_min: Optional[int] = None,
        max_steps: Optional[int] = None
    ) -> bool:
        """Tune sleep-related parameters."""
        updates = {}
        params = {
            "controls_config.sleep_conf_threshold": conf_threshold,
            "controls_config.sleep_time_factor": time_factor,
            "controls_config.sleep_log_min": log_min,
            "training_config.sleep_max_steps": max_steps
        }
        
        for param, value in params.items():
            if value is not None:
                if self.validate_param(param, value):
                    updates[param] = value
                else:
                    return False
        
        if updates:
            success = self.config_manager.update_batch(updates)
            if success:
                # Update cached config
                for key, value in updates.items():
                    section, param = key.split(".", 1)
                    getattr(self, f"{section}_config")[param] = value
                
                # Notify trainer
                if self.trainer:
                    if "controls_config.sleep_conf_threshold" in updates:
                        self.trainer.config.sleep_conf_threshold = updates["controls_config.sleep_conf_threshold"]
                    if "controls_config.sleep_log_min" in updates:
                        self.trainer.config.sleep_log_min = updates["controls_config.sleep_log_min"]
                    if "training_config.sleep_max_steps" in updates:
                        self.trainer.config.sleep_max_steps = updates["training_config.sleep_max_steps"]
                
                self.config_manager.save_config()
                self.logger.record({
                    "event": "set_sleep_params",
                    "params": updates,
                    "success": success,
                    "timestamp": time.time()
                })
            return success
        return True
    
    def set_global_blend(
        self,
        weight_cap: Optional[float] = None,
        base_temp: Optional[float] = None
    ) -> bool:
        """Tune global blend parameters."""
        updates = {}
        prefix = "controls_config."
        params = {
            "scaffold_weight_cap": weight_cap,
            "base_temperature": base_temp
        }
        
        for param, value in params.items():
            if value is not None:
                full_key = f"{prefix}{param}"
                if self.validate_param(full_key, value):
                    updates[full_key] = value
                else:
                    return False
        
        if updates:
            success = self.config_manager.update_batch(updates)
            if success:
                # Update cached config
                for key, value in updates.items():
                    param = key.split(".")[-1]
                    self.controls_config[param] = value
                
                self.config_manager.save_config()
                self.logger.record({
                    "event": "set_global_blend",
                    "params": updates,
                    "success": success,
                    "timestamp": time.time()
                })
            return success
        return True
    
    def tune_lifecycle(
        self,
        capacity_factor: Optional[float] = None,
        curve: Optional[str] = None,
        lora_capacity: Optional[int] = None
    ) -> bool:
        """Tune lifecycle-related parameters."""
        updates = {}
        prefix = "training_config."
        params = {
            "lifecycle_capacity_factor": capacity_factor,
            "lifecycle_curve": curve,
            "lora_capacity": lora_capacity
        }
        
        for param, value in params.items():
            if value is not None:
                full_key = f"{prefix}{param}"
                if self.validate_param(full_key, value):
                    updates[full_key] = value
                else:
                    return False
        
        if updates:
            success = self.config_manager.update_batch(updates)
            if success:
                # Update cached config
                for key, value in updates.items():
                    param = key.split(".")[-1]
                    self.training_config[param] = value
                
                # Notify trainer
                if self.trainer:
                    if "training_config.lifecycle_capacity_factor" in updates:
                        self.trainer.config.lifecycle_capacity_factor = updates["training_config.lifecycle_capacity_factor"]
                    if "training_config.lifecycle_curve" in updates:
                        self.trainer.config.lifecycle_curve = updates["training_config.lifecycle_curve"]
                    if "training_config.lora_capacity" in updates:
                        self.trainer.lora_capacity = updates["training_config.lora_capacity"]
                
                self.config_manager.save_config()
                self.logger.record({
                    "event": "tune_lifecycle",
                    "params": updates,
                    "success": success,
                    "timestamp": time.time()
                })
            return success
        return True
    
    def toggle_dynamic_layers(self, enable: bool) -> bool:
        """Toggle dynamic layer usage."""
        if self.validate_param("core_config.use_dynamic_layers", enable):
            success = self.config_manager.update("core_config.use_dynamic_layers", enable)
            if success:
                self.core_config["use_dynamic_layers"] = enable
                self.config_manager.save_config()
                self.logger.record({
                    "event": "toggle_dynamic_layers",
                    "enable": enable,
                    "success": success,
                    "timestamp": time.time()
                })
            return success
        return False
    
    def set_quantization_mode(self, mode: str) -> bool:
        """Set quantization mode."""
        if self.validate_param("core_config.quantization", mode):
            success = self.config_manager.update("core_config.quantization", mode)
            if success:
                self.core_config["quantization"] = mode
                self.config_manager.save_config()
                self.logger.record({
                    "event": "set_quantization_mode",
                    "mode": mode,
                    "success": success,
                    "timestamp": time.time()
                })
            return success
        return False
    
    def set_scaffold_influence(
        self,
        weight: Optional[float] = None,
        blend_strength: Optional[float] = None,
        layer_weights: Optional[List[float]] = None,
        base_model: Optional[Any] = None
    ) -> bool:
        """Set scaffold influence for cross-attention layers."""
        if not self.cross_attention_injector or not base_model:
            self.logger.record({
                "error": "CrossAttentionInjector or base_model not provided",
                "timestamp": time.time()
            })
            return False
        
        # Validate layer_weights length if provided
        if layer_weights is not None:
            try:
                # Get current cross-attention layers
                layers = self.cross_attention_injector.get_cross_attention_layers(
                    base_model,
                    mode=self.core_config.get("layer_selection_mode", "balanced")
                )
                
                # Validate layer count matches weights
                if len(layer_weights) != len(layers):
                    self.logger.record_event(
                        event_type="scaffold_influence_error",
                        message="Layer weights length mismatch",
                        level="error",
                        additional_info={
                            "expected_layers": len(layers),
                            "provided_weights": len(layer_weights),
                            "layers": layers,
                            "weights": layer_weights
                        }
                    )
                    return False
                    
                # Validate weight values
                for i, weight in enumerate(layer_weights):
                    if not (0.0 <= weight <= 1.0):
                        self.logger.record_event(
                            event_type="scaffold_influence_error",
                            message="Invalid layer weight value",
                            level="error",
                            additional_info={
                                "layer_index": i,
                                "weight": weight,
                                "valid_range": (0.0, 1.0)
                            }
                        )
                        return False
                        
                # Log successful validation
                self.logger.record_event(
                    event_type="scaffold_influence_validated",
                    message="Layer weights validated successfully",
                    level="info",
                    additional_info={
                        "layer_count": len(layers),
                        "layer_weights": layer_weights,
                        "layer_selection_mode": self.core_config.get("layer_selection_mode", "balanced")
                    }
                )
                
            except Exception as e:
                self.logger.record_event(
                    event_type="scaffold_influence_error",
                    message=f"Failed to validate layer weights: {str(e)}",
                    level="error",
                    additional_info={"error": str(e)}
                )
                return False
        
        try:
            self.cross_attention_injector.set_influence(
                model=base_model,
                core_config=self.core_config,
                cross_attn_config=self.cross_attn_config,
                training_config=self.training_config,
                controls_config=self.controls_config,
                weight=weight,
                blend_strength=blend_strength,
                layer_weights=layer_weights
            )
            
            # Log successful influence update
            self.logger.record_event(
                event_type="scaffold_influence_updated",
                message="Scaffold influence updated successfully",
                level="info",
                additional_info={
                    "weight": weight,
                    "blend_strength": blend_strength,
                    "layer_weights": layer_weights,
                    "layer_count": len(layers) if layer_weights else None
                }
            )
            return True
            
        except Exception as e:
            self.logger.record_event(
                event_type="scaffold_influence_error",
                message=f"Failed to set scaffold influence: {str(e)}",
                level="error",
                additional_info={"error": str(e)}
            )
            return False
    
    def tune_cross_attention(
        self,
        weight: Optional[float] = None,
        blend_strength: Optional[float] = None,
        layer_weights: Optional[List[float]] = None,
        dynamic_mode: Optional[str] = None,
        base_model: Optional[Any] = None
    ) -> bool:
        """Tune cross-attention settings."""
        success = True
        
        # Update scaffold influence
        if any(param is not None for param in [weight, blend_strength, layer_weights]):
            success &= self.set_scaffold_influence(weight, blend_strength, layer_weights, base_model)
        
        # Update dynamic mode
        if dynamic_mode is not None:
            full_key = "controls_config.dynamic_cross_attn_mode"
            validated_mode = dynamic_mode if dynamic_mode != "off" else None
            if self.validate_param(full_key, dynamic_mode):
                success &= self.config_manager.update(full_key, validated_mode)
                if success:
                    self.controls_config["dynamic_cross_attn_mode"] = validated_mode
                    self.config_manager.save_config()
                    self.logger.record({
                        "event": "tune_cross_attention",
                        "dynamic_mode": dynamic_mode,
                        "success": True,
                        "timestamp": time.time()
                    })
            else:
                success = False
        
        return success
    
    def toggle_memory(
        self,
        mode: str,
        use_scaffold_memory: Optional[bool] = None,
        use_token_map_memory: Optional[bool] = None
    ) -> bool:
        """Toggle memory usage modes."""
        modes = {
            'scaffold_mem': (True, False),
            'token_mem': (False, True),
            'both_mem': (True, True),
            'no_mem': (False, False)
        }
        
        if mode not in modes and (use_scaffold_memory is None or use_token_map_memory is None):
            self.logger.record({
                "error": f"Invalid memory mode: {mode}. Use: {', '.join(modes.keys())} or specify use_scaffold_memory/use_token_map_memory",
                "timestamp": time.time()
            })
            return False
        
        scaffold_mem, token_mem = modes.get(mode, (use_scaffold_memory, use_token_map_memory))
        
        updates = {
            "controls_config.use_scaffold_memory": scaffold_mem,
            "controls_config.use_token_map_memory": token_mem
        }
        
        success = self.config_manager.update_batch(updates)
        if success:
            self.controls_config["use_scaffold_memory"] = scaffold_mem
            self.controls_config["use_token_map_memory"] = token_mem
            self.config_manager.save_config()
            self.logger.record({
                "event": "toggle_memory",
                "mode": mode if mode in modes else "custom",
                "scaffold_memory": scaffold_mem,
                "token_map_memory": token_mem,
                "success": success,
                "timestamp": time.time()
            })
        return success
    
    def update_component_references(
        self,
        curiosity_manager: Optional[ICuriosityManager] = None,
        trainer: Optional[ITrainer] = None,
        cross_attention_injector: Optional[ICrossAttentionInjector] = None
    ) -> None:
        """Update references to dependent components."""
        if curiosity_manager is not None:
            self.curiosity_manager = curiosity_manager
        if trainer is not None:
            self.trainer = trainer
        if cross_attention_injector is not None:
            self.cross_attention_injector = cross_attention_injector
