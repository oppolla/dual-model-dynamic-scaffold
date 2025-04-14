from typing import Optional, Dict, Any, List
import time
from sovl_config import ConfigManager
from sovl_logger import Logger
from sovl_utils import NumericalGuard
from sovl_curiosity import CuriosityManager
from sovl_trainer import SOVLTrainer
from sovl_scaffold import CrossAttentionInjector

class SOVLTuner:
    """Centralized module for tuning SOVL system parameters dynamically."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        logger: Logger,
        curiosity_manager: Optional[CuriosityManager] = None,
        trainer: Optional[SOVLTrainer] = None,
        cross_attention_injector: Optional[CrossAttentionInjector] = None
    ):
        self.config_manager = config_manager
        self.logger = logger
        self.numerical_guard = NumericalGuard()
        self.curiosity_manager = curiosity_manager
        self.trainer = trainer
        self.cross_attention_injector = cross_attention_injector
        
        # Cache configuration sections
        self.core_config = config_manager.get_section("core_config")
        self.controls_config = config_manager.get_section("controls_config")
        self.training_config = config_manager.get_section("training_config")
        self.curiosity_config = config_manager.get_section("curiosity_config")
        self.cross_attn_config = config_manager.get_section("cross_attn_config")
        self.lora_config = config_manager.get_section("lora_config")
        
        # Define parameter ranges for validation
        self.tuning_ranges = {
            "curiosity_config.enable_curiosity": (bool, [True, False]),
            "curiosity_config.novelty_threshold_spontaneous": (0.5, 1.0),
            "curiosity_config.novelty_threshold_response": (0.5, 1.0),
            "curiosity_config.pressure_threshold": (0.5, 0.9),
            "curiosity_config.pressure_drop": (0.1, 0.5),
            "curiosity_config.silence_threshold": (5.0, 60.0),
            "curiosity_config.question_cooldown": (30.0, 120.0),
            "curiosity_config.queue_maxlen": (5, 20),
            "curiosity_config.weight_ignorance": (0.0, 1.0),
            "curiosity_config.weight_novelty": (0.0, 1.0),
            "curiosity_config.max_new_tokens": (5, 12),
            "curiosity_config.base_temperature": (0.5, 1.5),
            "curiosity_config.temperament_influence": (0.1, 0.6),
            "curiosity_config.top_k": (10, 50),
            "curiosity_config.attention_weight": (0.0, 1.0),
            "curiosity_config.question_timeout": (60.0, 86400.0),
            "controls_config.temp_eager_threshold": (0.7, 0.9),
            "controls_config.temp_sluggish_threshold": (0.4, 0.6),
            "controls_config.temp_mood_influence": (0.0, 1.0),
            "controls_config.temp_curiosity_boost": (0.0, 0.5),
            "controls_config.temp_restless_drop": (0.0, 0.5),
            "controls_config.temp_melancholy_noise": (0.0, 0.05),
            "controls_config.conf_feedback_strength": (0.0, 1.0),
            "controls_config.temp_smoothing_factor": (0.0, 1.0),
            "controls_config.temperament_decay_rate": (0.0, 1.0),
            "controls_config.scaffold_weight_cap": (0.5, 1.0),
            "controls_config.base_temperature": (0.5, 1.5),
            "controls_config.sleep_conf_threshold": (0.5, 0.9),
            "controls_config.sleep_time_factor": (0.5, 5.0),
            "controls_config.sleep_log_min": (5, 20),
            "training_config.sleep_max_steps": (10, 1000),
            "controls_config.dream_swing_var": (0.05, 0.2),
            "controls_config.dream_lifecycle_delta": (0.05, 0.2),
            "controls_config.dream_temperament_on": (bool, [True, False]),
            "controls_config.dream_noise_scale": (0.01, 0.1),
            "controls_config.dream_memory_weight": (0.0, 0.5),
            "controls_config.dream_memory_maxlen": (5, 20),
            "controls_config.dream_prompt_weight": (0.0, 1.0),
            "controls_config.dream_novelty_boost": (0.0, 0.05),
            "controls_config.dream_memory_decay": (0.0, 1.0),
            "controls_config.dream_prune_threshold": (0.0, 1.0),
            "training_config.lifecycle_capacity_factor": (0.001, 0.1),
            "training_config.lifecycle_curve": (str, ["sigmoid_linear", "exponential"]),
            "training_config.lora_capacity": (0, 1000),
            "core_config.use_dynamic_layers": (bool, [True, False]),
            "core_config.quantization": (str, ["fp16", "int8", "int4"]),
            "controls_config.dynamic_cross_attn_mode": (str, ["confidence", "temperament", "off", None])
        }
        
    def validate_param(self, param_name: str, value: Any) -> bool:
        """Validate a parameter against its allowed range or type."""
        if param_name not in self.tuning_ranges:
            self.logger.record({
                "error": f"Unknown parameter: {param_name}",
                "timestamp": time.time()
            })
            return False
        
        rule = self.tuning_ranges[param_name]
        type_rule, allowed = rule if isinstance(rule, tuple) and len(rule) == 2 else (None, rule)
        
        if type_rule is bool:
            if not isinstance(value, bool):
                self.logger.record({
                    "error": f"Invalid {param_name}: {value} must be boolean",
                    "timestamp": time.time()
                })
                return False
            return value in allowed
        
        if type_rule is str:
            if not isinstance(value, str) or value not in allowed:
                self.logger.record({
                    "error": f"Invalid {param_name}: {value} not in {allowed}",
                    "timestamp": time.time()
                })
                return False
            return True
        
        if isinstance(rule, tuple) and len(rule) == 2 and all(isinstance(x, (int, float)) for x in rule):
            min_val, max_val = rule
            if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                self.logger.record({
                    "error": f"Invalid {param_name}: {value} not in [{min_val}, {max_val}]",
                    "timestamp": time.time()
                })
                return False
        return True
    
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
        """Tune curiosity-related parameters."""
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
                    self.curiosity_config[param] = value
                
                # Notify curiosity manager
                if self.curiosity_manager:
                    self.curiosity_manager.tune(**{k.split(".")[-1]: v for k, v in updates.items()})
                
                # Notify trainer if curiosity parameters are used
                if self.trainer and any(k in updates for k in [
                    "curiosity_config.weight_ignorance",
                    "curiosity_config.weight_novelty",
                    "curiosity_config.pressure_threshold",
                    "curiosity_config.pressure_drop",
                    "curiosity_config.novelty_threshold_spontaneous",
                    "curiosity_config.novelty_threshold_response",
                    "curiosity_config.silence_threshold",
                    "curiosity_config.question_cooldown",
                    "curiosity_config.queue_maxlen",
                    "curiosity_config.max_new_tokens",
                    "curiosity_config.base_temperature",
                    "curiosity_config.temperament_influence",
                    "curiosity_config.top_k"
                ]):
                    self.trainer.config.update({
                        k.split(".")[-1]: v for k, v in updates.items()
                        if k in [
                            "curiosity_config.weight_ignorance",
                            "curiosity_config.weight_novelty",
                            "curiosity_config.pressure_threshold",
                            "curiosity_config.pressure_drop",
                            "curiosity_config.novelty_threshold_spontaneous",
                            "curiosity_config.novelty_threshold_response",
                            "curiosity_config.silence_threshold",
                            "curiosity_config.question_cooldown",
                            "curiosity_config.queue_maxlen",
                            "curiosity_config.max_new_tokens",
                            "curiosity_config.base_temperature",
                            "curiosity_config.temperament_influence",
                            "curiosity_config.top_k"
                        ]
                    })
                
                self.config_manager.save_config()
                self.logger.record({
                    "event": "tune_curiosity",
                    "params": updates,
                    "success": success,
                    "timestamp": time.time()
                })
            return success
        return True
    
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
        """Tune temperament-related parameters."""
        updates = {}
        prefix = "controls_config."
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
                    "event": "adjust_temperament",
                    "params": updates,
                    "success": success,
                    "timestamp": time.time()
                })
            return success
        return True
    
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
            layers = (
                self.cross_attention_injector.get_cross_attention_layers(
                    base_model,
                    mode=self.core_config.get("layer_selection_mode", "balanced")
                ) if self.core_config.get("use_dynamic_layers", False)
                else self.core_config.get("cross_attn_layers", [5, 7])
            )
            if len(layer_weights) != len(layers):
                self.logger.record({
                    "error": f"layer_weights length ({len(layer_weights)}) must match cross-attn layers ({len(layers)})",
                    "timestamp": time.time()
                })
                return False
            # Validate individual weights
            for w in layer_weights:
                if not (0.0 <= w <= 1.0):
                    self.logger.record({
                        "error": f"Invalid layer weight: {w} not in [0.0, 1.0]",
                        "timestamp": time.time()
                    })
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
            weight_display = f"[{', '.join(f'{w:.2f}' for w in layer_weights)}]" if layer_weights else f"{weight if weight is not None else 'unchanged'}"
            self.logger.record({
                "event": "set_scaffold_influence",
                "weight": weight_display,
                "blend_strength": blend_strength,
                "success": True,
                "timestamp": time.time()
            })
            return True
        except Exception as e:
            self.logger.record({
                "error": f"Failed to set scaffold influence: {str(e)}",
                "timestamp": time.time()
            })
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
        curiosity_manager: Optional[CuriosityManager] = None,
        trainer: Optional[SOVLTrainer] = None,
        cross_attention_injector: Optional[CrossAttentionInjector] = None
    ) -> None:
        """Update references to dependent components."""
        if curiosity_manager is not None:
            self.curiosity_manager = curiosity_manager
        if trainer is not None:
            self.trainer = trainer
        if cross_attention_injector is not None:
            self.cross_attention_injector = cross_attention_injector
