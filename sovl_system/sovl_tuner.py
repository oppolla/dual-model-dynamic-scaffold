from typing import Optional, Dict, Any
from sovl_config import ConfigManager
from sovl_logger import Logger
from sovl_utils import NumericalGuard

class SOVLTuner:
    """Manages dynamic tuning of SOVL system parameters."""
    
    def __init__(self, config_manager: ConfigManager, logger: Logger):
        self.config_manager = config_manager
        self.logger = logger
        self.numerical_guard = NumericalGuard()  # For safe numerical operations
        self.core_config = config_manager.get_section("core_config")
        self.controls_config = config_manager.get_section("controls_config")
        self.training_config = config_manager.get_section("training_config")
        self.curiosity_config = config_manager.get_section("curiosity_config")
        self.cross_attn_config = config_manager.get_section("cross_attn_config")
        self.lora_config = config_manager.get_section("lora_config")
        
        # Cache tuning ranges for validation
        self.tuning_ranges = {
            "curiosity_weight_ignorance": (0.0, 1.0),
            "curiosity_pressure_threshold": (0.5, 0.9),
            "temp_eager_threshold": (0.7, 0.9),
            "scaffold_weight_cap": (0.5, 1.0),
            # ... add ranges for all tunable parameters
        }
        
    def validate_param(self, param_name: str, value: Any) -> bool:
        """Validate a parameter value against its allowed range."""
        if param_name in self.tuning_ranges:
            min_val, max_val = self.tuning_ranges[param_name]
            if not (isinstance(value, (int, float)) and min_val <= value <= max_val):
                self.logger.record({
                    "error": f"Invalid {param_name}: {value} not in [{min_val}, {max_val}]",
                    "timestamp": time.time()
                })
                return False
        return True
    
    def tune_curiosity(self, **kwargs) -> bool:
        """Tune curiosity parameters."""
        updates = {}
        prefix = "curiosity_config."
        for param, value in kwargs.items():
            full_key = f"{prefix}{param}"
            if self.validate_param(full_key, value):
                updates[full_key] = value
            else:
                return False
        
        if updates:
            success = self.config_manager.update_batch(updates)
            self.logger.record({
                "event": "tune_curiosity",
                "params": updates,
                "success": success,
                "timestamp": time.time()
            })
            return success
        return True
    
    def adjust_temperament(self, **kwargs) -> bool:
        """Tune temperament parameters."""
        updates = {}
        prefix = "controls_config."
        for param, value in kwargs.items():
            full_key = f"{prefix}{param}"
            if self.validate_param(full_key, value):
                updates[full_key] = value
            else:
                return False
        
        if updates:
            success = self.config_manager.update_batch(updates)
            self.logger.record({
                "event": "adjust_temperament",
                "params": updates,
                "success": success,
                "timestamp": time.time()
            })
            return success
        return True
    
    def tune_cross_attention(self, weight: Optional[float] = None, 
                           blend_strength: Optional[float] = None,
                           layer_weights: Optional[list] = None,
                           dynamic_mode: Optional[str] = None) -> bool:
        """Tune cross-attention parameters."""
        updates = {}
        if dynamic_mode in ['confidence', 'temperament', 'off']:
            updates["controls_config.dynamic_cross_attn_mode"] = dynamic_mode if dynamic_mode != 'off' else None
        # Delegate weight/blend_strength to sovl_scaffold.py if migrated there
        # For now, log and update config
        if updates:
            success = self.config_manager.update_batch(updates)
            self.logger.record({
                "event": "tune_cross_attention",
                "params": updates,
                "success": success,
                "timestamp": time.time()
            })
            return success
        return True
    
    def tune_dream(self, **kwargs) -> bool:
        """Tune dreaming parameters."""
        updates = {}
        prefix = "controls_config."
        for param, value in kwargs.items():
            full_key = f"{prefix}{param}"
            if self.validate_param(full_key, value):
                updates[full_key] = value
            else:
                return False
        
        if updates:
            success = self.config_manager.update_batch(updates)
            self.logger.record({
                "event": "tune_dream",
                "params": updates,
                "success": success,
                "timestamp": time.time()
            })
            return success
        return True
    
    def set_global_blend(self, weight_cap: Optional[float] = None, 
                        base_temp: Optional[float] = None) -> bool:
        """Tune global blend parameters."""
        updates = {}
        prefix = "controls_config."
        if weight_cap is not None and self.validate_param("scaffold_weight_cap", weight_cap):
            updates[f"{prefix}scaffold_weight_cap"] = weight_cap
        if base_temp is not None and self.validate_param("base_temperature", base_temp):
            updates[f"{prefix}base_temperature"] = base_temp
        
        if updates:
            success = self.config_manager.update_batch(updates)
            self.logger.record({
                "event": "set_global_blend",
                "params": updates,
                "success": success,
                "timestamp": time.time()
            })
            return success
        return True
    
    def set_sleep_params(self, **kwargs) -> bool:
        """Tune sleep parameters."""
        updates = {}
        prefix = "controls_config."
        for param, value in kwargs.items():
            full_key = f"{prefix}{param}"
            if self.validate_param(full_key, value):
                updates[full_key] = value
            else:
                return False
        
        if updates:
            success = self.config_manager.update_batch(updates)
            self.logger.record({
                "event": "set_sleep_params",
                "params": updates,
                "success": success,
                "timestamp": time.time()
            })
            return success
        return True
    
    def tune_lifecycle(self, **kwargs) -> bool:
        """Tune lifecycle parameters."""
        updates = {}
        prefix = "training_config."
        for param, value in kwargs.items():
            full_key = f"{prefix}{param}"
            if self.validate_param(full_key, value):
                updates[full_key] = value
            else:
                return False
        
        if updates:
            success = self.config_manager.update_batch(updates)
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
        if self.core_config.get("use_dynamic_layers") != enable:
            success = self.config_manager.update("core_config.use_dynamic_layers", enable)
            self.logger.record({
                "event": "toggle_dynamic_layers",
                "enable": enable,
                "success": success,
                "timestamp": time.time()
            })
            return success
        return True
    
    def set_quantization_mode(self, mode: str) -> bool:
        """Set quantization mode."""
        if mode in ["fp16", "int8", "int4"]:
            success = self.config_manager.update("core_config.quantization", mode)
            self.logger.record({
                "event": "set_quantization_mode",
                "mode": mode,
                "success": success,
                "timestamp": time.time()
            })
            return success
        self.logger.record({
            "error": f"Invalid quantization mode: {mode}",
            "timestamp": time.time()
        })
        return False
