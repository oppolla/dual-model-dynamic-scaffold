from typing import Dict, Tuple, Any
from dataclasses import dataclass
from sovl_logger import Logger

@dataclass
class ValidationRange:
    """Defines a validation range for a configuration parameter."""
    min_value: Any
    max_value: Any
    description: str = ""

class ValidationSchema:
    """Shared validation schema for configuration parameters."""
    
    # Core configuration ranges
    CORE_RANGES: Dict[str, ValidationRange] = {
        "hidden_size": ValidationRange(128, 2048, "Model hidden size"),
        "quantization": ValidationRange("fp16", "int8", "Quantization mode"),
        "use_dynamic_layers": ValidationRange(True, False, "Dynamic layer usage")
    }
    
    # Controls configuration ranges
    CONTROLS_RANGES: Dict[str, ValidationRange] = {
        "temp_eager_threshold": ValidationRange(0.7, 0.9, "Eager threshold"),
        "temp_sluggish_threshold": ValidationRange(0.4, 0.6, "Sluggish threshold"),
        "temp_mood_influence": ValidationRange(0.0, 1.0, "Mood influence"),
        "temp_curiosity_boost": ValidationRange(0.0, 0.5, "Curiosity boost"),
        "temp_restless_drop": ValidationRange(0.0, 0.5, "Restless drop"),
        "temp_melancholy_noise": ValidationRange(0.0, 0.05, "Melancholy noise"),
        "conf_feedback_strength": ValidationRange(0.0, 1.0, "Confidence feedback strength"),
        "temp_smoothing_factor": ValidationRange(0.0, 1.0, "Temperament smoothing factor"),
        "temperament_decay_rate": ValidationRange(0.0, 1.0, "Temperament decay rate"),
        "scaffold_weight_cap": ValidationRange(0.5, 1.0, "Scaffold weight cap"),
        "base_temperature": ValidationRange(0.5, 1.5, "Base temperature"),
        "sleep_conf_threshold": ValidationRange(0.5, 0.9, "Sleep confidence threshold"),
        "sleep_time_factor": ValidationRange(0.5, 5.0, "Sleep time factor"),
        "sleep_log_min": ValidationRange(5, 20, "Minimum sleep log entries"),
        "dream_swing_var": ValidationRange(0.05, 0.2, "Dream swing variance"),
        "dream_lifecycle_delta": ValidationRange(0.05, 0.2, "Dream lifecycle delta"),
        "dream_temperament_on": ValidationRange(True, False, "Dream temperament enabled"),
        "dream_noise_scale": ValidationRange(0.01, 0.1, "Dream noise scale"),
        "dream_memory_weight": ValidationRange(0.0, 0.5, "Dream memory weight"),
        "dream_memory_maxlen": ValidationRange(5, 20, "Dream memory max length"),
        "dream_prompt_weight": ValidationRange(0.0, 1.0, "Dream prompt weight"),
        "dream_novelty_boost": ValidationRange(0.0, 0.05, "Dream novelty boost"),
        "dream_memory_decay": ValidationRange(0.0, 1.0, "Dream memory decay"),
        "dream_prune_threshold": ValidationRange(0.0, 1.0, "Dream prune threshold")
    }
    
    # Curiosity configuration ranges
    CURIOSITY_RANGES: Dict[str, ValidationRange] = {
        "enable_curiosity": ValidationRange(True, False, "Curiosity enabled"),
        "novelty_threshold_spontaneous": ValidationRange(0.5, 1.0, "Spontaneous novelty threshold"),
        "novelty_threshold_response": ValidationRange(0.5, 1.0, "Response novelty threshold"),
        "pressure_threshold": ValidationRange(0.5, 0.9, "Pressure threshold"),
        "pressure_drop": ValidationRange(0.1, 0.5, "Pressure drop"),
        "silence_threshold": ValidationRange(5.0, 60.0, "Silence threshold"),
        "question_cooldown": ValidationRange(30.0, 120.0, "Question cooldown"),
        "queue_maxlen": ValidationRange(1, 50, "Question queue max length"),
        "weight_ignorance": ValidationRange(0.0, 1.0, "Ignorance weight"),
        "weight_novelty": ValidationRange(0.0, 1.0, "Novelty weight"),
        "max_new_tokens": ValidationRange(5, 12, "Maximum new tokens"),
        "base_temperature": ValidationRange(0.5, 1.5, "Base temperature"),
        "temperament_influence": ValidationRange(0.1, 0.6, "Temperament influence"),
        "top_k": ValidationRange(10, 50, "Top-k sampling"),
        "attention_weight": ValidationRange(0.0, 1.0, "Attention weight"),
        "question_timeout": ValidationRange(60.0, 86400.0, "Question timeout")
    }
    
    # Training configuration ranges
    TRAINING_RANGES: Dict[str, ValidationRange] = {
        "lifecycle_capacity_factor": ValidationRange(0.001, 0.1, "Lifecycle capacity factor"),
        "lifecycle_curve": ValidationRange("sigmoid_linear", "exponential", "Lifecycle curve type"),
        "lora_capacity": ValidationRange(0, 1000, "LoRA capacity")
    }
    
    @classmethod
    def get_range(cls, section: str, key: str) -> ValidationRange:
        """Get validation range for a configuration parameter."""
        ranges_map = {
            "core_config": cls.CORE_RANGES,
            "controls_config": cls.CONTROLS_RANGES,
            "curiosity_config": cls.CURIOSITY_RANGES,
            "training_config": cls.TRAINING_RANGES
        }
        
        if section not in ranges_map:
            raise ValueError(f"Unknown configuration section: {section}")
            
        if key not in ranges_map[section]:
            raise ValueError(f"Unknown configuration key: {key} in section {section}")
            
        return ranges_map[section][key]
    
    @classmethod
    def validate_value(cls, section: str, key: str, value: Any, logger: Logger) -> Tuple[bool, str]:
        """Validate a configuration value against its defined range."""
        try:
            validation_range = cls.get_range(section, key)
            
            if isinstance(validation_range.min_value, bool):
                if not isinstance(value, bool):
                    return False, f"{key} must be a boolean value"
                return True, ""
                
            if isinstance(validation_range.min_value, str):
                if not isinstance(value, str):
                    return False, f"{key} must be a string value"
                if value not in [validation_range.min_value, validation_range.max_value]:
                    return False, f"{key} must be one of: {validation_range.min_value}, {validation_range.max_value}"
                return True, ""
                
            if not isinstance(value, (int, float)):
                return False, f"{key} must be a numeric value"
                
            if not (validation_range.min_value <= value <= validation_range.max_value):
                return False, (
                    f"{key} must be between {validation_range.min_value} and {validation_range.max_value}. "
                    f"Got: {value}"
                )
                
            return True, ""
            
        except ValueError as e:
            logger.record_event(
                event_type="validation_error",
                message=f"Validation error for {section}.{key}: {str(e)}",
                level="error"
            )
            return False, str(e) 