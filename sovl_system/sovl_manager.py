import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
import bitsandbytes as bnb
from typing import Optional, List, Dict, Any
import traceback
import os
from threading import Lock
import time

class ModelManager:
    """
    A robust module for managing model loading, initialization, and switching in the SOVL system.
    Handles base model, scaffold models, tokenizers, and related configurations.
    """
    def __init__(self, config_manager, logger, device: torch.device):
        """
        Initialize the ModelManager.

        Args:
            config_manager: ConfigManager instance for accessing configuration.
            logger: Logger instance for recording events and errors.
            device: Torch device (cuda/cpu) for model placement.
        """
        self.config_manager = config_manager
        self.logger = logger
        self.device = device
        self.memory_lock = Lock()

        # Cache configuration sections
        self.core_config = config_manager.get_section("core_config")
        self.lora_config = config_manager.get_section("lora_config")
        self.controls_config = config_manager.get_section("controls_config")

        # Model storage
        self.base_model = None
        self.scaffold_models = []  # List to support multiple scaffolds if needed
        self.base_tokenizer = None
        self.scaffold_tokenizer = None
        self.base_config = None
        self.scaffold_config = None

        # Quantization and model settings
        self.quantization_mode = self.core_config.get("quantization", "fp16")
        self.scaffold_unk_id = self.controls_config.get("scaffold_unk_id", None)

        # Initialize models and tokenizers
        self.load_models()

    def load_models(self):
        """Load base and scaffold models along with their tokenizers."""
        with self.memory_lock:
            try:
                start_time = time.time()
                self._load_base_model()
                base_load_time = time.time() - start_time
                
                start_time = time.time()
                self._load_scaffold_model()
                scaffold_load_time = time.time() - start_time
                
                start_time = time.time()
                self._load_tokenizers()
                tokenizer_load_time = time.time() - start_time
                
                total_load_time = time.time() - start_time
                
                self.logger.record({
                    "event": "models_loaded",
                    "base_model": self.core_config.get("base_model_name", "unknown"),
                    "scaffold_model": self.core_config.get("scaffold_model_name", "unknown"),
                    "quantization": self.quantization_mode,
                    "load_times": {
                        "base_model": base_load_time,
                        "scaffold_model": scaffold_load_time,
                        "tokenizers": tokenizer_load_time,
                        "total": total_load_time
                    },
                    "memory_usage": {
                        "base_model": self._get_model_memory_usage(self.base_model),
                        "scaffold_model": self._get_model_memory_usage(self.scaffold_models[0]) if self.scaffold_models else None
                    },
                    "timestamp": time.time()
                })
                print("All models and tokenizers loaded successfully.")
            except Exception as e:
                self.logger.record({
                    "error": f"Model loading failed: {str(e)}",
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc()
                })
                raise

    def _get_model_memory_usage(self, model: Optional[nn.Module]) -> Optional[Dict[str, Any]]:
        """Get memory usage statistics for a model."""
        if model is None:
            return None
            
        try:
            if torch.cuda.is_available():
                return {
                    "allocated": torch.cuda.memory_allocated(),
                    "reserved": torch.cuda.memory_reserved(),
                    "max_allocated": torch.cuda.max_memory_allocated()
                }
            return {
                "parameters": sum(p.numel() for p in model.parameters()),
                "buffers": sum(b.numel() for b in model.buffers())
            }
        except Exception as e:
            self.logger.record({
                "warning": f"Failed to get memory usage: {str(e)}",
                "timestamp": time.time()
            })
            return None

    def _load_base_model(self):
        """Load the base model with specified quantization."""
        base_model_name = self.core_config.get("base_model_name", "gpt2")
        print(f"Loading base model: {base_model_name}")
        try:
            start_time = time.time()
            self.base_config = AutoConfig.from_pretrained(base_model_name)
            config_load_time = time.time() - start_time
            
            start_time = time.time()
            quantization_config = self._get_quantization_config()
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                config=self.base_config,
                **quantization_config
            ).to(self.device)
            model_load_time = time.time() - start_time
            
            start_time = time.time()
            self.base_model.eval()
            for param in self.base_model.parameters():
                param.requires_grad = False
            setup_time = time.time() - start_time
            
            self.logger.record({
                "event": "base_model_loaded",
                "model_name": base_model_name,
                "quantization": self.quantization_mode,
                "load_times": {
                    "config": config_load_time,
                    "model": model_load_time,
                    "setup": setup_time,
                    "total": config_load_time + model_load_time + setup_time
                },
                "memory_usage": self._get_model_memory_usage(self.base_model),
                "timestamp": time.time()
            })
            print(f"Base model '{base_model_name}' loaded and frozen.")
        except Exception as e:
            self.logger.record({
                "error": f"Failed to load base model: {str(e)}",
                "model_name": base_model_name,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    def _load_scaffold_model(self):
        """Load the scaffold model, optionally with LoRA adapters."""
        scaffold_model_name = self.core_config.get("scaffold_model_name", "gpt2")
        print(f"Loading scaffold model: {scaffold_model_name}")
        try:
            start_time = time.time()
            self.scaffold_config = AutoConfig.from_pretrained(scaffold_model_name)
            config_load_time = time.time() - start_time
            
            start_time = time.time()
            quantization_config = self._get_quantization_config()
            scaffold_model_raw = AutoModelForCausalLM.from_pretrained(
                scaffold_model_name,
                config=self.scaffold_config,
                **quantization_config
            )
            model_load_time = time.time() - start_time
            
            start_time = time.time()
            if self.controls_config.get("enable_lora_adapters", True):
                lora_config = LoraConfig(
                    r=self.lora_config.get("lora_rank", 8),
                    lora_alpha=self.lora_config.get("lora_alpha", 16),
                    target_modules=self.lora_config.get("lora_target_modules", ["c_attn", "c_proj", "c_fc"]),
                    lora_dropout=self.lora_config.get("lora_dropout", 0.1),
                    bias="none",
                    task_type=TaskType.CAUSAL_LM
                )
                self.scaffold_models = [get_peft_model(scaffold_model_raw, lora_config).to(self.device)]
                print("LoRA adapters applied to scaffold model.")
            else:
                self.scaffold_models = [scaffold_model_raw.to(self.device)]
                print("Scaffold model loaded without LoRA adapters.")
            setup_time = time.time() - start_time
            
            self.logger.record({
                "event": "scaffold_model_loaded",
                "model_name": scaffold_model_name,
                "lora_enabled": self.controls_config.get("enable_lora_adapters", True),
                "load_times": {
                    "config": config_load_time,
                    "model": model_load_time,
                    "setup": setup_time,
                    "total": config_load_time + model_load_time + setup_time
                },
                "memory_usage": self._get_model_memory_usage(self.scaffold_models[0]) if self.scaffold_models else None,
                "timestamp": time.time()
            })
        except Exception as e:
            self.logger.record({
                "error": f"Failed to load scaffold model: {str(e)}",
                "model_name": scaffold_model_name,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    def _load_tokenizers(self):
        """Load tokenizers for base and scaffold models."""
        base_model_name = self.core_config.get("base_model_name", "gpt2")
        scaffold_model_name = self.core_config.get("scaffold_model_name", "gpt2")
        print(f"Loading tokenizers: {base_model_name}, {scaffold_model_name}")
        try:
            start_time = time.time()
            self.base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            base_tokenizer_time = time.time() - start_time
            
            start_time = time.time()
            self.scaffold_tokenizer = AutoTokenizer.from_pretrained(scaffold_model_name)
            scaffold_tokenizer_time = time.time() - start_time
            
            start_time = time.time()
            # Set padding tokens if not defined
            if self.base_tokenizer.pad_token is None:
                self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            if self.scaffold_tokenizer.pad_token is None:
                self.scaffold_tokenizer.pad_token = self.scaffold_tokenizer.eos_token
                
            # Update model configs
            self.base_model.config.pad_token_id = self.base_tokenizer.pad_token_id
            if self.scaffold_models:
                self.scaffold_models[0].config.pad_token_id = self.scaffold_tokenizer.pad_token_id
                
            # Set scaffold_unk_id if not set
            if self.scaffold_unk_id is None:
                self.scaffold_unk_id = self.scaffold_tokenizer.unk_token_id
                self.controls_config["scaffold_unk_id"] = self.scaffold_unk_id
                self.config_manager.update("controls_config.scaffold_unk_id", self.scaffold_unk_id)
            setup_time = time.time() - start_time
                
            self.logger.record({
                "event": "tokenizers_loaded",
                "base_tokenizer": base_model_name,
                "scaffold_tokenizer": scaffold_model_name,
                "load_times": {
                    "base_tokenizer": base_tokenizer_time,
                    "scaffold_tokenizer": scaffold_tokenizer_time,
                    "setup": setup_time,
                    "total": base_tokenizer_time + scaffold_tokenizer_time + setup_time
                },
                "timestamp": time.time()
            })
        except Exception as e:
            self.logger.record({
                "error": f"Failed to load tokenizers: {str(e)}",
                "base_model_name": base_model_name,
                "scaffold_model_name": scaffold_model_name,
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            raise

    def _get_quantization_config(self) -> Dict[str, Any]:
        """Return quantization configuration based on mode."""
        if self.quantization_mode == "int8":
            return {"load_in_8bit": True}
        elif self.quantization_mode == "int4":
            return {"load_in_4bit": True}
        return {}

    def switch_base_model(self, model_name: str, quantization: Optional[str] = None):
        """
        Switch the base model to a new one.

        Args:
            model_name: Name of the new model (e.g., "gpt2", "bert-base-uncased").
            quantization: Optional quantization mode ("fp16", "int8", "int4").
        """
        with self.memory_lock:
            try:
                # Clean up existing base model
                if self.base_model is not None:
                    del self.base_model
                    self.base_model = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Update configuration
                self.core_config["base_model_name"] = model_name
                if quantization:
                    self.quantization_mode = quantization
                    self.core_config["quantization"] = quantization
                self.config_manager.update("core_config.base_model_name", model_name)
                if quantization:
                    self.config_manager.update("core_config.quantization", quantization)

                # Load new base model and tokenizer
                self._load_base_model()
                self._load_tokenizers()  # Reload tokenizers to match new base model

                self.logger.record({
                    "event": "base_model_switched",
                    "new_model": model_name,
                    "quantization": self.quantization_mode,
                    "timestamp": time.time()
                })
                print(f"Switched base model to {model_name} with quantization {self.quantization_mode}")
            except Exception as e:
                self.logger.record({
                    "error": f"Failed to switch base model: {str(e)}",
                    "model_name": model_name,
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc()
                })
                raise

    def switch_scaffold_model(self, model_name: str, quantization: Optional[str] = None, apply_lora: Optional[bool] = None):
        """
        Switch the scaffold model to a new one.

        Args:
            model_name: Name of the new scaffold model.
            quantization: Optional quantization mode ("fp16", "int8", "int4").
            apply_lora: Whether to apply LoRA adapters (defaults to config setting).
        """
        with self.memory_lock:
            try:
                # Clean up existing scaffold models
                if self.scaffold_models:
                    for model in self.scaffold_models:
                        del model
                    self.scaffold_models = []
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Update configuration
                self.core_config["scaffold_model_name"] = model_name
                if quantization:
                    self.quantization_mode = quantization
                    self.core_config["quantization"] = quantization
                if apply_lora is not None:
                    self.controls_config["enable_lora_adapters"] = apply_lora
                    self.config_manager.update("controls_config.enable_lora_adapters", apply_lora)
                self.config_manager.update("core_config.scaffold_model_name", model_name)
                if quantization:
                    self.config_manager.update("core_config.quantization", quantization)

                # Load new scaffold model's tokenizer
                self._load_scaffold_model()
                self._load_tokenizers()  # Reload tokenizers to match new scaffold model

                self.logger.record({
                    "event": "scaffold_model_switched",
                    "new_model": model_name,
                    "quantization": self.quantization_mode,
                    "lora_enabled": self.controls_config.get("enable_lora_adapters", True),
                    "timestamp": time.time()
                })
                print(f"Switched scaffold model to {model_name} with quantization {self.quantization_mode}")
            except Exception as e:
                self.logger.record({
                    "error": f"Failed to switch scaffold model: {str(e)}",
                    "model_name": model_name,
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc()
                })
                raise

    def set_quantization_mode(self, mode: str):
        """
        Set quantization mode without reloading models.

        Args:
            mode: Quantization mode ("fp16", "int8", "int4").
        """
        if mode in ["fp16", "int8", "int4"] and mode != self.quantization_mode:
            self.quantization_mode = mode
            self.core_config["quantization"] = mode
            self.config_manager.update("core_config.quantization", mode)
            self.logger.record({
                "event": "quantization_mode_set",
                "mode": mode,
                "timestamp": time.time()
            })
            print(f"Quantization mode set to '{mode}'. Restart or reload models to apply.")
        else:
            print(f"Invalid mode '{mode}' or no change.")

    def reload_models(self):
        """Reload all models with current configuration."""
        with self.memory_lock:
            try:
                # Clean up existing models
                if self.base_model:
                    del self.base_model
                for model in self.scaffold_models:
                    del model
                self.base_model = None
                self.scaffold_models = []
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Reload models
                self.load_models()

                self.logger.record({
                    "event": "models_reloaded",
                    "base_model": self.core_config.get("base_model_name", "unknown"),
                    "scaffold_model": self.core_config.get("scaffold_model_name", "unknown"),
                    "quantization": self.quantization_mode,
                    "timestamp": time.time()
                })
                print("All models reloaded successfully.")
            except Exception as e:
                self.logger.record({
                    "error": f"Model reload failed: {str(e)}",
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc()
                })
                raise

    def get_base_model(self) -> Optional[nn.Module]:
        """Return the base model."""
        return self.base_model

    def get_scaffold_model(self, index: int = 0) -> Optional[nn.Module]:
        """Return the scaffold model at the specified index."""
        return self.scaffold_models[index] if index < len(self.scaffold_models) else None

    def get_base_tokenizer(self) -> Optional[AutoTokenizer]:
        """Return the base tokenizer."""
        return self.base_tokenizer

    def get_scaffold_tokenizer(self) -> Optional[AutoTokenizer]:
        """Return the scaffold tokenizer."""
        return self.scaffold_tokenizer

    def get_scaffold_unk_id(self) -> Optional[int]:
        """Return the scaffold unknown token ID."""
        return self.scaffold_unk_id

    def cleanup(self):
        """Clean up models and free memory."""
        with self.memory_lock:
            try:
                if self.base_model:
                    del self.base_model
                    self.base_model = None
                for model in self.scaffold_models:
                    del model
                self.scaffold_models = []
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.logger.record({
                    "event": "model_manager_cleanup",
                    "timestamp": time.time()
                })
                print("ModelManager cleanup completed.")
            except Exception as e:
                self.logger.record({
                    "error": f"ModelManager cleanup failed: {str(e)}",
                    "timestamp": time.time(),
                    "stack_trace": traceback.format_exc()
                })
                print(f"ModelManager cleanup failed: {e}")
