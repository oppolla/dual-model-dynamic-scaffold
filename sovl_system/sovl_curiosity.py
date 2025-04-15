import time
from typing import Any, Dict, List, Optional
from collections import deque
import traceback

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


class CuriosityManager:
    """Orchestrates curiosity computation, pressure management, and question generation."""
    
    _DEFAULT_CONFIG = {
        "weight_ignorance": 0.7,
        "weight_novelty": 0.3,
        "pressure_threshold": 0.7,
        "pressure_drop": 0.3,
        "novelty_threshold_spontaneous": 0.9,
        "novelty_threshold_response": 0.8,
        "silence_threshold": 20.0,
        "question_cooldown": 60.0,
        "queue_maxlen": 10,
        "max_new_tokens": 8,
        "base_temperature": 1.1,
        "temperament_influence": 0.4,
        "top_k": 30,
        "default_hidden_size": 768,
        "metrics_maxlen": 1000
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[Any] = None,
        device: Optional[torch.device] = None
    ):
        self.config = {**self._DEFAULT_CONFIG, **(config or {})}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize curiosity components."""
        self.curiosity = Curiosity(
            weight_ignorance=self.config["weight_ignorance"],
            weight_novelty=self.config["weight_novelty"],
            metrics_maxlen=self.config["metrics_maxlen"],
            logger=self.logger
        )
        self.pressure = CuriosityPressure()
        self.callbacks = CuriosityCallbacks(logger=self.logger)
        self.last_question_time: float = time.time()
        self.unanswered_questions: deque = deque(maxlen=self.config["queue_maxlen"])
        self.metrics: deque = deque(maxlen=self.config["metrics_maxlen"])

    def compute_curiosity(
        self,
        state: Any,
        tokenizer: Any,
        model: Any,
        query: Optional[str] = None
    ) -> float:
        """Compute curiosity score for a query using state information."""
        base_conf = self._get_base_confidence(state)
        scaf_conf = self._get_scaffold_confidence(state)
        query_embedding = self._generate_query_embedding(state, tokenizer, model, query)
        memory_embeddings = self._get_valid_memory_embeddings(state)

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

    def _get_base_confidence(self, state: Any) -> float:
        """Extract base confidence from state."""
        return state.confidence_history[-1] if state.confidence_history else 0.5

    def _get_scaffold_confidence(self, state: Any) -> float:
        """Extract scaffold confidence from state."""
        return (
            state.sleep_confidence_sum / state.sleep_confidence_count
            if state.sleep_confidence_count > 0 else 0.5
        )

    def _generate_query_embedding(
        self, state: Any, tokenizer: Any, model: Any, query: Optional[str]
    ) -> Optional[torch.Tensor]:
        """Generate query embedding from input query or state."""
        if not (query and tokenizer and model):
            return getattr(state, 'last_prompt_embedding', None) or torch.zeros(
                self.config["default_hidden_size"], device=self.device
            )

        try:
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
                    return hidden_states[:, -1, :].squeeze()
                self._log_warning("Model output lacks hidden states")
        except Exception as e:
            self._log_error(f"Failed to generate query embedding: {str(e)}")
        return None

    def _get_valid_memory_embeddings(self, state: Any) -> List[torch.Tensor]:
        """Extract valid memory embeddings from state."""
        memory_embeddings = []
        hidden_size = self.config["default_hidden_size"]
        if not hasattr(state, 'dream_memory') or not state.dream_memory:
            return memory_embeddings

        try:
            for tensor, _ in state.dream_memory:
                if tensor.shape[-1] == hidden_size:
                    memory_embeddings.append(tensor.to(self.device))
                else:
                    self._log_warning(
                        f"Dream memory tensor shape {tensor.shape} mismatches hidden_size {hidden_size}"
                    )
        except Exception as e:
            self._log_error(f"Invalid dream memory format: {str(e)}")
        return memory_embeddings

    def update_pressure(
        self,
        temperament: float,
        confidence: float,
        silence_duration: float
    ) -> None:
        """Update curiosity pressure based on system state."""
        silence_threshold = max(self.config["silence_threshold"], 1e-6)
        self.pressure.update(
            temperament=temperament,
            confidence=confidence,
            silence=silence_duration,
            silence_threshold=silence_threshold
        )
        self.callbacks.trigger_callback("pressure_updated", pressure=self.pressure.value)

    def should_erupt(self, threshold: Optional[float] = None) -> bool:
        """Check if curiosity pressure should trigger a question."""
        thresh = threshold or self.config["pressure_threshold"]
        if not 0.5 <= thresh <= 0.9:
            thresh = 0.7
            self._log_warning(f"Invalid pressure threshold {thresh}, using default 0.7")
        
        if self.pressure.should_erupt(thresh):
            self.pressure.drop_pressure(self.config["pressure_drop"])
            self.callbacks.trigger_callback("pressure_erupted", pressure=self.pressure.value)
            return True
        return False

    def generate_question(
        self,
        state: Any,
        tokenizer: Any,
        model: Any,
        prompt: Optional[str] = None,
        spontaneous: bool = False
    ) -> Optional[str]:
        """Generate a curiosity-driven question if conditions are met."""
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
            self._update_question_state(question, curiosity_score, current_time)
        return question

    def _can_generate_question(self, tokenizer: Any, model: Any) -> bool:
        """Check if question generation is possible."""
        if not (tokenizer and model):
            self._log_warning("Missing tokenizer or model for question generation")
            return False
        return True

    def _check_cooldown(self, current_time: float) -> bool:
        """Check if question generation cooldown has elapsed."""
        return (current_time - self.last_question_time) >= self.config["question_cooldown"]

    def _should_generate_question(self, curiosity_score: float, prompt: Optional[str], spontaneous: bool) -> bool:
        """Determine if a question should be generated."""
        threshold = (
            self.config["novelty_threshold_spontaneous"] if spontaneous
            else self.config["novelty_threshold_response"]
        )
        return (
            (spontaneous and curiosity_score >= threshold) or
            (not spontaneous and prompt and curiosity_score >= threshold) or
            self.should_erupt()
        )

    def _generate_question_text(
        self, state: Any, tokenizer: Any, model: Any, prompt: Optional[str]
    ) -> Optional[str]:
        """Generate question text using model and tokenizer."""
        base_prompt = self._get_base_prompt(state, prompt)
        try:
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
                    max_new_tokens=self.config["max_new_tokens"],
                    temperature=max(0.1, self.config["base_temperature"] + 
                                  self.config["temperament_influence"] * state.temperament_score),
                    top_k=self.config["top_k"],
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            question = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            return question if question else None
        except Exception as e:
            self._log_error(f"Question generation failed: {str(e)}")
            return None

    def _get_base_prompt(self, state: Any, prompt: Optional[str]) -> str:
        """Select appropriate base prompt for question generation."""
        return (
            prompt if prompt and isinstance(prompt, str)
            else getattr(getattr(state, 'history', None), 'messages', [{}])[-1].get('prompt', "What is this about?")
        )

    def _update_question_state(self, question: str, curiosity_score: float, current_time: float) -> None:
        """Update state after question generation."""
        self.unanswered_questions.append((question, curiosity_score))
        self.last_question_time = current_time
        self.pressure.drop_pressure(self.config["pressure_drop"])
        self.callbacks.trigger_callback("question_generated", question=question, score=curiosity_score)

    def update_metrics(
        self,
        question: str,
        score: float,
        spontaneous: bool = False,
        answered: bool = False,
        conversation_id: Optional[str] = None,
        state_hash: Optional[str] = None
    ) -> None:
        """Update curiosity metrics with automatic cleanup and logging."""
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
        self._log_event("metrics_update", metric)
        self.callbacks.trigger_callback("metrics_updated", **metric)

    def _log_metrics_update(self, metric: Dict) -> None:
        """Log metrics update event."""
        if self.logger:
            self.logger.record_event(
                event_type="curiosity_metrics_update",
                message="Curiosity metrics updated",
                level="info",
                additional_info=metric
            )

    def get_metrics(self, limit: Optional[int] = None) -> List[Dict]:
        """Get metrics with optional limit."""
        metrics = list(self.metrics)
        return metrics[-limit:] if limit is not None else metrics

    def clear_metrics(self) -> None:
        """Clear all metrics."""
        self.metrics.clear()
        self.callbacks.trigger_callback("metrics_cleared")

    def check_silence(
        self,
        state: Any,
        tokenizer: Any,
        model: Any,
        elapsed: float
    ) -> Optional[str]:
        """Check for prolonged silence and generate a question if needed."""
        if not self._should_check_silence(elapsed):
            return None

        question = self._try_generate_silence_question(state, tokenizer, model)
        if question:
            return question
        return self._try_reuse_unanswered_question()

    def _should_check_silence(self, elapsed: float) -> bool:
        """Determine if silence check should proceed."""
        return (
            elapsed > self.config["silence_threshold"] and
            self.pressure.value > self.config["pressure_threshold"] and
            self._check_cooldown(time.time())
        )

    def _try_generate_silence_question(self, state: Any, tokenizer: Any, model: Any) -> Optional[str]:
        """Attempt to generate a new question due to silence."""
        question = self.generate_question(state, tokenizer, model, spontaneous=True)
        if question:
            score = self.compute_curiosity(state, tokenizer, model, question)
            self.update_metrics(question, score, spontaneous=True)
            self._finalize_silence_question(question)
            return question
        return None

    def _try_reuse_unanswered_question(self) -> Optional[str]:
        """Attempt to reuse an unanswered question."""
        if self.unanswered_questions:
            question, score = self.unanswered_questions.popleft()
            self.update_metrics(question, score, spontaneous=True)
            self._finalize_silence_question(question, 0.7)
            return question
        return None

    def _finalize_silence_question(self, question: str, pressure_drop_factor: float = 1.0) -> None:
        """Update state after generating a silence question."""
        self.pressure.drop_pressure(self.config["pressure_drop"] * pressure_drop_factor)
        self.last_question_time = time.time()

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
        if "queue_maxlen" in updates:
            self.unanswered_questions = deque(self.unanswered_questions, maxlen=updates["queue_maxlen"])
        if "metrics_maxlen" in updates:
            self.metrics = deque(self.metrics, maxlen=updates["metrics_maxlen"])
        if "weight_ignorance" in updates:
            self.curiosity.weight_ignorance = updates["weight_ignorance"]
        if "weight_novelty" in updates:
            self.curiosity.weight_novelty = updates["weight_novelty"]

        self.config.update(updates)
        if updates and self.logger:
            self.logger.record_event(
                event_type="curiosity_tune",
                message="Curiosity parameters tuned",
                level="info",
                additional_info={"params": updates}
            )

    def get_pressure(self) -> float:
        """Get current pressure value."""
        return self.pressure.value

    def reduce_pressure(self, amount: float) -> None:
        """Reduce pressure by specified amount."""
        self.pressure.drop_pressure(amount)

    def save_state(self) -> Dict:
        """Save current state."""
        return {
            "pressure": self.pressure.value,
            "last_question_time": self.last_question_time,
            "unanswered_questions": list(self.unanswered_questions),
            "metrics": list(self.metrics)
        }

    def load_state(self, state_dict: Dict) -> None:
        """Load state from dictionary."""
        self.pressure.value = state_dict.get("pressure", 0.0)
        self.last_question_time = state_dict.get("last_question_time", time.time())
        self.unanswered_questions = deque(
            state_dict.get("unanswered_questions", []),
            maxlen=self.config["queue_maxlen"]
        )
        self.metrics = deque(
            state_dict.get("metrics", []),
            maxlen=self.config["metrics_maxlen"]
        )

    def generate_curiosity_question(
        self, state: Any, tokenizer: Any, model: Any, context: Optional[str] = None, spontaneous: bool = False
    ) -> Optional[str]:
        """Generate a curiosity-driven question."""
        if not self.config.get("enable_curiosity", True):
            return None
        question = self.generate_question(state, tokenizer, model, context, spontaneous)
        if question and hasattr(state, 'curiosity'):
            state.curiosity.update_question_history(question, time.time())
            self._log_event("curiosity_question", {"prompt": question, "spontaneous": spontaneous})
        return question

    def _check_silence(
        self, state: Any, tokenizer: Any, model: Any, elapsed: float
    ) -> Optional[str]:
        """Check silence and generate question if needed."""
        if not self.config.get("enable_curiosity", True):
            return None
        if hasattr(state, 'curiosity'):
            state.curiosity.prune_old_questions(self.config.get("question_timeout", 3600.0))
        question = self.generate_question(state, tokenizer, model, spontaneous=True)
        if question and hasattr(state, 'curiosity'):
            state.curiosity.update_question_history(question, time.time())
            self._log_event("silence_question", {"prompt": question})
        return question

    def tune_curiosity(self, **kwargs) -> None:
        """Tune curiosity parameters with kwargs."""
        self.tune(**kwargs)

    def _log_warning(self, message: str) -> None:
        """Log warning if logger is available."""
        if self.logger:
            self.logger.record_event(
                event_type="curiosity_warning",
                message=message,
                level="warning"
            )

    def _log_event(self, event: str, data: Dict) -> None:
        """Log event with additional data."""
        if self.logger:
            self.logger.record_event(
                event_type=f"curiosity_{event}",
                message=f"Curiosity event: {event}",
                level="info",
                additional_info=data
            )

    def _update_state_novelty(self, state: Any, score: float) -> None:
        """Update state novelty scores if available."""
        if hasattr(state, 'curiosity') and hasattr(state.curiosity, 'novelty_scores'):
            state.curiosity.novelty_scores.append(score)
