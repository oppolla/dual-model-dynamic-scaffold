import time
from typing import Any, Dict, List, Optional
from collections import deque

import torch
from torch import nn


class Curiosity:
    """
    Computes curiosity scores based on ignorance and novelty.
    """
    def __init__(
        self,
        weight_ignorance: float = 0.7,
        weight_novelty: float = 0.3,
        logger: Optional[Any] = None
    ):
        self.weight_ignorance = weight_ignorance
        self.weight_novelty = weight_novelty
        self.logger = logger
        self.cosine_similarity = nn.CosineSimilarity(dim=-1, eps=1e-8)
        self.metrics_maxlen = self.config.get("metrics_maxlen", 1000)  # Default to 1000 entries
        self.metrics: deque = deque(maxlen=self.metrics_maxlen)

    def compute_curiosity(
        self,
        base_conf: float,
        scaf_conf: float,
        memory_embeddings: List[torch.Tensor],
        query_embedding: torch.Tensor,
        device: torch.device
    ) -> float:
        """
        Compute curiosity score based on confidence and embeddings.

        Args:
            base_conf (float): Base model confidence.
            scaf_conf (float): Scaffold model confidence.
            memory_embeddings (List[torch.Tensor]): List of memory embeddings.
            query_embedding (torch.Tensor): Query embedding.
            device (torch.device): Device for tensor operations.

        Returns:
            float: Curiosity score.
        """
        try:
            # Early return if no embeddings to process
            if not memory_embeddings or query_embedding is None:
                return self._compute_ignorance_score(base_conf, scaf_conf)

            # Compute components
            ignorance = self._compute_ignorance_score(base_conf, scaf_conf)
            novelty = self._compute_novelty_score(memory_embeddings, query_embedding, device)

            # Combine scores with weights
            score = (
                self.weight_ignorance * ignorance +
                self.weight_novelty * novelty
            )
            return max(0.0, min(1.0, score))

        except Exception as e:
            if self.logger:
                self.logger.record({
                    "error": f"Curiosity computation failed: {str(e)}",
                    "timestamp": time.time()
                })
            return 0.5

    def _compute_ignorance_score(self, base_conf: float, scaf_conf: float) -> float:
        """Compute ignorance component of curiosity score."""
        ignorance = 1.0 - (base_conf * 0.5 + scaf_conf * 0.5)
        return max(0.0, min(1.0, ignorance))

    def _compute_novelty_score(
        self,
        memory_embeddings: List[torch.Tensor],
        query_embedding: torch.Tensor,
        device: torch.device
    ) -> float:
        """Compute novelty component of curiosity score."""
        query_embedding = query_embedding.to(device)
        similarities = [
            self.cosine_similarity(
                query_embedding,
                emb.to(device)
            ).item()
            for emb in memory_embeddings
        ]
        max_similarity = max(similarities, default=0.0)
        return 1.0 - max(0.0, min(1.0, max_similarity))


class CuriosityPressure:
    """
    Manages curiosity pressure accumulation and eruption.
    """
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
        """
        Update pressure based on temperament, confidence, and silence.

        Args:
            temperament (float): Current temperament score.
            confidence (float): Current confidence score.
            silence (float): Time since last interaction.
            silence_threshold (float): Threshold for silence contribution.
        """
        current_time = time.time()
        time_delta = current_time - self.last_update
        self.last_update = current_time

        temperament_effect = 0.1 * max(0.0, temperament)
        confidence_effect = 0.05 * (1.0 - confidence)
        silence_effect = 0.2 * (silence / silence_threshold) if silence > silence_threshold else 0.0

        self.value += time_delta * (temperament_effect + confidence_effect + silence_effect)
        self.value = max(0.0, min(1.0, self.value))

    def should_erupt(self, threshold: float) -> bool:
        """
        Check if pressure exceeds threshold.

        Args:
            threshold (float): Pressure threshold for eruption.

        Returns:
            bool: True if eruption should occur.
        """
        return self.value >= threshold

    def drop_pressure(self, amount: float) -> None:
        """
        Reduce pressure by a specified amount.

        Args:
            amount (float): Amount to reduce pressure.
        """
        self.value = max(0.0, self.value - amount)


class CuriosityCallbacks:
    """
    Handles curiosity-related callbacks.
    """
    def __init__(self, logger: Optional[Any] = None):
        self.callbacks = {}
        self.logger = logger

    def register_callback(self, event: str, callback: callable) -> None:
        """
        Register a callback for an event.

        Args:
            event (str): Event name.
            callback (callable): Callback function.
        """
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)

    def trigger_callback(self, event: str, **kwargs) -> None:
        """
        Trigger callbacks for an event.

        Args:
            event (str): Event name.
            **kwargs: Arguments to pass to callbacks.
        """
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(**kwargs)
                except Exception as e:
                    if self.logger:
                        self.logger.record({
                            "error": f"Callback {event} failed: {str(e)}",
                            "timestamp": time.time()
                        })


class CuriosityManager:
    """
    Orchestrates curiosity computation, pressure management, and question generation.
    """
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[Any] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the CuriosityManager.

        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary with curiosity parameters.
            logger (Optional[Any]): Logger for events and errors.
            device (Optional[torch.device]): Device for tensor operations (default: cuda if available).
        """
        # Default configuration
        default_config = {
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
            "default_hidden_size": 768
        }
        self.config = {**default_config, **(config or {})}

        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger

        # Initialize components
        self.curiosity = Curiosity(
            weight_ignorance=self.config["weight_ignorance"],
            weight_novelty=self.config["weight_novelty"],
            logger=self.logger
        )
        self.pressure = CuriosityPressure()
        self.callbacks = CuriosityCallbacks(logger=self.logger)
        self.last_question_time: float = time.time()
        self.unanswered_questions: deque = deque(maxlen=self.config["queue_maxlen"])
        self.metrics: List[Dict] = []

    def compute_curiosity(
        self,
        state: Any,
        tokenizer: Any,
        model: Any,
        query: Optional[str] = None
    ) -> float:
        """
        Compute curiosity score for a query using state information.

        Args:
            state: Current system state.
            tokenizer: Tokenizer for encoding queries.
            model: Model to generate embeddings.
            query (Optional[str]): Query to compute curiosity for.

        Returns:
            float: The computed curiosity score.
        """
        base_conf = state.confidence_history[-1] if state.confidence_history else 0.5
        scaf_conf = (
            state.sleep_confidence_sum / state.sleep_confidence_count
            if state.sleep_confidence_count > 0 else 0.5
        )

        query_embedding = None
        if query and tokenizer and model:
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
                    # Safely access hidden states
                    hidden_states = (
                        outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else
                        outputs.base_model_output.hidden_states[-1]
                        if hasattr(outputs, 'base_model_output') and hasattr(outputs.base_model_output, 'hidden_states')
                        else None
                    )
                    if hidden_states is not None:
                        query_embedding = hidden_states[:, -1, :].squeeze()
                    else:
                        if self.logger:
                            self.logger.record({
                                "warning": "Model output lacks hidden states",
                                "timestamp": time.time()
                            })
            except Exception as e:
                if self.logger:
                    self.logger.record({
                        "error": f"Failed to generate query embedding: {str(e)}",
                        "timestamp": time.time()
                    })

        # Validate dream memory
        memory_embeddings = []
        hidden_size = self.config["default_hidden_size"]
        if hasattr(state, 'dream_memory') and state.dream_memory:
            try:
                for tensor, _ in state.dream_memory:
                    if tensor.shape[-1] == hidden_size:
                        memory_embeddings.append(tensor.to(self.device))
                    else:
                        if self.logger:
                            self.logger.record({
                                "warning": f"Dream memory tensor shape {tensor.shape} mismatches hidden_size {hidden_size}",
                                "timestamp": time.time()
                            })
            except Exception as e:
                if self.logger:
                    self.logger.record({
                        "error": f"Invalid dream memory format: {str(e)}",
                        "timestamp": time.time()
                    })

        query_emb = (
            query_embedding if query_embedding is not None else
            state.last_prompt_embedding if hasattr(state, 'last_prompt_embedding') and state.last_prompt_embedding is not None else
            torch.zeros(hidden_size, device=self.device)
        )

        score = self.curiosity.compute_curiosity(
            base_conf=base_conf,
            scaf_conf=scaf_conf,
            memory_embeddings=memory_embeddings,
            query_embedding=query_emb,
            device=self.device
        )

        if hasattr(state, 'curiosity') and hasattr(state.curiosity, 'novelty_scores'):
            state.curiosity.novelty_scores.append(score)
        self.callbacks.trigger_callback("curiosity_computed", score=score)
        return score

    def update_pressure(
        self,
        temperament: float,
        confidence: float,
        silence_duration: float
    ) -> None:
        """
        Update curiosity pressure based on system state.

        Args:
            temperament (float): Current temperament score.
            confidence (float): Current confidence score.
            silence_duration (float): Time since last interaction.
        """
        silence_threshold = max(self.config["silence_threshold"], 1e-6)  # Prevent division by zero
        self.pressure.update(
            temperament=temperament,
            confidence=confidence,
            silence=silence_duration,
            silence_threshold=silence_threshold
        )
        self.callbacks.trigger_callback("pressure_updated", pressure=self.pressure.value)

    def should_erupt(self, threshold: Optional[float] = None) -> bool:
        """
        Check if curiosity pressure should trigger a question.

        Args:
            threshold (Optional[float]): Custom pressure threshold.

        Returns:
            bool: True if pressure erupts, False otherwise.
        """
        thresh = threshold if threshold is not None else self.config["pressure_threshold"]
        if not (0.5 <= thresh <= 0.9):
            thresh = 0.7  # Fallback to default
            if self.logger:
                self.logger.record({
                    "warning": f"Invalid pressure threshold {thresh}, using default 0.7",
                    "timestamp": time.time()
                })
        erupted = self.pressure.should_erupt(thresh)
        if erupted:
            self.pressure.drop_pressure(self.config["pressure_drop"])
            self.callbacks.trigger_callback("pressure_erupted", pressure=self.pressure.value)
        return erupted

    def generate_question(
        self,
        state: Any,
        tokenizer: Any,
        model: Any,
        prompt: Optional[str] = None,
        spontaneous: bool = False
    ) -> Optional[str]:
        """
        Generate a curiosity-driven question if conditions are met.

        Args:
            state: Current system state.
            tokenizer: Tokenizer for encoding prompts.
            model: Model for generating questions.
            prompt (Optional[str]): Prompt to base the question on.
            spontaneous (bool): If True, prioritize spontaneous question generation.

        Returns:
            Optional[str]: Generated question or None if conditions not met.
        """
        if not (tokenizer and model):
            if self.logger:
                self.logger.record({
                    "warning": "Missing tokenizer or model for question generation",
                    "timestamp": time.time()
                })
            return None

        # Check cooldown
        current_time = time.time()
        if (current_time - self.last_question_time) < self.config["question_cooldown"]:
            return None

        # Compute curiosity
        curiosity_score = self.compute_curiosity(state, tokenizer, model, prompt)

        # Check conditions
        threshold = (
            self.config["novelty_threshold_spontaneous"] if spontaneous
            else self.config["novelty_threshold_response"]
        )
        should_generate = (
            (spontaneous and curiosity_score >= threshold) or
            (not spontaneous and prompt and curiosity_score >= threshold) or
            self.should_erupt()
        )
        if not should_generate:
            return None

        # Select base prompt
        base_prompt = (
            prompt if prompt and isinstance(prompt, str) else
            (state.history.messages[-1]["prompt"] if hasattr(state, 'history') and state.history.messages else "What is this about?")
        )

        # Generate question
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
                    temperature=max(0.1, self.config["base_temperature"] + self.config["temperament_influence"] * state.temperament_score),
                    top_k=self.config["top_k"],
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            question = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            if not question:
                return None

            # Update state
            self.unanswered_questions.append((question, curiosity_score))
            self.last_question_time = current_time
            self.pressure.drop_pressure(self.config["pressure_drop"])
            self.callbacks.trigger_callback("question_generated", question=question, score=curiosity_score)
            return question

        except Exception as e:
            if self.logger:
                self.logger.record({
                    "error": f"Question generation failed: {str(e)}",
                    "timestamp": time.time()
                })
            return None

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
        # Check if adding will cause trimming
        will_trim = len(self.metrics) == self.metrics_maxlen
        # Create new metric entry
        new_metric = {
            "question": question,
            "score": score,
            "spontaneous": spontaneous,
            "answered": answered,
            "timestamp": time.time()
        }

        # Add to metrics with automatic size limit
        self.metrics.append(new_metric)

        # Log if trimming occurred
        if will_trim and self.logger:
            self.logger.record({
                "event": "metrics_trimmed",
                "current_size": len(self.metrics),
                "max_size": self.metrics_maxlen,
                "timestamp": time.time()
            })

        # Log the metrics update
        if self.logger:
            self.logger.record({
                "event": "metrics_updated",
                "question": question,
                "score": score,
                "spontaneous": spontaneous,
                "answered": answered,
                "timestamp": time.time(),
                "conversation_id": conversation_id,
                "state_hash": state_hash
            })

        # Trigger callback
        self.callbacks.trigger_callback(
            "metrics_updated",
            question=question,
            score=score,
            spontaneous=spontaneous,
            answered=answered
        )

    def get_metrics(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get metrics with optional limit.

        Args:
            limit (Optional[int]): Maximum number of metrics to return.

        Returns:
            List[Dict]: List of metrics, optionally limited in size.
        """
        if limit is None:
            return list(self.metrics)
        return list(self.metrics)[-limit:]  

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
        """
        Check for prolonged silence and generate a question if needed.

        Args:
            state: Current system state.
            tokenizer: Tokenizer for encoding prompts.
            model: Model for generating questions.
            elapsed (float): Time since last interaction.

        Returns:
            Optional[str]: Generated question or None if not triggered.
        """
        if elapsed <= self.config["silence_threshold"]:
            return None

        if self.pressure.value <= self.config["pressure_threshold"]:
            return None

        if (time.time() - self.last_question_time) <= self.config["question_cooldown"]:
            return None

        # Try generating a new question
        question = self.generate_question(state, tokenizer, model, spontaneous=True)
        if question:
            score = self.compute_curiosity(state, tokenizer, model, question)
            self.update_metrics(question, score, spontaneous=True)
            self.pressure.drop_pressure(self.config["pressure_drop"])
            self.last_question_time = time.time()
            return question

        # Fall back to unanswered questions
        if self.unanswered_questions:
            question, score = self.unanswered_questions.popleft()
            self.update_metrics(question, score, spontaneous=True)
            self.pressure.drop_pressure(self.config["pressure_drop"] * 0.7)
            self.last_question_time = time.time()
            return question

        return None

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
        """
        Tune curiosity parameters.

        Args:
            enable (Optional[bool]): Enable or disable curiosity.
            spontaneous_threshold (Optional[float]): Threshold for spontaneous questions.
            response_threshold (Optional[float]): Threshold for response-driven questions.
            pressure_threshold (Optional[float]): Threshold for pressure eruption.
            pressure_drop (Optional[float]): Amount to drop pressure after eruption.
            silence_threshold (Optional[float]): Silence duration threshold.
            question_cooldown (Optional[float]): Cooldown between questions.
            queue_maxlen (Optional[int]): Maximum length of unanswered questions queue.
            weight_ignorance (Optional[float]): Weight for ignorance in curiosity score.
            weight_novelty (Optional[float]): Weight for novelty in curiosity score.
            max_new_tokens (Optional[int]): Maximum tokens for generated questions.
            base_temperature (Optional[float]): Base temperature for generation.
            temperament_influence (Optional[float]): Temperament influence on generation.
            top_k (Optional[int]): Top-k sampling parameter.
        """
        updates = {}
        if enable is not None:
            updates["enable_curiosity"] = bool(enable)
        if spontaneous_threshold is not None and 0.5 <= spontaneous_threshold <= 1.0:
            updates["novelty_threshold_spontaneous"] = spontaneous_threshold
        if response_threshold is not None and 0.5 <= response_threshold <= 1.0:
            updates["novelty_threshold_response"] = response_threshold
        if pressure_threshold is not None and 0.5 <= pressure_threshold <= 0.9:
            updates["pressure_threshold"] = pressure_threshold
        if pressure_drop is not None and 0.1 <= pressure_drop <= 0.5:
            updates["pressure_drop"] = pressure_drop
        if silence_threshold is not None and 5.0 <= silence_threshold <= 60.0:
            updates["silence_threshold"] = silence_threshold
        if question_cooldown is not None and 30.0 <= question_cooldown <= 120.0:
            updates["question_cooldown"] = question_cooldown
        if queue_maxlen is not None and 5 <= queue_maxlen <= 20:
            updates["queue_maxlen"] = queue_maxlen
            self.unanswered_questions = deque(self.unanswered_questions, maxlen=queue_maxlen)
        if weight_ignorance is not None and 0.0 <= weight_ignorance <= 1.0:
            updates["weight_ignorance"] = weight_ignorance
            self.curiosity.weight_ignorance = weight_ignorance
        if weight_novelty is not None and 0.0 <= weight_novelty <= 1.0:
            updates["weight_novelty"] = weight_novelty
            self.curiosity.weight_novelty = weight_novelty
        if max_new_tokens is not None and 5 <= max_new_tokens <= 12:
            updates["max_new_tokens"] = max_new_tokens
        if base_temperature is not None and 0.5 <= base_temperature <= 1.5:
            updates["base_temperature"] = base_temperature
        if temperament_influence is not None and 0.1 <= temperament_influence <= 0.6:
            updates["temperament_influence"] = temperament_influence
        if metrics_maxlen is not None and 100 <= metrics_maxlen <= 10000:
            updates["metrics_maxlen"] = metrics_maxlen
            self.metrics_maxlen = metrics_maxlen
            self.metrics = deque(self.metrics, maxlen=metrics_maxlen)    
        if top_k is not None and 10 <= top_k <= 50:
            updates["top_k"] = top_k

        self.config.update(updates)
        if self.logger:
            self.logger.record({
                "event": "tune_curiosity",
                "params": updates,
                "timestamp": time.time()
            })

    def get_pressure(self) -> float:
        return self.pressure.value

    def reduce_pressure(self, amount: float) -> None:
        self.pressure.drop_pressure(amount)

    def save_state(self) -> Dict:
        return {
            "pressure": self.pressure.value,
            "last_question_time": self.last_question_time,
            "unanswered_questions": list(self.unanswered_questions),
            "metrics": list(self.metrics)  # Convert deque to list for serialization
        }

    def load_state(self, state_dict: Dict) -> None:
        # ... existing validation code ...
        metrics = state_dict.get("metrics", [])
        if not isinstance(metrics, list):
            metrics = []
        self.metrics = deque(metrics, maxlen=self.metrics_maxlen)

    def generate_curiosity_question(self, state: Any, tokenizer: Any, model: Any, context: Optional[str] = None, spontaneous: bool = False) -> Optional[str]:
        if not self.config.get("enable_curiosity", True):
            return None
        question = self.generate_question(state, tokenizer, model, context, spontaneous)
        if question:
            state.curiosity.update_question_history(question, time.time())
            if self.logger:
                self.logger.record({
                    "event": "curiosity_question",
                    "prompt": question,
                    "spontaneous": spontaneous,
                    "timestamp": time.time()
                })
        return question

    def check_silence(self, state: Any, tokenizer: Any, model: Any, elapsed: float) -> Optional[str]:
        if not self.config.get("enable_curiosity", True):
            return None
        state.curiosity.prune_old_questions(self.config.get("question_timeout", 3600.0))
        question = self.generate_question(state, tokenizer, model, spontaneous=True)
        if question:
            state.curiosity.update_question_history(question, time.time())
            if self.logger:
                self.logger.record({
                    "event": "silence_question",
                    "prompt": question,
                    "timestamp": time.time()
                })
        return question

    def tune_curiosity(self, **kwargs) -> None:
        updates = {}
        for key, value in kwargs.items():
            if key in self.config:
                updates[key] = value
                self.config[key] = value
        if updates:
            if self.logger:
                self.logger.record({
                    "event": "tune_curiosity",
                    "params": updates,
                    "timestamp": time.time()
                })
            self.tune(**updates)
