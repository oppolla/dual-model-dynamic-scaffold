import random
import time
import torch
import torch.nn.functional as F
from typing import List, Dict, Callable, Optional, Tuple, Any
from collections import deque
from sovl_logger import Logger
from sovl_state import SOVLState

class Curiosity:
    """
    Handles curiosity computation based on ignorance and novelty.
    """
    def __init__(self, weight_ignorance: float = 0.5, weight_novelty: float = 0.5, logger: Optional[Logger] = None):
        """
        Initialize Curiosity with weights for ignorance and novelty.

        Args:
            weight_ignorance (float): Weight for ignorance in curiosity computation.
            weight_novelty (float): Weight for novelty in curiosity computation.
            logger (Optional[Logger]): Logger for error reporting.
        """
        self.weight_ignorance = weight_ignorance
        self.weight_novelty = weight_novelty
        self.logger = logger

    def compute_curiosity(
        self,
        base_conf: float,
        scaf_conf: float,
        memory_embeddings: List[Tuple[torch.Tensor, float]],
        query_embedding: torch.Tensor,
        device: torch.device
    ) -> float:
        """
        Compute curiosity as a weighted sum of ignorance and novelty.

        Args:
            base_conf (float): Base model confidence.
            scaf_conf (float): Scaffold model confidence.
            memory_embeddings (List[Tuple[torch.Tensor, float]]): Embeddings and weights from memory.
            query_embedding (torch.Tensor): Embedding of the current query.
            device (torch.device): Device for tensor operations.

        Returns:
            float: The computed curiosity score.
        """
        try:
            if query_embedding is None or query_embedding.numel() == 0:
                if self.logger:
                    self.logger.record({"warning": "Empty query_embedding in compute_curiosity", "timestamp": time.time()})
                return 0.5

            # Move query_embedding to correct device
            query_embedding = query_embedding.to(device)

            # Compute max similarity with memory embeddings
            mem_sim = 0.0
            if memory_embeddings:
                for emb, _ in memory_embeddings:
                    emb = emb.to(device)
                    if emb.shape == query_embedding.shape:
                        sim = F.cosine_similarity(query_embedding, emb, dim=-1).item()
                        mem_sim = max(mem_sim, sim)
                    else:
                        if self.logger:
                            self.logger.record({
                                "warning": f"Shape mismatch in memory_embedding: {emb.shape} vs {query_embedding.shape}",
                                "timestamp": time.time()
                            })

            ignorance = 1.0 - max(min(base_conf, 1.0), min(scaf_conf, 1.0), 0.0)
            novelty = 1.0 - mem_sim
            score = (ignorance * self.weight_ignorance + novelty * self.weight_novelty)
            return max(0.0, min(1.0, score))

        except Exception as e:
            if self.logger:
                self.logger.record({
                    "error": f"Error in compute_curiosity: {str(e)}",
                    "timestamp": time.time()
                })
            return 0.5

class CuriosityPressure:
    """
    Manages curiosity pressure levels and eruption thresholds.
    """
    def __init__(self, initial_value: float = 0.0):
        """
        Initialize the CuriosityPressure.

        Args:
            initial_value (float): Initial pressure value.
        """
        self.value = max(0.0, min(1.0, initial_value))

    def update(self, temperament: float, confidence: float, silence: float, silence_threshold: float) -> None:
        """
        Update the pressure value based on temperament, confidence, and silence.

        Args:
            temperament (float): Temperament value.
            confidence (float): Confidence level.
            silence (float): Silence duration (seconds).
            silence_threshold (float): Threshold to normalize silence.
        """
        temperament = max(0.0, min(1.0, temperament))
        confidence = max(0.0, min(1.0, confidence))
        normalized_silence = min(silence / silence_threshold, 1.0) if silence_threshold > 0 else 0.0
        self.value += (temperament * 0.1 + (1 - confidence) * 0.05 + normalized_silence * 0.02)
        self.value = max(0.0, min(1.0, self.value))

    def should_erupt(self, threshold: float) -> bool:
        """
        Determine if the curiosity pressure should trigger an event.

        Args:
            threshold (float): The threshold for eruption.

        Returns:
            bool: True if the pressure exceeds the threshold, False otherwise.
        """
        return self.value > threshold and random.random() < 0.3

    def drop_pressure(self, drop_amount: float) -> None:
        """
        Reduce pressure by a specified amount.

        Args:
            drop_amount (float): Amount to reduce pressure by.
        """
        self.value = max(0.0, self.value - drop_amount)

class CuriosityCallbacks:
    """
    Manages callbacks for curiosity-related events.
    """
    def __init__(self, logger: Optional[Logger] = None):
        """
        Initialize the callback manager.

        Args:
            logger (Optional[Logger]): Logger for unregistered events.
        """
        self.callbacks: Dict[str, List[Callable]] = {}
        self.logger = logger

    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register a callback for a specific event.

        Args:
            event (str): The name of the event.
            callback (Callable): The callback function.
        """
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)

    def trigger_callback(self, event: str, *args, **kwargs) -> None:
        """
        Trigger all registered callbacks for an event.

        Args:
            event (str): The name of the event.
            *args: Positional arguments for the callbacks.
            **kwargs: Keyword arguments for the callbacks.
        """
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    if self.logger:
                        self.logger.record({
                            "error": f"Callback error for {event}: {str(e)}",
                            "timestamp": time.time()
                        })
        elif self.logger:
            self.logger.record({
                "warning": f"No callback registered for event: {event}",
                "timestamp": time.time()
            })

class CuriosityManager:
    """
    Orchestrates curiosity computation, pressure management, and question generation.
    """
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[Logger] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the CuriosityManager.

        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary with curiosity parameters.
            logger (Optional[Logger]): Logger for events and errors.
            device (Optional[torch.device]): Device for tensor operations (default: cuda if available).
        """
        # Default configuration
        default_config = {
            "weight_ignorance": 0.5,
            "weight_novelty": 0.5,
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
            "default_hidden_size": 768  # For fallback embeddings
        }
        config = config or {}
        self.config = {**default_config, **config}

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
        self.last_question_time: float = 0.0

    def compute_curiosity(
        self,
        state: SOVLState,
        query_embedding: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute curiosity score using state information.

        Args:
            state (SOVLState): Current system state.
            query_embedding (Optional[torch.Tensor]): Optional embedding of current query.

        Returns:
            float: The computed curiosity score.
        """
        base_conf = state.confidence_history[-1] if state.confidence_history else 0.5
        scaf_conf = (
            state.sleep_confidence_sum / state.sleep_confidence_count
            if state.sleep_confidence_count > 0 else 0.5
        )
        
        query_emb = query_embedding if query_embedding is not None else state.last_prompt_embedding
        query_emb = query_emb if query_emb is not None else torch.zeros(
            self.config["default_hidden_size"], device=self.device
        )
        
        score = self.curiosity.compute_curiosity(
            base_conf=base_conf,
            scaf_conf=scaf_conf,
            memory_embeddings=list(state.dream_memory),
            query_embedding=query_emb,
            device=self.device
        )
        
        # Update state with new score
        state.curiosity.novelty_scores.append(score)
        self.callbacks.trigger_callback("curiosity_computed", score=score)
        return score

    def update_pressure(self, state: SOVLState) -> None:
        """
        Update curiosity pressure based on current state.

        Args:
            state (SOVLState): Current system state.
        """
        silence = getattr(state, "silence_duration", 0.0)
        self.pressure.update(
            temperament=state.temperament_score,
            confidence=state.confidence_history[-1] if state.confidence_history else 0.5,
            silence=silence,
            silence_threshold=self.config["silence_threshold"]
        )
        state.curiosity.pressure = self.pressure.value
        self.callbacks.trigger_callback("pressure_updated", pressure=self.pressure.value)

    def check_pressure_eruption(self, state: SOVLState) -> bool:
        """
        Check if curiosity pressure should erupt.

        Args:
            state (SOVLState): Current system state.

        Returns:
            bool: True if pressure erupts, False otherwise.
        """
        erupted = self.pressure.should_erupt(self.config["pressure_threshold"])
        if erupted:
            self.pressure.drop_pressure(self.config["pressure_drop"])
            state.curiosity.pressure = self.pressure.value
            state.curiosity.question_count += 1
            self.callbacks.trigger_callback("pressure_erupted", pressure=self.pressure.value)
        return erupted

    def generate_question(
        self,
        state: SOVLState,
        tokenizer: Optional[Callable] = None,
        model: Optional[torch.nn.Module] = None,
        prompt: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate a curiosity-driven question if conditions are met.

        Args:
            state (SOVLState): Current system state.
            tokenizer (Optional[Callable]): Tokenizer for encoding prompts.
            model (Optional[torch.nn.Module]): Model for generating questions.
            prompt (Optional[str]): Prompt to base the question on.

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
        if state.global_step - state.curiosity.last_question_time < self.config["question_cooldown"]:
            return None

        # Compute curiosity
        curiosity_score = self.compute_curiosity(state)

        # Check silence contribution
        silence = getattr(state, "silence_duration", 0.0)
        is_silence_driven = silence >= self.config["silence_threshold"]

        # Determine if question should be generated
        is_spontaneous = curiosity_score >= self.config["novelty_threshold_spontaneous"]
        is_response_driven = curiosity_score >= self.config["novelty_threshold_response"] and prompt is not None
        if not (is_spontaneous or is_response_driven or is_silence_driven):
            return None

        # Select base prompt
        base_prompt = prompt if prompt else (
            state.history.messages[-1]["prompt"] if state.history.messages else "What is this about?"
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
                    temperature=self.config["base_temperature"] + self.config["temperament_influence"] * state.temperament_score,
                    top_k=self.config["top_k"],
                    do_sample=True
                )
            question = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            # Update state
            state.curiosity.unanswered_questions.append((question, curiosity_score))
            state.curiosity.last_question_time = state.global_step
            state.curiosity.question_count += 1
            self.pressure.drop_pressure(self.config["pressure_drop"])
            state.curiosity.pressure = self.pressure.value

            self.callbacks.trigger_callback("question_generated", question=question, score=curiosity_score)
            return question

        except Exception as e:
            if self.logger:
                self.logger.record({
                    "error": f"Question generation failed: {str(e)}",
                    "timestamp": time.time()
                })
            return None

    def get_scaffold_influence_weight(self, curiosity_score: float) -> float:
        """
        Compute a scaffold influence weight based on curiosity.

        Args:
            curiosity_score (float): The curiosity score.

        Returns:
            float: Suggested weight for scaffold influence.
        """
        # Higher curiosity -> higher scaffold influence
        weight = 0.3 + 0.5 * curiosity_score  # Maps [0,1] to [0.3,0.8]
        return max(0.0, min(1.0, weight))

    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register a callback for a curiosity-related event.

        Args:
            event (str): The name of the event.
            callback (Callable): The callback function.
        """
        self.callbacks.register_callback(event, callback)

    def get_pressure(self) -> float:
        """
        Get current pressure value.

        Returns:
            float: Current pressure value.
        """
        return self.pressure.value

    def set_pressure(self, value: float) -> None:
        """
        Set pressure value.

        Args:
            value (float): New pressure value.
        """
        self.pressure.value = max(0.0, min(1.0, value))

    def save_state(self, state: SOVLState) -> None:
        """
        Save the current state to the provided SOVLState instance.

        Args:
            state (SOVLState): The state instance to save to.
        """
        state.curiosity.pressure = self.pressure.value
        state.curiosity.last_question_time = self.last_question_time

    def load_state(self, state: SOVLState) -> None:
        """
        Load state from a SOVLState instance.

        Args:
            state (SOVLState): The state instance to load from.
        """
        self.pressure.value = state.curiosity.pressure
        self.last_question_time = state.curiosity.last_question_time

    def reset(self, state: SOVLState) -> None:
        """
        Reset curiosity manager state.

        Args:
            state (SOVLState): The state instance to reset.
        """
        self.pressure.value = 0.0
        self.last_question_time = 0.0
        state.curiosity = CuriosityState()
