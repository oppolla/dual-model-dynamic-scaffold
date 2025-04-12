import random
import torch
import torch.nn.functional as F
from typing import List, Dict, Callable, Optional, Tuple, Deque
from collections import deque

class Curiosity:
    """
    Handles curiosity computation based on ignorance and novelty.
    """
    def __init__(self, weight_ignorance: float = 0.5, weight_novelty: float = 0.5):
        """
        Initialize Curiosity with weights for ignorance and novelty.

        Args:
            weight_ignorance (float): Weight for ignorance in curiosity computation.
            weight_novelty (float): Weight for novelty in curiosity computation.
        """
        self.weight_ignorance = weight_ignorance
        self.weight_novelty = weight_novelty

    def compute_curiosity(
        self, base_conf: float, scaf_conf: float, memory_embeddings: List[Tuple[torch.Tensor, float]], query_embedding: torch.Tensor
    ) -> float:
        """
        Compute curiosity as a weighted sum of ignorance and novelty.

        Args:
            base_conf (float): Base model confidence.
            scaf_conf (float): Scaffold model confidence.
            memory_embeddings (List[Tuple[torch.Tensor, float]]): Embeddings and weights from memory.
            query_embedding (torch.Tensor): Embedding of the current query.

        Returns:
            float: The computed curiosity score.
        """
        mem_sim = max(
            [F.cosine_similarity(query_embedding, emb).item() for emb, _ in memory_embeddings], default=0
        )
        ignorance = 1 - max(base_conf, scaf_conf)
        novelty = 1 - mem_sim
        return (
            ignorance * self.weight_ignorance +
            novelty * self.weight_novelty
        )

class CuriosityPressure:
    """
    Manages curiosity pressure levels and eruption thresholds.
    """
    def __init__(self):
        """
        Initialize the CuriosityPressure with default value.
        """
        self.value = 0.0

    def update(self, temperament: float, confidence: float, silence: float):
        """
        Update the pressure value based on temperament, confidence, and silence.

        Args:
            temperament (float): Temperament value.
            confidence (float): Confidence level.
            silence (float): Silence level.
        """
        self.value += (temperament * 0.1 + (1 - confidence) * 0.05 + silence * 0.02)
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

    def drop_pressure(self, drop_amount: float):
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
    def __init__(self):
        """
        Initialize the callback manager with empty callback registry.
        """
        self.callbacks = {}

    def register_callback(self, event: str, callback: Callable):
        """
        Register a callback for a specific event.

        Args:
            event (str): The name of the event.
            callback (Callable): The callback function.
        """
        self.callbacks[event] = callback

    def trigger_callback(self, event: str, *args, **kwargs):
        """
        Trigger a registered callback.

        Args:
            event (str): The name of the event.
            *args: Positional arguments for the callback.
            **kwargs: Keyword arguments for the callback.
        """
        if event in self.callbacks:
            self.callbacks[event](*args, **kwargs)
        else:
            print(f"No callback registered for event: {event}")

class CuriosityManager:
    """
    Orchestrates curiosity computation, pressure management, and state interactions.
    """
    def __init__(
        self,
        weight_ignorance: float = 0.5,
        weight_novelty: float = 0.5,
        pressure_threshold: float = 0.7,
        pressure_drop: float = 0.3,
        novelty_threshold_spontaneous: float = 0.9,
        novelty_threshold_response: float = 0.8,
        silence_threshold: float = 20.0,
        question_cooldown: float = 60.0,
        queue_maxlen: int = 10,
        max_new_tokens: int = 8,
        base_temperature: float = 1.1,
        temperament_influence: float = 0.4,
        top_k: int = 30
    ):
        """
        Initialize the CuriosityManager.

        Args:
            weight_ignorance (float): Weight for ignorance in curiosity computation.
            weight_novelty (float): Weight for novelty in curiosity computation.
            pressure_threshold (float): Threshold for pressure eruption.
            pressure_drop (float): Amount to drop pressure after eruption.
            novelty_threshold_spontaneous (float): Threshold for spontaneous curiosity.
            novelty_threshold_response (float): Threshold for response-driven curiosity.
            silence_threshold (float): Threshold for silence contribution to pressure.
            question_cooldown (float): Cooldown period for question generation.
            queue_maxlen (int): Maximum length of unanswered question queue.
            max_new_tokens (int): Maximum tokens for generated questions.
            base_temperature (float): Base temperature for question generation.
            temperament_influence (float): Influence of temperament on curiosity.
            top_k (int): Top-k sampling for question generation.
        """
        self.curiosity = Curiosity(weight_ignorance, weight_novelty)
        self.pressure = CuriosityPressure()
        self.callbacks = CuriosityCallbacks()
        self.pressure_threshold = pressure_threshold
        self.pressure_drop = pressure_drop
        self.novelty_threshold_spontaneous = novelty_threshold_spontaneous
        self.novelty_threshold_response = novelty_threshold_response
        self.silence_threshold = silence_threshold
        self.question_cooldown = question_cooldown
        self.queue_maxlen = queue_maxlen
        self.max_new_tokens = max_new_tokens
        self.base_temperature = base_temperature
        self.temperament_influence = temperament_influence
        self.top_k = top_k
        self.last_question_time = 0.0

    def compute_curiosity(
        self,
        base_conf: float,
        scaf_conf: float,
        memory_embeddings: List[Tuple[torch.Tensor, float]],
        query_embedding: torch.Tensor
    ) -> float:
        """
        Compute curiosity score.

        Args:
            base_conf (float): Base model confidence.
            scaf_conf (float): Scaffold model confidence.
            memory_embeddings (List[Tuple[torch.Tensor, float]]): Embeddings and weights from memory.
            query_embedding (torch.Tensor): Embedding of the current query.

        Returns:
            float: The computed curiosity score.
        """
        score = self.curiosity.compute_curiosity(base_conf, scaf_conf, memory_embeddings, query_embedding)
        self.callbacks.trigger_callback("curiosity_computed", score=score)
        return score

    def update_pressure(self, temperament: float, confidence: float, silence: float):
        """
        Update curiosity pressure.

        Args:
            temperament (float): Temperament value.
            confidence (float): Confidence level.
            silence (float): Silence level.
        """
        self.pressure.update(temperament, confidence, silence)
        self.callbacks.trigger_callback("pressure_updated", pressure=self.pressure.value)

    def check_pressure_eruption(self) -> bool:
        """
        Check if curiosity pressure should erupt.

        Returns:
            bool: True if pressure erupts, False otherwise.
        """
        erupted = self.pressure.should_erupt(self.pressure_threshold)
        if erupted:
            self.pressure.drop_pressure(self.pressure_drop)
            self.callbacks.trigger_callback("pressure_erupted", pressure=self.pressure.value)
        return erupted

    def generate_question(
        self,
        state: Optional[object] = None,
        tokenizer: Optional[Callable] = None,
        model: Optional[torch.nn.Module] = None,
        prompt: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate a curiosity-driven question if conditions are met.

        Args:
            state (Optional[object]): SOVLState instance for accessing memory and history.
            tokenizer (Optional[Callable]): Tokenizer for encoding prompts.
            model (Optional[torch.nn.Module]): Model for generating questions.
            prompt (Optional[str]): Prompt to base the question on.

        Returns:
            Optional[str]: Generated question or None if conditions not met.
        """
        if not (state and tokenizer and model):
            return None

        current_time = state.global_step  # Assuming global_step tracks time-like progression
 if current_time - self.last_question_time < self.question_cooldown:
            return None

        curiosity_score = self.compute_curiosity(
            base_conf=state.confidence_history[-1] if state.confidence_history else 0.5,
            scaf_conf=state.sleep_confidence_sum / state.sleep_confidence_count if state.sleep_confidence_count > 0 else 0.5,
            memory_embeddings=list(state.dream_memory),
            query_embedding=state.last_prompt_embedding if state.last_prompt_embedding is not None else torch.zeros(state.hidden_size)
        )

        # Determine if question should be generated
        is_spontaneous = curiosity_score >= self.novelty_threshold_spontaneous
        is_response_driven = curiosity_score >= self.novelty_threshold_response and prompt is not None
        if not (is_spontaneous or is_response_driven):
            return None

        # Select base prompt
        base_prompt = prompt if prompt else (
            state.history.messages[-1]["prompt"] if state.history.messages else "What is this about?"
        )

        # Generate question
        inputs = tokenizer(
            base_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.base_temperature + self.temperament_influence * (state.temperament_score if state else 0.0),
                top_k=self.top_k,
                do_sample=True
            )
        question = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Update state
        state.unanswered_q.append((question, curiosity_score))
        if len(state.unanswered_q) > self.queue_maxlen:
            state.unanswered_q.popleft()
        self.last_question_time = current_time
        self.pressure.drop_pressure(self.pressure_drop)

        self.callbacks.trigger_callback("question_generated", question=question, score=curiosity_score)
        return question

    def register_callback(self, event: str, callback: Callable):
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

    def set_pressure(self, value: float):
        """
        Set pressure value.

        Args:
            value (float): New pressure value.
        """
        self.pressure.value = max(0.0, min(1.0, value))
