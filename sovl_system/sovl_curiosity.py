import random
import torch
import torch.nn.functional as F
from typing import List, Dict, Callable, Optional


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
        self, base_conf: float, scaf_conf: float, memory_embeddings: List[torch.Tensor], query_embedding: torch.Tensor
    ) -> float:
        """
        Compute curiosity as a weighted sum of ignorance and novelty.

        Args:
            base_conf (float): Base model confidence.
            scaf_conf (float): Scaffold model confidence.
            memory_embeddings (List[torch.Tensor]): Embeddings from memory.
            query_embedding (torch.Tensor): Embedding of the current query.

        Returns:
            float: The computed curiosity score.
        """
        mem_sim = max(
            [F.cosine_similarity(query_embedding, emb).item() for emb in memory_embeddings], default=0
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


class CuriositySystem:
    """
    A complete curiosity system that integrates computation, pressure, and callbacks.
    """
    def __init__(self, weight_ignorance: float = 0.5, weight_novelty: float = 0.5):
        """
        Initialize the CuriositySystem.

        Args:
            weight_ignorance (float): Weight for ignorance in curiosity computation.
            weight_novelty (float): Weight for novelty in curiosity computation.
        """
        self.curiosity = Curiosity(weight_ignorance, weight_novelty)
        self.pressure = CuriosityPressure()
        self.callbacks = CuriosityCallbacks()

    def compute_curiosity(
        self, base_conf: float, scaf_conf: float, memory_embeddings: List[torch.Tensor], query_embedding: torch.Tensor
    ) -> float:
        """
        Delegate curiosity computation to the Curiosity class.

        Args:
            base_conf (float): Base model confidence.
            scaf_conf (float): Scaffold model confidence.
            memory_embeddings (List[torch.Tensor]): Embeddings from memory.
            query_embedding (torch.Tensor): Embedding of the current query.

        Returns:
            float: The computed curiosity score.
        """
        return self.curiosity.compute_curiosity(base_conf, scaf_conf, memory_embeddings, query_embedding)

    def update_pressure(self, temperament: float, confidence: float, silence: float):
        """
        Update the curiosity pressure.

        Args:
            temperament (float): Temperament value.
            confidence (float): Confidence level.
            silence (float): Silence level.
        """
        self.pressure.update(temperament, confidence, silence)

    def check_pressure_eruption(self, threshold: float) -> bool:
        """
        Check if the curiosity pressure should erupt.

        Args:
            threshold (float): The threshold for eruption.

        Returns:
            bool: True if the pressure erupts, False otherwise.
        """
        return self.pressure.should_erupt(threshold)

    def register_callback(self, event: str, callback: Callable):
        """
        Register a callback for a curiosity-related event.

        Args:
            event (str): The name of the event.
            callback (Callable): The callback function.
        """
        self.callbacks.register_callback(event, callback)

    def trigger_event(self, event: str, *args, **kwargs):
        """
        Trigger a curiosity-related event.

        Args:
            event (str): The name of the event.
            *args: Positional arguments for the event.
            **kwargs: Keyword arguments for the event.
        """
        self.callbacks.trigger_callback(event, *args, **kwargs)