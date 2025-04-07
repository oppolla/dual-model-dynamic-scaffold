def _compute_salience(self, prompt: str, response: str) -> float:
    """
    Compute salience with human action signals, nonsense penalty, frustration boost, and emotional connection reward:
    - Intent: Purposeful keywords.
    - Emotion: General emotional weight.
    - Recency: Time since last interaction.
    - Frequency: Engagement density.
    - Complexity: Prompt effort.
    - Nonsense: Penalty for gibberish or trolling.
    - Frustration/Anger: Boost for genuine irritation (silent learning).
    - Emotional Connection: Reward for aligned emotional rapport.
    """
    intent_keywords = {"how", "why", "what", "please", "help"}
    emotion_keywords = {"great", "bad", "love", "hate", "sorry", "happy", "sad"}
    frustration_keywords = {"stupid", "wrong", "useless", "mad", "angry", "damn", "no", "not", "fail"}
    positive_emotions = {"great", "love", "happy"}  # For alignment
    negative_emotions = {"bad", "hate", "sorry", "sad", "mad", "angry"}
    dictionary_words = set(["the", "is", "to", "and", "you", "i", "how", "why", "what", "please", "help", 
                           "great", "bad", "love", "hate", "sorry", "happy", "sad", "stupid", "wrong", "useless"])
    
    prompt_words = set(prompt.lower().split())
    response_words = set(response.lower().split())
    prompt_text = prompt.lower()
    prompt_raw = prompt
    
    # Intent (25%)
    intent_score = sum(1 for kw in intent_keywords if kw in prompt_words) / len(intent_keywords)
    
    # Emotion (25%)
    emotion_score = sum(1 for kw in emotion_keywords if kw in prompt_words or kw in response_words) / len(emotion_keywords)
    
    # Recency (20%)
    time_delta = time.time() - self.last_activity
    if time_delta < 60:
        recency_score = 1.0
    elif time_delta < 600:
        recency_score = 1.0 - (time_delta - 60) / (600 - 60)
    else:
        recency_score = 0.1
    
    # Frequency (15%)
    current_time = time.time()
    if current_time - self.last_frequency_check >= 300:
        self.recent_interactions = 0
        self.last_frequency_check = current_time
    frequency_score = min(self.recent_interactions / 5.0, 1.0)
    
    # Complexity (15%)
    complexity_score = min(
        (prompt.count("?") + prompt.count("!") + prompt.count(".") + len(prompt_words)) / 10.0,
        1.0
    )
    
    # Nonsense Score
    if prompt_words:
        max_word_repeat = max([prompt_text.split().count(w) for w in prompt_words], default=0)
        repetition_score = min(max_word_repeat / 3.0, 1.0)
    else:
        repetition_score = 1.0
    real_word_ratio = sum(1 for w in prompt_words if w in dictionary_words) / max(len(prompt_words), 1)
    randomness_score = 1.0 - real_word_ratio
    punct_count = prompt.count("?") + prompt.count("!") + prompt.count(".")
    punct_ratio = punct_count / max(len(prompt_words), 1)
    punctuation_score = min(punct_ratio / 2.0, 1.0)
    nonsense_score = (repetition_score + randomness_score + punctuation_score) / 3.0
    
    # Frustration/Anger Score
    frustration_score = sum(1 for kw in frustration_keywords if kw in prompt_words) / len(frustration_keywords)
    caps_ratio = sum(c.isupper() for c in prompt_raw) / max(len(prompt_raw), 1)
    caps_score = 1.0 if caps_ratio > 0.7 else 0.0
    punct_excess_score = 1.0 if punct_count > 3 and punct_ratio > 1.0 else 0.0
    frustration_repeat_score = 0.0
    if self.interaction_buffer:
        last_prompt = self.interaction_buffer[-1]["prompt"].lower().split()
        overlap = len(set(last_prompt) & prompt_words) / max(len(last_prompt), len(prompt_words), 1)
        if overlap > 0.5:
            frustration_repeat_score = 0.5
    frustration_total = max(frustration_score, caps_score, punct_excess_score, frustration_repeat_score)
    
    # Emotional Connection Score (0-1): Reward aligned emotional rapport
    # 1. Detect prompt emotion polarity
    prompt_positive = sum(1 for kw in positive_emotions if kw in prompt_words)
    prompt_negative = sum(1 for kw in negative_emotions if kw in prompt_words)
    prompt_polarity = 1 if prompt_positive > prompt_negative else (-1 if prompt_negative > prompt_positive else 0)
    
    # 2. Detect response emotion polarity
    response_positive = sum(1 for kw in positive_emotions if kw in response_words)
    response_negative = sum(1 for kw in negative_emotions if kw in response_words)
    response_polarity = 1 if response_positive > response_negative else (-1 if response_negative > response_positive else 0)
    
    # 3. Alignment: Match polarities or complement (e.g., sad → sorry)
    alignment_score = 0.0
    if prompt_polarity == response_polarity and prompt_polarity != 0:  # Matching positive/negative
        alignment_score = min((prompt_positive + response_positive + prompt_negative + response_negative) / 4.0, 1.0)
    elif prompt_polarity == -1 and "sorry" in response_words:  # Negative prompt, sympathetic response
        alignment_score = min((prompt_negative + response_negative) / 4.0, 1.0)
    
    # Base salience
    base_salience = (
        0.25 * intent_score +
        0.25 * emotion_score +
        0.20 * recency_score +
        0.15 * frequency_score +
        0.15 * complexity_score
    )
    
    # Apply nonsense penalty
    adjusted_salience = base_salience * (1.0 - 0.9 * nonsense_score)
    
    # Apply frustration boost (silent, up to 50%)
    frustration_adjusted = adjusted_salience * (1.0 + 0.5 * frustration_total * (1.0 - min(nonsense_score / 0.3, 1.0)))
    
    # Apply emotional connection reward (up to 30%, only if frustration low)
    # Guard: No boost if frustration high (> 0.3) or nonsense high (> 0.3)
    connection_guard = 1.0 - min(max(frustration_total, nonsense_score) / 0.3, 1.0)
    final_salience = frustration_adjusted * (1.0 + 0.3 * alignment_score * connection_guard)
    
    return min(max(final_salience, 0.0), 1.0)

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. It is fully open-sourced under the [MIT License] (we sincerely appreciate all attributions and readily accept most contributions, but please don’t hold us liable).

# SKETCH

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AdamW, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
import threading
import time
import json
import psutil
import random
from typing import List, Dict, Optional

# ... [Assume existing config globals like BASE_MODEL_NAME, LEARNING_RATE, etc.] ...

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BareBonesDMAO_Learn:
    def __init__(self):
        # --- Existing Model Setup ---
        self.base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME).to(DEVICE).eval()
        self.base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        
        scaffold_model_raw = AutoModelForCausalLM.from_pretrained(SCAFFOLD_MODEL_NAME).to(DEVICE)
        lora_config = LoraConfig(
            r=LORA_RANK, target_modules=["q_proj", "v_proj"], lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT
        )
        self.scaffold_model = get_peft_model(scaffold_model_raw, lora_config)
        self.scaffold_tokenizer = AutoTokenizer.from_pretrained(SCAFFOLD_MODEL_NAME)
        if self.scaffold_tokenizer.pad_token is None:
            self.scaffold_tokenizer.pad_token = self.scaffold_tokenizer.eos_token
        
        self.optimizer = None
        self.dry_run = False
        self.dry_run_params = {}
        
        # --- Logging and Sleep Learning Setup ---
        self.interaction_buffer: List[Dict] = []  # In-memory log storage
        self.log_file = "interactions.json"      # Persistent storage
        self.lock = threading.Lock()             # Thread safety for logging/training
        self.last_activity = time.time()         # Tracks last interaction for recency
        self.idle_threshold = 0.7                # Max CPU load for idle (70%)
        self.min_idle_time = 300                 # Check interval (5 mins)
        self.min_examples = 50                   # Min logs to trigger training
        self.max_training_time = 1800            # Max training time (30 mins)
        self.salience_threshold = 0.5            # Min salience for training data
        self.last_trained = 0                    # Last training timestamp
        self.time_window = 86400                 # Retrain every 24 hours
        self.training_active = False             # Flag to prevent concurrent training
        self.recent_interactions = 0             # Count of interactions in last 5 mins
        self.last_frequency_check = time.time()  # Time of last frequency reset
        
        self._insert_cross_attention()           # Existing fusion setup
        self._start_sleep_scheduler()            # Start background training
        print("System initialized with logging and sleep learning.")

    # ... [Existing _insert_cross_attention, setup_optimizer unchanged] ...

    def log_interaction(self, prompt: str, response: str):
        """Log an interaction with a salience score reflecting human action"""
        with self.lock:
            salience = self._compute_salience(prompt, response)
            interaction = {
                "prompt": prompt,
                "response": response,
                "timestamp": time.time(),
                "salience": salience
            }
            self.interaction_buffer.append(interaction)
            self.recent_interactions += 1  # Bump frequency counter for engagement density
            if len(self.interaction_buffer) > 1000:  # Cap memory, offload to disk
                self._save_to_file()

    def _compute_salience(self, prompt: str, response: str) -> float:
        """
        Compute salience based on multiple human action signals:
        - Intent: Keywords indicating purpose (questions, commands).
        - Emotion: Emotional weight in words.
        - Recency: Time since last interaction (quick replies = higher salience).
        - Frequency: Number of interactions in last 5 mins (engagement density).
        - Complexity: Prompt effort via punctuation and unique words.
        """
        # Define keyword sets for intent and emotion detection
        intent_keywords = {"how", "why", "what", "please", "help"}
        emotion_keywords = {"great", "bad", "love", "hate", "sorry", "happy", "sad"}
        
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        
        # Intent Score (25%): Detects purposeful inputs like questions or requests
        # Higher if user’s asking or directing—signals engagement
        intent_score = sum(1 for kw in intent_keywords if kw in prompt_words) / len(intent_keywords)
        
        # Emotion Score (25%): Captures emotional weight in prompt or response
        # Reflects user’s mood or reaction—key for meaningful interactions
        emotion_score = sum(1 for kw in emotion_keywords if kw in prompt_words or kw in response_words) / len(emotion_keywords)
        
        # Recency Score (20%): Prioritizes quick back-and-forths
        # Short gaps = active convo, long gaps = potential disengagement
        time_delta = time.time() - self.last_activity
        if time_delta < 60:  # < 1 min: Full boost
            recency_score = 1.0
        elif time_delta < 600:  # 1-10 mins: Linear decay
            recency_score = 1.0 - (time_delta - 60) / (600 - 60)
        else:  # > 10 mins: Low salience
            recency_score = 0.1
        
        # Frequency Score (15%): Measures engagement density
        # More interactions in 5 mins = higher salience, caps at 5 for 1.0
        current_time = time.time()
        if current_time - self.last_frequency_check >= 300:  # Reset every 5 mins
            self.recent_interactions = 0
            self.last_frequency_check = current_time
        frequency_score = min(self.recent_interactions / 5.0, 1.0)
        
        # Complexity Score (15%): Gauges effort in prompt via punctuation and variety
        # More structure or unique words = more thought, capped for balance
        complexity_score = min(
            (prompt.count("?") + prompt.count("!") + prompt.count(".") + len(prompt_words)) / 10.0,
            1.0
        )
        
        # Combine with weights, cap at 1.0
        # Balanced to value content (intent/emotion) and behavior (recency/frequency/complexity)
        return min(
            0.25 * intent_score +
            0.25 * emotion_score +
            0.20 * recency_score +
            0.15 * frequency_score +
            0.15 * complexity_score,
            1.0
        )

    def _save_to_file(self):
        """Dump interaction buffer to file and clear memory"""
        with self.lock:
            if self.interaction_buffer:
                with open(self.log_file, "a") as f:
                    for interaction in self.interaction_buffer:
                        json.dump(interaction, f)
                        f.write("\n")
                self.interaction_buffer.clear()

    def _load_from_file(self) -> List[Dict]:
        """Load all logged interactions from file"""
        interactions = []
        try:
            with open(self.log_file, "r") as f:
                for line in f:
                    interactions.append(json.loads(line.strip()))
        except FileNotFoundError:
            pass
        return interactions

    def is_idle(self) -> bool:
        """Check if system is idle based on CPU load and inactivity"""
        cpu_load = psutil.cpu_percent(interval=1) / 100
        time_since_activity = time.time() - self.last_activity
        return cpu_load < self.idle_threshold and time_since_activity >= self.min_idle_time

    def _should_train(self) -> bool:
        """Determine if sleep training should run"""
        with self.lock:
            all_interactions = self.interaction_buffer + self._load_from_file()
            if len(all_interactions) < self.min_examples:
                return False
            avg_salience = sum(i["salience"] for i in all_interactions) / len(all_interactions)
            recent_ok = (time.time() - self.last_trained) > self.time_window
            return self.is_idle() and avg_salience >= self.salience_threshold and recent_ok and not self.training_active

    def get_training_data(self) -> List[Dict]:
        """Prepare salient interactions for training"""
        with self.lock:
            all_interactions = self.interaction_buffer + self._load_from_file()
            salient_data = [
                {"prompt": i["prompt"], "completion": i["response"]}
                for i in all_interactions if i["salience"] >= self.salience_threshold
            ]
            random.shuffle(salient_data)
            self._save_to_file()
            return salient_data[:max(self.min_examples, len(salient_data))]

    def _start_sleep_scheduler(self):
        """Launch background thread for sleep learning"""
        def scheduler_loop():
            while True:
                time.sleep(self.min_idle_time)  # Check every 5 mins
                if self._should_train():
                    self._run_sleep_training()
        
        self.scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self.scheduler_thread.start()

    def _run_sleep_training(self):
        """Execute sleep training on logged data"""
        with self.lock:
            self.training_active = True
        try:
            print("\n--- Starting Sleep Training ---")
            train_data = self.get_training_data()
            if not train_data:
                print("No sufficient data for training.")
                return
            
            num_steps = len(train_data) // BATCH_SIZE
            if not self.optimizer:
                self.setup_optimizer(num_steps)
            
            start_time = time.time()
            self.scaffold_model.train()
            self.base_model.eval()
            
            for i in range(0, len(train_data), BATCH_SIZE):
                if time.time() - start_time > self.max_training_time:
                    raise TimeoutError("Sleep training exceeded time limit")
                batch = train_data[i:i + BATCH_SIZE]
                loss = self.train_step(batch)
                if loss is not None:
                    print(f"Step {i // BATCH_SIZE + 1}/{num_steps} | Loss: {loss:.4f}")
            
            self.last_trained = time.time()
            print("--- Sleep Training Completed ---")
        except Exception as e:
            print(f"Sleep training failed: {e}")
            self._rollback_training()
        finally:
            with self.lock:
                self.training_active = False

    def _rollback_training(self):
        """Revert scaffold to pre-training state on failure"""
        if self.optimizer:
            self.optimizer.zero_grad()
        self.scaffold_model.eval()
        print("Training rolled back.")

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 50, scaffold_weight: Optional[float] = None, **kwargs) -> str:
        """Generate response, log it, and update activity"""
        # ... [Existing dry run and generation logic] ...
        response = self.base_tokenizer.decode(generated_ids, skip_special_tokens=True)
        self.log_interaction(prompt, response)
        self.last_activity = time.time()  # Update for recency
        print(f"Generation took {end_time - start_time:.2f} seconds.")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return response

    # ... [Existing train_step, run_training_cycle, etc. unchanged] ...

if __name__ == "__main__":
    # ... [Existing main block with minimal changes] ...
    dmao_system = BareBonesDMAO_Learn()
    try:
        while True:
            user_cmd = input("\nEnter command or prompt: ")
            cmd = user_cmd.lower().strip()
            if cmd in ['quit', 'exit']:
                break
            elif cmd == 'train':
                dmao_system.run_training_cycle(TRAIN_DATA, VALID_DATA, epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE)
            else:
                response = dmao_system.generate(user_cmd, max_new_tokens=60, temperature=0.7)
                print("\nResponse:", response)
    finally:
        # ... [Existing cleanup] ...
