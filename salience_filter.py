def __init__(self):
    # ... [Existing __init__] ...
    self.interaction_buffer: List[Dict] = []
    self.log_file = "interactions.json"
    self.lock = threading.Lock()
    self.last_activity = time.time()
    self.idle_threshold = 0.7
    self.min_idle_time = 300
    self.min_examples = 50
    self.max_training_time = 1800
    self.salience_threshold = 0.5
    self.last_trained = 0
    self.time_window = 86400
    self.training_active = False
    self.recent_interactions = 0  # Count in last 5 mins
    self.last_frequency_check = time.time()
    self._start_sleep_scheduler()

def _compute_salience(self, prompt: str, response: str) -> float:
    """Compute salience with intent, emotion, recency, frequency, and complexity"""
    intent_keywords = {"how", "why", "what", "please", "help"}
    emotion_keywords = {"great", "bad", "love", "hate", "sorry", "happy", "sad"}
    
    prompt_words = set(prompt.lower().split())
    response_words = set(response.lower().split())
    
    # Intent: Questions or commands
    intent_score = sum(1 for kw in intent_keywords if kw in prompt_words) / len(intent_keywords)
    
    # Emotion: Emotional weight
    emotion_score = sum(1 for kw in emotion_keywords if kw in prompt_words or kw in response_words) / len(emotion_keywords)
    
    # Recency: Time since last interaction
    time_delta = time.time() - self.last_activity
    if time_delta < 60:  # < 1 min
        recency_score = 1.0
    elif time_delta < 600:  # 1-10 mins
        recency_score = 1.0 - (time_delta - 60) / (600 - 60)
    else:  # > 10 mins
        recency_score = 0.1
    
    # Frequency: Interactions in last 5 mins
    current_time = time.time()
    if current_time - self.last_frequency_check >= 300:  # Reset every 5 mins
        self.recent_interactions = 0
        self.last_frequency_check = current_time
    frequency_score = min(self.recent_interactions / 5.0, 1.0)  # Cap at 5 for 1.0
    
    # Complexity: Punctuation and unique words
    complexity_score = min(
        (prompt.count("?") + prompt.count("!") + prompt.count(".") + len(prompt_words)) / 10.0,
        1.0
    )
    
    # Weighted combination
    return min(
        0.25 * intent_score +  # Adjusted for more factors
        0.25 * emotion_score +
        0.2 * recency_score +
        0.15 * frequency_score +  # New: Engagement density
        0.15 * complexity_score,  # New: Effort in prompt
        1.0
    )

def log_interaction(self, prompt: str, response: str):
    """Log interaction with salience score"""
    with self.lock:
        salience = self._compute_salience(prompt, response)
        interaction = {
            "prompt": prompt,
            "response": response,
            "timestamp": time.time(),
            "salience": salience
        }
        self.interaction_buffer.append(interaction)
        self.recent_interactions += 1  # Bump frequency counter
        if len(self.interaction_buffer) > 1000:
            self._save_to_file()
