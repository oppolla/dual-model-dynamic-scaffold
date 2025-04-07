def _compute_salience(self, prompt: str, response: str) -> float:
    """Compute salience with intent, emotion, length, and recency"""
    intent_keywords = {"how", "why", "what", "please", "help"}
    emotion_keywords = {"great", "bad", "love", "hate", "sorry", "happy", "sad"}
    
    prompt_words = set(prompt.lower().split())
    response_words = set(response.lower().split())
    
    # Intent: Questions or commands
    intent_score = sum(1 for kw in intent_keywords if kw in prompt_words) / len(intent_keywords)
    
    # Emotion: Emotional weight in prompt or response
    emotion_score = sum(1 for kw in emotion_keywords if kw in prompt_words or kw in response_words) / len(emotion_keywords)
    
    # Length: Depth of interaction
    length_score = min(len(prompt) + len(response), 100) / 100.0
    
    # Recency: Time since last interaction (boost quick exchanges)
    time_delta = time.time() - self.last_activity  # Seconds since last
    if time_delta < 60:  # Within 1 min: Full boost
        recency_score = 1.0
    elif time_delta < 600:  # 1-10 mins: Linear decay
        recency_score = 1.0 - (time_delta - 60) / (600 - 60)
    else:  # > 10 mins: Minimal score
        recency_score = 0.1
    
    # Weighted combination
    return min(
        0.3 * intent_score +  # Lowered to fit recency
        0.3 * emotion_score + # Lowered to fit recency
        0.2 * length_score +  # Kept as is
        0.2 * recency_score,  # New time factor
        1.0
    )
