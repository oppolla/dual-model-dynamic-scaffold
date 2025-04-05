- Add blend function to CrossAttentionFuser (0-1, default 0.5)
- Confidence Threshold Tuning for CrossAttentionFuser
- Attention Sparsity Control for SparseCrossAttention
- Add batch size parameter for _train_epoch
- Add Learning Rate Scheduler (dynamic learning rate that adjusts based on training progress or data salience)
- Add Training Data Weighting (control over sample weighting (e.g., salience-based or recency-based)
- More control over scaffold connections (LoRA Rank Adjustment Rate, LoRA Target Module Selection)
- Final output Response Length Control
- Add Buffer Size Limit for interaction_buffer for stablity
- Leave ASCSystem open for hook additions
- Scaffold Reset Trigger (take model out behind the shed)
- 





