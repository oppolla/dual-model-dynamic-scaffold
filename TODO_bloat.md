TODO:
- Add batch size parameter for _train_epoch
- Add Learning Rate Scheduler (dynamic learning rate that adjusts based on training progress or data salience)
- Add Training Data Weighting control (control over sample weighting (e.g., salience-based or recency-based)
- More control over scaffold connections (LoRA Rank Adjustment Rate, LoRA Target Module Selection)
- Add Buffer Size Limit for interaction_buffer for stablity
- Leave ASCSystem open for hook additions
- Add control / options for layer connection [8, 16 24], [13, 34. 66], [11, 45, 47, 50, 71] etc


DONE:
- Add blend function to CrossAttentionFuser (0-1, default 0.5) - DONE
- Confidence Threshold Tuning for CrossAttentionFuser - DONE
- Attention Sparsity Control for SparseCrossAttention - DONE
- Final output Response Length Control - DONE
- Scaffold Reset Trigger (kill the scaffold model) - DONE
- Split Temperment from Long-Term memory in log data training
- Add Regularization to loss function to mitigate overfitting - DONE
- Add data augmentor that creates syntheyic days based real logs - DONE
