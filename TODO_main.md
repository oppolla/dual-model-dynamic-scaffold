- If initial test sucessful, add the sleep learning phase to system for live updates.
- Add Logging and Monitoring fuctions
- Make a Web UI User Interface
- Seperate Configuration Management into own file (config.JSON) - DONE / CONSTANT UPDATE AS NEW CONFIGS ADDED
- Adaptive Layer Selection (Automatically adapts to different model sizes) - DONE
- Add Token Mapping Cache (faster token mapping for repeated tokens)
- Optional Just-In-Time Scaffold Loading (big efficiency boost) - DONE
- Quantized Inference Mode (big efficiency boost) - DONE
- Add Dry Run mode for ease of testing

TESTING GOALS:
- def check_memory_health(self): - Find the optimal value for memory limit intervention (default 0.85 / 85% )
Learning Rate:

- The rate at which the model updates weights during training. A balance is critical to avoid overshooting (too high) or stalling (too low).
Temperature (if applicable in the LLM context):

- Controls the randomness of predictions. Higher values produce more diverse outputs, while lower values make outputs more deterministic.
Batch Size:

- The number of samples processed before the model updates. Larger batch sizes can stabilize learning but may require more computational resources.
Weight Initialization:

- Ensures that the network starts with a configuration conducive to effective learning. Poor initialization can lead to vanishing or exploding gradients.
Dropout Rate (if applicable):

- Regularization parameter to prevent overfitting by randomly dropping connections during training.
Epochs:

- The number of complete passes through the dataset. You need to monitor for overfitting or underfitting to decide the optimal number.
Loss Function:

- Verify if the selected loss function aligns with the system’s goals. Certain tasks may benefit from custom loss functions.
Gradient Clipping:

- Helps stabilize training in systems prone to exploding gradients.

  
Dynamic Adaptation Parameters:
If the SOVL system uses a dynamic second LLM ("scaffolded second dynamic LLM"), focus on parameters that govern the interaction between the base LLM and the scaffolded LLM, such as:
- Synchronization intervals.
- Weight-sharing mechanisms.
- Memory retention and sleep cycle tuning.

Evaluation Metrics:
- Define clear metrics for success (e.g., accuracy, precision, recall, or task-specific metrics).
Environment-Specific Parameters:
- Adaptation to the virtual environment, such as context window size, interaction frequency, or reward signal design (if reinforcement learning is involved).
  
Sleep Mechanism Tuning:
For the continuous learning sleep mechanism, consider:
- Frequency and duration of "sleep" phases.
- Data retention or replay during the sleep phase.
- Weight consolidation strategies to retain learned information.
- Run experiments systematically by isolating one parameter at a time to understand its impact, and consider using grid search, random search, or Bayesian optimization for hyperparameter tuning.
