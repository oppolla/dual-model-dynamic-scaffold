# TODO:

## General Taks:

- Migrate events module
- further develop initialization and startup feel

- Model Evaluation Mode
   What's Missing: There’s no dedicated mode or command for evaluating the model on a test dataset (separate from training or generation). This is crucial for assessing final model performance.
   Suggestion:
Add an evaluate command to COMMAND_CATEGORIES under "Testing" or "Evaluation".

Implement a method to load test data and compute metrics (similar to the validation loop).

Example:
python

def evaluate_model(self, model, test_data, device: str) -> Dict[str, float]:
    """Evaluate model on test data."""
    model.eval()
    metrics = {"test_loss": 0.0, "test_accuracy": 0.0}
    with torch.no_grad():
        for batch in test_data:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            metrics["test_loss"] += loss.item()
            # Add accuracy or other metrics
    metrics["test_loss"] /= len(test_data)
    self.logger.log_event(
        event_type="evaluation",
        message=f"Test metrics: {metrics}",
        level="info"
    )
    return metrics

Add a command-line argument for test data:
python

parser.add_argument("--test-data", help="Path to test data file")



## Module interaction clean progress

### With sovl_main.py

- sovl_logger - DONE
- sovl_state - DONE
- sovl_scaffold - DONE
- sovl_config - DONE
- sovl_io - DONE
- sovl_temperament - DONE
- sovl_generation - DONE
