import numpy as np
import matplotlib.pyplot as plt
import torch
from copy import deepcopy
from torch.optim.lr_scheduler import LambdaLR
from typing import List, Tuple

class LRFinder:
    @staticmethod
    def find_lr(
        model,
        train_data: list,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        start_lr: float = 1e-7,
        end_lr: float = 10,
        num_iters: int = 100,
        batch_size: int = 2,
        accumulation_steps: int = 4
    ) -> Tuple[List[float], List[float]]:
        """
        Learning rate range test implementation
        
        Args:
            model: Your training model instance
            train_data: List of training samples
            optimizer: Configured optimizer
            device: Target device (cuda/cpu)
            start_lr: Initial learning rate
            end_lr: Maximum learning rate
            num_iters: Number of iterations to run
            batch_size: Batch size for test
            accumulation_steps: Gradient accumulation steps
            
        Returns:
            Tuple of (learning rates, losses)
        """
        lr_lambda = lambda step: start_lr * (end_lr/start_lr)**(step/num_iters)
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        lrs, losses = [], []
        best_loss = float('inf')
        
        model.train()
        optimizer.zero_grad()
        
        try:
            for iter_num in range(num_iters):
                batch = random.sample(train_data, batch_size)
                loss = model.train_step(batch)
                
                if loss is None:
                    continue
                    
                (loss/accumulation_steps).backward()
                
                if (iter_num + 1) % accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                current_lr = optimizer.param_groups[0]['lr']
                lrs.append(current_lr)
                losses.append(loss.item())
                
                if loss.item() > 4 * best_loss:
                    break
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    
        except Exception as e:
            print(f"LR finder interrupted: {str(e)}")
            
        return lrs, losses

    @staticmethod
    def plot_lr_results(
        lrs: List[float],
        losses: List[float],
        skip_start: int = 10,
        skip_end: int = 5
    ) -> Tuple[float, float]:
        """
        Plot and analyze LR finder results
        
        Returns:
            Tuple of (suggested_min_lr, suggested_max_lr)
        """
        if len(lrs) <= skip_start + skip_end:
            raise ValueError("Not enough data to plot")
            
        # Smooth losses
        smoothed_losses = np.convolve(losses, np.ones(10)/10, mode='valid')
        
        # Find interesting points
        min_idx = np.argmin(smoothed_losses[skip_start:-skip_end]) + skip_start
        max_idx = -skip_end
        
        # Plot
        plt.figure(figsize=(10,6))
        plt.plot(lrs[skip_start:max_idx], smoothed_losses[skip_start:max_idx])
        plt.xscale('log')
        plt.xlabel("Learning Rate (log scale)")
        plt.ylabel("Loss")
        plt.title("Learning Rate Range Test")
        
        min_lr = lrs[min_idx]
        max_lr = lrs[max_idx]
        
        plt.axvline(x=min_lr, color='red', linestyle='--', 
                   label=f'Min LR: {min_lr:.1e}')
        plt.axvline(x=max_lr, color='green', linestyle='--', 
                   label=f'Max LR: {max_lr:.1e}')
        plt.legend()
        plt.show()
        
        return min_lr, max_lr

# 2. Modify your main system file:

# At the top of dmao_system.py:

# python
# Copy
# from lr_finder import LRFinder
# import random  # Ensure this is imported for sample()
# Then add this method to your BareBonesDMAO_Learn class:

# python
# Copy
# def find_optimal_lr(self, train_data):
#     """Run learning rate finder routine"""
#     print("Running learning rate finder...")
    
#     # Store original state
#     original_state = {
#         'model': deepcopy(self.scaffold_model.state_dict()),
#         'optim': deepcopy(self.optimizer.state_dict()) if self.optimizer else None,
#         'proj': deepcopy(self.scaffold_proj.state_dict()) if hasattr(self, 'scaffold_proj') else None
#     }
    
#     # Temporary optimizer
#     temp_optim = AdamW(
#         list(self.scaffold_model.parameters()) + 
#         (list(self.scaffold_proj.parameters()) if hasattr(self, 'scaffold_proj') else []),
#         lr=1e-7  # Will be overridden
#     )
    
#     # Run finder
#     lrs, losses = LRFinder.find_lr(
#         model=self,
#         train_data=train_data,
#         optimizer=temp_optim,
#         device=DEVICE,
#         batch_size=BATCH_SIZE
#     )
    
#     # Plot and get suggestions
#     min_lr, max_lr = LRFinder.plot_lr_results(lrs, losses)
    
#     # Restore original state
#     self.scaffold_model.load_state_dict(original_state['model'])
#     if original_state['proj']:
#         self.scaffold_proj.load_state_dict(original_state['proj'])
#     if self.optimizer and original_state['optim']:
#         self.optimizer.load_state_dict(original_state['optim'])
        
#     print(f"\nSuggested LR range: {min_lr:.2e} - {max_lr:.2e}")
#     print(f"Recommended training LR: ~{max_lr/10:.2e} (1/10th of max)")
    
#     return max_lr/10  # Return suggested training LR
# 3. Update your training flow:

# In your main execution block:

# python
# Copy
# if __name__ == "__main__":
#     dmao_system = BareBonesDMAO_Learn()
    
#     # Add this before training:
#     if input("Run learning rate finder first? (y/n) ").lower() == 'y':
#         optimal_lr = dmao_system.find_optimal_lr(TRAIN_DATA)
#         # Update your training config
#         global LEARNING_RATE
#         LEARNING_RATE = optimal_lr
    
#     # Rest of your training code...
# Key Advantages of This Structure:
# Clean Separation of Concerns:

# LR finding logic is self-contained

# Main system file stays focused on core functionality

# Safer Execution:

# Proper state saving/restoration

# Error handling built-in

# Reusable Components:

# Can be used with any PyTorch model

# Easy to adapt for different projects

# Configurable:

# All key parameters are exposed

# Easy to adjust for different scenarios

# Visual Feedback:

# Immediate plot of results

# Clear LR recommendations

# This approach follows software engineering best practices while maintaining the flexibility needed for ML experimentation. The separate file can evolve independently as you refine your LR finding strategy.

# Would you like me to suggest any additional utility functions to include in the LR finder module?```

 
