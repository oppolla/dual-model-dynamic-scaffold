import torch
import threading
import time
import psutil
import numpy as np
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch import nn
import copy
from transformers import BartForConditionalGeneration, BartTokenizer

# --- Configuration ---
BASE_MODEL_NAME = "deepseek-ai/deepseek-llm-67b-chat"
SCAFFOLD_MODEL_NAME = "deepseek-ai/deepseek-r1-distill-qwen1.5-1.5b"
LORA_RANK = 8
CROSS_ATTN_LAYERS = [8, 16, 24]

DMAO_CONFIG = {
    'check_interval': 300,        # 5 minutes
    'min_examples': 50,           # Minimum examples for training
    'max_training_time': 1800,    # 30 minutes
    'system_load_limit': 0.7,     # 70% max utilization
    'salience_threshold': 0.75,   # Importance threshold
    'time_window': 86400,         # 24-hour update window
    'train_epochs': 3,            # Training epochs
    'learning_rate': 1e-5,        # Learning rate
    'max_rank': 16,               # Max LoRA rank
    'sparse_topk': 10,            # Top-k attention sparsity
    'min_rank': 2,                # Min LoRA rank
    'confidence_threshold': 0.65,  # Confidence for scaffold use
    'default_max_output_length': 512,  # Default maximum output length in tokens
    'min_output_length': 10,           # Minimum output length in tokens
}

# --- Optimized Modules ---
class AdaptiveLoRALinear(nn.Module):
    """Dynamic rank LoRA with importance-based adaptation"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.importance = nn.Parameter(torch.rand(1))
        self.rank = DMAO_CONFIG['min_rank']
        self.lora_A = nn.Parameter(torch.zeros(self.rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.rank))
        
    def forward(self, x):
        effective_rank = DMAO_CONFIG['min_rank'] + int(
            (DMAO_CONFIG['max_rank'] - DMAO_CONFIG['min_rank']) * self.importance
        )
        return self.base(x) + (x @ self.lora_A[:effective_rank].T) @ self.lora_B[:, :effective_rank].T

class SparseCrossAttention(nn.Module):
    """Top-k optimized attention with learned threshold and sparsity control"""
    def __init__(self, base_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(base_dim, num_heads, batch_first=True)
        self.topk_learned = nn.Linear(base_dim, 1)
        self.sparsity_factor = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))  # Initialized to mid-range

    def forward(self, base_hidden, scaffold_hidden):
        scores = self.topk_learned(base_hidden).squeeze(-1)
        seq_len = base_hidden.size(1)
        effective_topk = max(1, min(seq_len, int(seq_len * torch.sigmoid(self.sparsity_factor))))
        topk_idx = torch.topk(scores, effective_topk, dim=1).indices
        sparse_hidden = base_hidden.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, base_hidden.size(-1)))
        attn_out, _ = self.attention(sparse_hidden, scaffold_hidden, scaffold_hidden)
        return base_hidden.scatter(1, topk_idx.unsqueeze(-1).expand(-1, -1, attn_out.size(-1)), attn_out)

class CrossAttentionFuser(nn.Module):
    """Fuses base and scaffold hidden states with sparse cross-attention"""
    def __init__(self, base_dim, scaffold_dim, num_heads=8):
        super().__init__()
        self.attention = SparseCrossAttention(base_dim, num_heads)
        self.gate = nn.Sequential(
            nn.Linear(base_dim, 1),
            nn.Sigmoid()
        )
        self.scaffold_proj = nn.Linear(scaffold_dim, base_dim)
        self.confidence_threshold = nn.Parameter(torch.tensor([DMAO_CONFIG['confidence_threshold']], dtype=torch.float32))

    def forward(self, base_hidden, scaffold_hidden, blend=0.5): #Base Model / Scaffold Model Blend control
        scaffold_hidden = self.scaffold_proj(scaffold_hidden)
        if self._get_confidence(scaffold_hidden) > self.confidence_threshold:
            attn_output = self.attention(base_hidden, scaffold_hidden)
            gate_weight = self.gate(base_hidden)
            augmented_output = base_hidden + gate_weight * attn_output
            return (1 - blend) * base_hidden + blend * augmented_output
        return base_hidden

    def _get_confidence(self, hidden_states):
        return torch.mean(torch.norm(hidden_states, dim=-1)).item()
    
class SalienceScorer(nn.Module):
    """Scores interaction importance using BERT"""
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, text):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            features = self.encoder(**inputs).last_hidden_state.mean(1)
        return self.classifier(features)

class DMAOSystem:
    """Dual-Model Adaptive Orchestrator System"""
    def __init__(self):
        # Base model (frozen)
        self.base_model = AutoModel.from_pretrained(BASE_MODEL_NAME).eval()
        self.base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        
        # Scaffold model with dynamic LoRA
        scaffold_config = AutoConfig.from_pretrained(SCAFFOLD_MODEL_NAME)
        original = AutoModel.from_pretrained(SCAFFOLD_MODEL_NAME, config=scaffold_config)
        self._replace_lora_layers(original)
        self.scaffold_model = get_peft_model(
            original,
            LoraConfig(
                r=LORA_RANK,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none"
            )
        )
        self.scaffold_tokenizer = AutoTokenizer.from_pretrained(SCAFFOLD_MODEL_NAME)
        
        # Store initial scaffold state for reset
        self.initial_scaffold_state = copy.deepcopy(self.scaffold_model.state_dict())
        
        # Production scaffold and cross-attention
        self.production_scaffold = copy.deepcopy(self.scaffold_model)
        self._insert_cross_attention()
        
        # DMAO infrastructure
        self.interaction_buffer = []
        self.salience_scorer = SalienceScorer().eval()
        self.training_lock = threading.Lock()
        self.last_trained = 0
        
        # Data augmenter for synthetic examples
        self.augmenter_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
        self.augmenter_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        self._start_scheduler()

    def _replace_lora_layers(self, model):
        """Replace linear layers with AdaptiveLoRALinear"""
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                setattr(model, name, AdaptiveLoRALinear(module.in_features, module.out_features))
            else:
                self._replace_lora_layers(module)

    def _insert_cross_attention(self):
        """Inject cross-attention into base model"""
        base_config = self.base_model.config
        scaffold_dim = self.scaffold_model.config.hidden_size
        
        for layer_idx in CROSS_ATTN_LAYERS:
            original_layer = self.base_model.layers[layer_idx]
            cross_attn = CrossAttentionFuser(
                base_dim=base_config.hidden_size,
                scaffold_dim=scaffold_dim,
                num_heads=base_config.num_attention_heads
            )
            
            class ModifiedLayer(nn.Module):
                def __init__(self, orig_layer, cross_attn):
                    super().__init__()
                    self.orig_layer = orig_layer
                    self.cross_attn = cross_attn
                    
                def forward(self, x, scaffold_context=None, **kwargs):
                    x = self.orig_layer(x, **kwargs)
                    if scaffold_context is not None:
                        x = self.cross_attn(x, scaffold_context)
                    return x
            
            self.base_model.layers[layer_idx] = ModifiedLayer(original_layer, cross_attn)

    def _start_scheduler(self):
        """Background training scheduler"""
        def scheduler_loop():
            while True:
                time.sleep(DMAO_CONFIG['check_interval'])
                if self._should_trigger_training():
                    self._run_dmao_training()
        
        self.scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self.scheduler_thread.start()

    def _should_trigger_training(self):
        """Check if training should start"""
        cpu_load = psutil.cpu_percent() / 100
        gpu_load = torch.cuda.utilization() / 100 if torch.cuda.is_available() else 0
        system_ok = cpu_load < DMAO_CONFIG['system_load_limit'] and gpu_load < DMAO_CONFIG['system_load_limit']
        
        data_ok = len(self.interaction_buffer) >= DMAO_CONFIG['min_examples']
        recent_ok = (time.time() - self.last_trained) > DMAO_CONFIG['time_window']
        
        if data_ok:
            avg_salience = sum(e['salience'] for e in self.interaction_buffer) / len(self.interaction_buffer)
            return avg_salience >= DMAO_CONFIG['salience_threshold'] and system_ok and recent_ok
        return False

    def _run_dmao_training(self):
        """Execute training with safety checks"""
        with self.training_lock, torch.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            print("\n--- Starting DMAO Training ---")
            try:
                train_data = self._prepare_training_data()
                optimizer = torch.optim.AdamW(self.scaffold_model.parameters(), lr=DMAO_CONFIG['learning_rate'])
                start_time = time.time()
                
                for epoch in range(DMAO_CONFIG['train_epochs']):
                    if time.time() - start_time > DMAO_CONFIG['max_training_time']:
                        raise TimeoutError("Training exceeded time limit")
                    self._train_epoch(epoch, train_data, optimizer)
                
                self._deploy_updated_scaffold()
                print("--- DMAO Training Completed ---")
            except Exception as e:
                print(f"Training aborted: {e}")
                self._rollback_training()

    def _prepare_training_data(self):
        """Cluster-based data preparation with synthetic augmentation"""
        embeddings = []
        for interaction in self.interaction_buffer:
            inputs = self.scaffold_tokenizer(interaction['input'], return_tensors='pt', truncation=True, padding=True)
            with torch.no_grad():
                emb = self.scaffold_model(**inputs).last_hidden_state.mean(1).cpu().numpy()
            embeddings.append(emb)
        
        kmeans = KMeans(n_clusters=5).fit(np.concatenate(embeddings))
        sample_weights = 1 / (np.bincount(kmeans.labels_, minlength=5) + 1e-6)
        selected_idx = np.random.choice(
            len(self.interaction_buffer),
            size=min(DMAO_CONFIG['min_examples'], len(self.interaction_buffer)),
            p=sample_weights / sample_weights.sum()
        )
        
        # Real data batch
        batch = []
        for idx in selected_idx:
            interaction = self.interaction_buffer[idx]
            inputs = self.scaffold_tokenizer(interaction['input'], return_tensors='pt', padding=True, truncation=True)
            labels = self.scaffold_tokenizer(interaction['output'], return_tensors='pt', padding=True, truncation=True)['input_ids']
            batch.append((inputs, labels))
        
        # Synthetic data augmentation (2x real data)
        synthetic_batch = []
        num_synthetic = min(DMAO_CONFIG['min_examples'], len(selected_idx) * 2)
        for idx in selected_idx[:num_synthetic]:
            interaction = self.interaction_buffer[idx]
            # Generate paraphrased input
            aug_input = self._augment_text(interaction['input'])
            # Generate paraphrased output
            aug_output = self._augment_text(interaction['output'])
            aug_inputs = self.scaffold_tokenizer(aug_input, return_tensors='pt', padding=True, truncation=True)
            aug_labels = self.scaffold_tokenizer(aug_output, return_tensors='pt', padding=True, truncation=True)['input_ids']
            synthetic_batch.append((aug_inputs, aug_labels))
        
        # Combine real and synthetic data
        batch.extend(synthetic_batch)
        self.interaction_buffer = []
        return batch
    
    def _augment_text(self, text):
        """Generate paraphrased text using BART"""
        with torch.no_grad():
            inputs = self.augmenter_tokenizer(f"paraphrase: {text}", return_tensors='pt', truncation=True, max_length=128)
            outputs = self.augmenter_model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
            paraphrased = self.augmenter_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return paraphrased

    def _train_epoch(self, epoch, data, optimizer):
        """Train one epoch with checkpointing and L2 regularization"""
        self.scaffold_model.train()
        total_loss = 0
        
        for inputs, labels in data:
            def _checkpoint_forward(scaffold_in, base_in, lbls):
                scaffold_out = self.scaffold_model(**scaffold_in).last_hidden_state
                base_out = self.base_model(**base_in, scaffold_context=scaffold_out)
                loss = F.cross_entropy(base_out.logits.view(-1, base_out.logits.size(-1)), lbls.view(-1))
                # Add L2 regularization
                l2_lambda = 0.01  # Regularization strength
                l2_norm = sum(p.pow(2.0).sum() for p in self.scaffold_model.parameters())
                return loss + l2_lambda * l2_norm
            
            base_inputs = self.base_tokenizer(inputs['input_ids'], return_tensors='pt', padding=True)
            loss = checkpoint(_checkpoint_forward, inputs, base_inputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.scaffold_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(data):.4f}")

    def _deploy_updated_scaffold(self):
        """Deploy trained scaffold model"""
        shadow_model = copy.deepcopy(self.scaffold_model)
        shadow_model.load_state_dict(self.scaffold_model.state_dict())
        with torch.no_grad():
            for prod_param, shadow_param in zip(self.production_scaffold.parameters(), shadow_model.parameters()):
                prod_param.copy_(shadow_param)
        del shadow_model
        self.last_trained = time.time()

    def generate_response(self, user_input, max_output_length=None):
        """Generate response using base and scaffold models with length control"""
        # Set default max length if not specified
        effective_max_length = (
            max_output_length if max_output_length is not None 
            else DMAO_CONFIG['default_max_output_length']
        )
        # Ensure max length is within reasonable bounds
        effective_max_length = max(
            DMAO_CONFIG['min_output_length'], 
            min(effective_max_length, 2048)  # Upper limit of 2048 tokens
        )

        scaffold_inputs = self.scaffold_tokenizer(user_input, return_tensors='pt', padding=True, truncation=True)
        scaffold_out = self.production_scaffold(**scaffold_inputs).last_hidden_state
        
        base_inputs = self.base_tokenizer(user_input, return_tensors='pt')
        outputs = self.base_model.generate(
            **base_inputs,
            scaffold_context=scaffold_out,
            max_length=effective_max_length,
            min_length=DMAO_CONFIG['min_output_length']
        )
        return self.base_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def log_interaction(self, user_input, model_output):
        """Log interaction with salience score"""
        text = user_input + " [SEP] " + model_output
        with torch.no_grad():
            salience = self.salience_scorer(text).item()
        self.interaction_buffer.append({
            'input': user_input,
            'output': model_output,
            'salience': salience,
            'timestamp': time.time()
        })

    def _rollback_training(self):
        """Revert to last good scaffold state"""
        self.scaffold_model.load_state_dict(self.production_scaffold.state_dict())

    def reset_scaffold(self):
        """Reset scaffold model to its initial pre-trained state"""
        with self.training_lock:
            # Reset both training and production scaffold models
            self.scaffold_model.load_state_dict(self.initial_scaffold_state)
            self.production_scaffold.load_state_dict(self.initial_scaffold_state)
            # Clear interaction buffer to prevent re-learning old patterns
            self.interaction_buffer = []
            self.last_trained = time.time()
            print("--- Scaffold model reset to initial state ---")
# --- Usage ---
if __name__ == "__main__":
    system = DMAOSystem()
    try:
        while True:
            user_input = input("User: ")
            if user_input.strip().upper() == "RESET_SCAFFOLD":
                system.reset_scaffold()
                print("Assistant: Scaffold reset complete. Ready for new input.")
                continue
            
            # Example with specific length
            response_short = system.generate_response(user_input, max_output_length=50)
            print(f"Assistant (50 tokens): {response_short}")
            
            # Example with default length
            response_default = system.generate_response(user_input)
            print(f"Assistant (default): {response_default}")
            
            system.log_interaction(user_input, response_default)
    except KeyboardInterrupt:
        print("\n--- System Shutdown ---")
