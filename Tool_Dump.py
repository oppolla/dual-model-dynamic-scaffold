@torch.no_grad()
def evaluate_generation_quality(self, num_samples=3):
    num_samples = min(num_samples, len(VALID_DATA))
    if num_samples <= 0:
        print("No validation data for evaluation!")
        return
    samples = random.sample(VALID_DATA, num_samples)
    print("\n=== Generation Evaluation ===")
    for example in samples:
        print(f"\nPrompt: {example['prompt']}")
        print(f"Expected: {example['completion']}")
        for weight in [0.0, 0.5, 1.0]:
            response = self.generate(example['prompt'], scaffold_weight=weight,
                                     max_new_tokens=60, temperature=0.7)
            print(f"w={weight}: {response}")

def run_k_fold_cross_validation(train_data, k_folds=5):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_data)):
        print(f"Training fold {fold+1}/{k_folds}...")
        fold_train_data = [train_data[i] for i in train_idx]
        fold_valid_data = [train_data[i] for i in valid_idx]

        dmao_system = BareBonesDMAO_Learn()
        dmao_system.run_training_cycle(fold_train_data, fold_valid_data, epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE)
        valid_loss = dmao_system.validate_epoch(fold_valid_data)
        fold_results.append(valid_loss)
        print(f"Fold {fold+1} validation loss: {valid_loss}")
        dmao_system.cleanup()

    avg_validation_loss = sum(fold_results) / len(fold_results)
    print(f"Average validation loss across {k_folds} folds: {avg_validation_loss}")
    return avg_validation_loss
