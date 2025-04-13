sovl_manager
class SacredEmberSystem:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = Logger(config_manager)
        self.state = SOVLState(config_manager)
        self.token_mapper = TokenMapper(
            base_tokenizer=None,  # Set after ModelManager
            scaffold_tokenizer=None,
            logger=self.logger,
            scaffold_unk_id=config_manager.get("scaffold_unk_id", 0)
        )
        self.model_manager = ModelManager(
            core_config=config_manager.get("core_config", {}),
            lora_config=config_manager.get("lora_config", {}),
            cross_attn_config=config_manager.get("cross_attention_config", {}),
            controls_config=config_manager.get("controls_config", {}),
            token_mapper=self.token_mapper,
            logger=self.logger,
            device=DEVICE
        )
        # Update token_mapper with tokenizers
        self.token_mapper.base_tokenizer = self.model_manager.get_base_tokenizer()
        self.token_mapper.scaffold_tokenizer = self.model_manager.get_scaffold_tokenizer()
        self.curiosity_system = CuriositySystem(
            config=config_manager.get("curiosity_config", {}),
            state=self.state.curiosity,
            model_manager=self.model_manager,
            logger=self.logger,
            device=DEVICE
        )

    def generate(self, prompt):
        model = self.model_manager.get_sacred_ember()
        scaffold = self.model_manager.get_scaffold()
        tokens = self.token_mapper.map_for_sacred_ember(prompt)
        # Generation logic using model, scaffold, tokens
        return output

if __name__ == "__main__":
    config_manager = ConfigManager("config.json")
    system = SacredEmberSystem(config_manager)
    run_cli(system=system, config_manager=config_manager)
