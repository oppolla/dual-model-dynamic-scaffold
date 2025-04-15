from typing import Dict, Any, Optional
from sovl_grafter import PluginInterface, PluginMetadata

class ExamplePlugin(PluginInterface):
    """Example plugin demonstrating plugin system capabilities."""
    
    def __init__(self):
        self.system = None
        self.state_version = "1.0"
    
    def initialize(self, system: 'SOVLSystem') -> None:
        """Initialize the plugin with access to the SOVLSystem."""
        self.system = system
        self.system.plugin_manager.register_hook(
            "pre_generate",
            self.pre_generate_hook,
            plugin_name="example_plugin",
            priority=10
        )
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="example_plugin",
            version="1.0.0",
            description="Example plugin for SOVLSystem",
            author="xAI",
            dependencies=[],
            priority=10,
            config_requirements=[
                ConfigSchema(
                    field="plugin_config.example_plugin.enabled",
                    type=bool,
                    default=True,
                    required=True
                ),
                ConfigSchema(
                    field="plugin_config.example_plugin.mode",
                    type=str,
                    default="default",
                    validator=lambda x: x in ["default", "advanced"],
                    required=True
                )
            ]
        )
    
    def execute(self, context: Dict[str, Any], *args, **kwargs) -> Any:
        """Execute the plugin's main functionality."""
        with NumericalGuard():
            return {"status": "executed", "context": context}
    
    def pre_generate_hook(self, context: Dict[str, Any]) -> None:
        """Hook executed before generating a response."""
        self.system.logger.record({
            "event": "example_plugin_pre_generate",
            "context": str(context)[:200],
            "timestamp": time.time(),
            "conversation_id": self.system.state.history.conversation_id
        })
    
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        self.system = None 