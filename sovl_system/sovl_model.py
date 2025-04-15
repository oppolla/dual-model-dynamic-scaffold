def sync_token_map(self, state: SOVLState) -> None:
    """Synchronize token map with state."""
    with state.lock:
        try:
            # Get current token map from state
            token_map = state.get_token_map()
            
            # Validate token map structure
            if not isinstance(token_map, dict):
                raise ValueError("Token map must be a dictionary")
                
            for token_id, mapping in token_map.items():
                if not isinstance(mapping, dict):
                    raise ValueError(f"Invalid mapping for token {token_id}")
                if 'ids' not in mapping or 'weight' not in mapping:
                    raise ValueError(f"Missing required fields in mapping for token {token_id}")
                if not isinstance(mapping['ids'], list):
                    raise ValueError(f"Invalid ids type for token {token_id}")
                if not isinstance(mapping['weight'], (int, float)):
                    raise ValueError(f"Invalid weight type for token {token_id}")
            
            # Update scaffold token mapper
            self.scaffold_token_mapper.token_map = token_map
            
            # Log synchronization
            self._log_event(
                "token_map_synced",
                {
                    "token_map_size": len(token_map),
                    "state_hash": state.state_hash()
                }
            )
            
        except Exception as e:
            self._log_error(
                f"Failed to sync token map: {str(e)}",
                error_type="token_map_error",
                stack_trace=traceback.format_exc(),
                context={
                    "state_hash": state.state_hash()
                }
            )
            raise 