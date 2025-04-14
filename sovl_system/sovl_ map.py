from collections import defaultdict
from typing import Dict, List, Union
import logging


class TokenMapping:
    """
    Handles the creation and management of token mappings between the base tokenizer and the scaffold tokenizer.
    """

    def __init__(self, base_tokenizer, scaffold_tokenizer):
        """
        Initialize the TokenMapping with the given tokenizers.

        Args:
            base_tokenizer: The tokenizer for the base model.
            scaffold_tokenizer: The tokenizer for the scaffold model.
        """
        self.base_tokenizer = base_tokenizer
        self.scaffold_tokenizer = scaffold_tokenizer

        # Token maps
        self.token_map = defaultdict(lambda: [self.scaffold_tokenizer.unk_token_id])
        self.special_token_map = {}

        # Logging setup
        self.logger = logging.getLogger("TokenMapping")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(handler)

        # Build the token map and special token map
        self._build_token_map()
        self._initialize_special_token_map()

        self.logger.info("TokenMapping initialized successfully.")

    def _build_token_map(self):
        """
        Create the token map between base tokenizer tokens and scaffold tokenizer tokens.
        """
        self.logger.info("Building token map...")
        for base_token, base_id in self.base_tokenizer.get_vocab().items():
            normalized = self._normalize_token(base_token)
            scaffold_ids = self.scaffold_tokenizer.encode(
                normalized, add_special_tokens=False, max_length=3, truncation=True
            ) or [self.scaffold_tokenizer.unk_token_id]
            self.token_map[base_id] = {"ids": scaffold_ids, "weight": 1.0}

        self.logger.info(f"Token map built with {len(self.token_map)} entries.")

    def _initialize_special_token_map(self):
        """
        Initialize the mapping for special tokens (e.g., PAD, EOS, UNK).
        """
        self.logger.info("Initializing special token map...")
        self.special_token_map = {
            self.base_tokenizer.pad_token_id: self._get_special_token_id("pad_token_id"),
            self.base_tokenizer.eos_token_id: self._get_special_token_id("eos_token_id", fallback="sep_token_id"),
            self.base_tokenizer.unk_token_id: self._get_special_token_id("unk_token_id"),
        }
        self.logger.info("Special token map initialized.")

    def _get_special_token_id(self, token_attr: str, fallback: str = None) -> int:
        """
        Retrieve the ID of a special token from the scaffold tokenizer.

        Args:
            token_attr (str): The attribute name of the special token (e.g., "pad_token_id").
            fallback (str): An optional fallback attribute if the primary attribute is not set.

        Returns:
            int: The ID of the special token or the unknown token ID if not found.
        """
        token_id = getattr(self.scaffold_tokenizer, token_attr, None)
        if token_id is None and fallback:
            token_id = getattr(self.scaffold_tokenizer, fallback, None)
        return token_id or self.scaffold_tokenizer.unk_token_id

    def _normalize_token(self, token: str) -> str:
        """
        Normalize a token for compatibility between tokenizers.

        Args:
            token (str): The token to normalize.

        Returns:
            str: The normalized token.
        """
        return token.replace("Ġ", "").replace("##", "")

    def get_token_map(self) -> Dict[int, Dict[str, Union[List[int], float]]]:
        """
        Retrieve the token map.

        Returns:
            Dict[int, Dict[str, Union[List[int], float]]]: The token map.
        """
        return self.token_map

    def get_special_token_map(self) -> Dict[int, int]:
        """
        Retrieve the special token map.

        Returns:
            Dict[int, int]: The special token map.
        """
        return self.special_token_map

    def log_token_mapping(self, sample_size: int = 10):
        """
        Log a sample of the token mapping for debugging.

        Args:
            sample_size (int): The number of token mappings to log.
        """
        self.logger.info("Logging a sample of the token map...")
        for i, (base_id, mapping) in enumerate(self.token_map.items()):
            if i >= sample_size:
                break
            self.logger.info(f"Base Token ID {base_id}: Scaffold IDs {mapping['ids']} (Weight: {mapping['weight']})")

        self.logger.info("Logging a sample of the special token map...")
        for base_id, scaffold_id in self.special_token_map.items():
            self.logger.info(f"Special Token ID {base_id}: Mapped to Scaffold Token ID {scaffold_id}")


def build_token_mapping(base_tokenizer, scaffold_tokenizer) -> TokenMapping:
    """
    Utility function to create a TokenMapping instance.

    Args:
        base_tokenizer: The tokenizer for the base model.
        scaffold_tokenizer: The tokenizer for the scaffold model.

    Returns:
        TokenMapping: An instance of the TokenMapping class.
    """
    return TokenMapping(base_tokenizer, scaffold_tokenizer)
