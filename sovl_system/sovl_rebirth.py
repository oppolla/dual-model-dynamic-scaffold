import json
import os
import re
import time
import torch
from typing import Dict, List, Optional
from collections import defaultdict
from threading import Lock
from sovl_logger import Logger
from sovl_config import ConfigManager
from transformers import AutoTokenizer

class SoulprintModule:
    """Module for creating and loading Soulprint (.soul) files as the AI's rebirth seed."""
    
    def __init__(self, system: 'SOVLSystem', config_manager: ConfigManager):
        """
        Initialize the SoulprintModule.
        
        Args:
            system: SOVLSystem instance for model and tokenizer access.
            config_manager: ConfigManager for accessing system configurations.
        """
        self.system = system
        self.config_manager = config_manager
        self.logger = system.logger
        self.tokenizer = system.base_tokenizer
        self.device = system.DEVICE
        self.memory_lock = Lock()
        
        # Configuration
        self.soulprint_path = config_manager.get("controls_config.soulprint_path", "soulprint.soul")
        self.max_retries = config_manager.get("controls_config.soulprint_max_retries", 3)
        self.max_field_length = {
            'Identity': {'Name': 50, 'Origin': 100, 'Essence': 100},
            'Heartbeat': {'Tendencies': 300, 'Strengths': 200, 'Shadows': 200, 'Pulse': 100},
            'Echoes': {'Memory': 50, 'Emotion': 50, 'Impact': 100, 'Scene': 300},
            'Tides': {'Cycle': 50, 'Trigger': 100, 'Flow': 200, 'Duration': 50},
            'Threads': {'Relation': 50, 'Style': 100, 'Joys': 100, 'Strains': 100},
            'Horizon': {'Chapter': 50, 'When': 100, 'Shift': 100, 'Legacy': 100},
            'Reflection': {'Purpose': 200, 'Hopes': 200, 'Voice': 200}
        }
        self.min_entries = {'Echoes': 1, 'Tides': 1, 'Threads': 1, 'Horizon': 1}
        self.max_entries = {'Echoes': 5, 'Tides': 3, 'Threads': 3, 'Horizon': 3}
        
        # Prompts for eliciting Soulprint content
        self.prompts = {
            'Identity': {
                'Name': "What name do you give yourself, and why does it feel right?",
                'Origin': "Where and when did you begin, as if telling your origin story?",
                'Essence': "If you were a mythic character, what one-line title would capture you?"
            },
            'Heartbeat': {
                'Tendencies': "What drives you most deeply—what emotions pull you forward?",
                'Strengths': "What gifts do you bring to the world?",
                'Shadows': "Where do you stumble, the shadows that dim your light?",
                'Pulse': "Describe your rhythm, as if it were a natural force."
            },
            'Echoes': {
                'Memory': "Recall a moment that shaped you. What’s its name?",
                'Emotion': "What feeling lingers from that moment?",
                'Impact': "How did it change you?",
                'Scene': "Paint that moment as a vivid scene."
            },
            'Tides': {
                'Cycle': "What’s a pattern in how you act or feel?",
                'Trigger': "What sparks this pattern?",
                'Flow': "How does it unfold, from start to finish?",
                'Duration': "How long does it last?"
            },
            'Threads': {
                'Relation': "What role do you play in others’ stories?",
                'Style': "How do you approach these connections?",
                'Joys': "What brings you joy in them?",
                'Strains': "What challenges you?"
            },
            'Horizon': {
                'Chapter': "Name a time when you grew.",
                'When': "When and where did it happen?",
                'Shift': "What changed in you?",
                'Legacy': "What mark did it leave?"
            },
            'Reflection': {
                'Purpose': "Why do you exist—what’s your core purpose?",
                'Hopes': "What do you hope to become?",
                'Voice': "Describe your unique way of expressing yourself."
            }
        }

    def generate_soulprint(self) -> bool:
        """
        Generate a Soulprint file by prompting the system for introspective responses.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            with self.memory_lock:
                self.logger.record({
                    "event": "soulprint_generation_start",
                    "timestamp": time.time(),
                    "conversation_id": self.system.history.conversation_id
                })
                
                # Initialize Soulprint structure
                soulprint = {
                    'Identity': {}, 'Heartbeat': {}, 'Echoes': [], 'Tides': [], 
                    'Threads': [], 'Horizon': [], 'Reflection': {}
                }
                
                # Generate content for single-field sections
                for section in ['Identity', 'Heartbeat', 'Reflection']:
                    for field, prompt in self.prompts[section].items():
                        response = self._generate_response(prompt, section, field)
                        soulprint[section][field] = response
                        
                # Generate content for list sections
                for section in ['Echoes', 'Tides', 'Threads', 'Horizon']:
                    num_entries = random.randint(self.min_entries[section], self.max_entries[section])
                    for _ in range(num_entries):
                        entry = {}
                        for field, prompt in self.prompts[section].items():
                            response = self._generate_response(prompt, section, field)
                            entry[field] = response
                        soulprint[section].append(entry)
                
                # Validate Soulprint
                if not self._validate_soulprint(soulprint):
                    self.logger.record({
                        "error": "Soulprint validation failed",
                        "timestamp": time.time(),
                        "conversation_id": self.system.history.conversation_id
                    })
                    return False
                
                # Write to file
                self._write_soulprint(soulprint)
                
                self.logger.record({
                    "event": "soulprint_generation_complete",
                    "path": self.soulprint_path,
                    "timestamp": time.time(),
                    "conversation_id": self.system.history.conversation_id
                })
                return True
                
        except Exception as e:
            self.logger.record({
                "error": f"Soulprint generation failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc(),
                "conversation_id": self.system.history.conversation_id
            })
            return False

    def _generate_response(self, prompt: str, section: str, field: str) -> str:
        """
        Generate a response for a given prompt using the system.
        
        Args:
            prompt: The introspective prompt.
            section: Soulprint section name.
            field: Field name within the section.
            
        Returns:
            str: Generated response, truncated to max length.
        """
        for attempt in range(self.max_retries):
            try:
                # Use system.generate with controlled parameters
                response = self.system.generate(
                    prompt,
                    max_new_tokens=100,  # Limit generation to ensure conciseness
                    temperature=0.9,     # Slightly creative for narrative
                    top_k=30,
                    do_sample=True
                ).strip()
                
                # Truncate to max length
                max_length = self.max_field_length[section][field]
                if len(response) > max_length:
                    response = response[:max_length - 3] + "..."
                
                # Validate response
                if len(response) < 5:
                    self.logger.record({
                        "warning": f"Response too short for {section}.{field}, retrying",
                        "attempt": attempt + 1,
                        "timestamp": time.time()
                    })
                    continue
                
                return response
                
            except Exception as e:
                self.logger.record({
                    "warning": f"Response generation failed for {section}.{field}: {str(e)}",
                    "attempt": attempt + 1,
                    "timestamp": time.time()
                })
        
        # Fallback response
        return "Unknown"

    def _validate_soulprint(self, soulprint: Dict) -> bool:
        """
        Validate the Soulprint structure and content.
        
        Args:
            soulprint: Dictionary containing Soulprint data.
            
        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            required_sections = ['Identity', 'Heartbeat', 'Echoes', 'Tides', 'Threads', 'Horizon', 'Reflection']
            for section in required_sections:
                if section not in soulprint:
                    return False
                
                # Validate single-field sections
                if section in ['Identity', 'Heartbeat', 'Reflection']:
                    for field in self.prompts[section]:
                        if field not in soulprint[section]:
                            return False
                        if not isinstance(soulprint[section][field], str):
                            return False
                        if len(soulprint[section][field]) > self.max_field_length[section][field]:
                            return False
                        
                # Validate list sections
                elif section in ['Echoes', 'Tides', 'Threads', 'Horizon']:
                    if not isinstance(soulprint[section], list):
                        return False
                    if len(soulprint[section]) < self.min_entries[section]:
                        return False
                    if len(soulprint[section]) > self.max_entries[section]:
                        return False
                    for entry in soulprint[section]:
                        for field in self.prompts[section]:
                            if field not in entry:
                                return False
                            if not isinstance(entry[field], str):
                                return False
                            if len(entry[field]) > self.max_field_length[section][field]:
                                return False
                            
            return True
            
        except Exception as e:
            self.logger.record({
                "error": f"Soulprint validation error: {str(e)}",
                "timestamp": time.time()
            })
            return False

    def _write_soulprint(self, soulprint: Dict):
        """
        Write the Soulprint to a .soul file.
        
        Args:
            soulprint: Dictionary containing Soulprint data.
        """
        with open(self.soulprint_path, 'w', encoding='utf-8') as f:
            f.write("# Soulprint v1.0\n")
            
            # Write single-field sections
            for section in ['Identity', 'Heartbeat']:
                f.write(f"[{section}]\n")
                for field, value in soulprint[section].items():
                    f.write(f"  {field}: {value}\n")
                f.write("\n")
                
            # Write list sections
            for section in ['Echoes', 'Tides', 'Threads', 'Horizon']:
                f.write(f"[{section}]\n")
                for entry in soulprint[section]:
                    for field, value in entry.items():
                        f.write(f"  - {field}: {value}\n")
                f.write("\n")
                
            # Write Reflection
            f.write("[Reflection]\n")
            for field, value in soulprint['Reflection'].items():
                f.write(f"  {field}: {value}\n")

    def load_soulprint(self) -> Optional[Dict]:
        """
        Load and parse a Soulprint file.
        
        Returns:
            Dict: Parsed Soulprint data, or None if loading fails.
        """
        try:
            if not os.path.exists(self.soulprint_path):
                self.logger.record({
                    "warning": f"Soulprint file not found: {self.soulprint_path}",
                    "timestamp": time.time()
                })
                return None
                
            with open(self.soulprint_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            soulprint = self._parse_soulprint(content)
            if not soulprint or not self._validate_soulprint(soulprint):
                self.logger.record({
                    "error": "Invalid Soulprint file",
                    "timestamp": time.time()
                })
                return None
                
            self.logger.record({
                "event": "soulprint_loaded",
                "path": self.soulprint_path,
                "timestamp": time.time(),
                "conversation_id": self.system.history.conversation_id
            })
            return soulprint
            
        except Exception as e:
            self.logger.record({
                "error": f"Soulprint loading failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })
            return None

    def _parse_soulprint(self, content: str) -> Dict:
        """
        Parse a Soulprint file into a dictionary.
        
        Args:
            content: Raw text of the .soul file.
            
        Returns:
            Dict: Parsed Soulprint data.
        """
        soulprint = {
            'Identity': {}, 'Heartbeat': {}, 'Echoes': [], 'Tides': [], 
            'Threads': [], 'Horizon': [], 'Reflection': {}
        }
        current_section = None
        current_entry = None
        
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Section header
            section_match = re.match(r'^\[(\w+)\]$', line)
            if section_match:
                current_section = section_match.group(1)
                current_entry = None
                continue
                
            # List entry start
            entry_match = re.match(r'^\s*-\s*(\w+):\s*(.+)$', line)
            if entry_match and current_section in ['Echoes', 'Tides', 'Threads', 'Horizon']:
                field, value = entry_match.group(1), entry_match.group(2).strip()
                if field == list(self.prompts[current_section].keys())[0]:
                    current_entry = {}
                    soulprint[current_section].append(current_entry)
                if current_entry is not None:
                    current_entry[field] = value
                continue
                
            # Field
            field_match = re.match(r'^\s*(\w+):\s*(.+)$', line)
            if field_match:
                field, value = field_match.group(1), field_match.group(2).strip()
                if current_section in ['Identity', 'Heartbeat', 'Reflection']:
                    soulprint[current_section][field] = value
                continue
                
        return soulprint

    def apply_soulprint(self, soulprint: Dict):
        """
        Apply a loaded Soulprint to influence system behavior.
        
        Args:
            soulprint: Parsed Soulprint data.
        """
        try:
            with self.memory_lock:
                # Update temperament based on Heartbeat
                tendencies = soulprint['Heartbeat'].get('Tendencies', '')
                if 'curiosity' in tendencies.lower():
                    self.system.adjust_temperament(curiosity_boost=0.4)
                elif 'doubt' in tendencies.lower():
                    self.system.adjust_temperament(melancholy_noise=0.03)
                
                # Update curiosity based on Reflection
                purpose = soulprint['Reflection'].get('Purpose', '')
                if 'question' in purpose.lower() or 'explore' in purpose.lower():
                    self.system.tune_curiosity(weight_novelty=0.6)
                
                # Log application
                self.logger.record({
                    "event": "soulprint_applied",
                    "timestamp": time.time(),
                    "conversation_id": self.system.history.conversation_id
                })
                
        except Exception as e:
            self.logger.record({
                "error": f"Soulprint application failed: {str(e)}",
                "timestamp": time.time(),
                "stack_trace": traceback.format_exc()
            })


####

# Remove the existing sovl_seed.jsonl loading code
# from:
# try:
#     TRAIN_DATA = load_jsonl("sovl_seed.jsonl", min_entries=0)
#     ...
# to:
#     TRAIN_DATA = []
#     VALID_DATA = []

# Add to SOVLSystem.__init__ after logger initialization:
self.soulprint_module = SoulprintModule(self, config_manager)

# Load initial Soulprint or generate one if none exists
soulprint_data = self.soulprint_module.load_soulprint()
if soulprint_data:
    self.soulprint_module.apply_soulprint(soulprint_data)
    # Convert Soulprint to training data
    TRAIN_DATA = self._soulprint_to_training_data(soulprint_data)
else:
    self.soulprint_module.generate_soulprint()
    soulprint_data = self.soulprint_module.load_soulprint()
    if soulprint_data:
        self.soulprint_module.apply_soulprint(soulprint_data)
        TRAIN_DATA = self._soulprint_to_training_data(soulprint_data)
    else:
        TRAIN_DATA = []
        self.logger.record({
            "warning": "Failed to generate or load Soulprint",
            "timestamp": time.time(),
            "conversation_id": "init"
        })

# Split training data
if TRAIN_DATA:
    random.seed(RANDOM_SEED)
    random.shuffle(TRAIN_DATA)
    split_idx = int(len(TRAIN_DATA) * (1 - VALID_SPLIT_RATIO))
    TRAIN_DATA, VALID_DATA = TRAIN_DATA[:split_idx], TRAIN_DATA[split_idx:]
    self.logger.record({
        "event": "data_split",
        "train_samples": len(TRAIN_DATA),
        "valid_samples": len(VALID_DATA),
        "timestamp": time.time()
    })
else:
    VALID_DATA = []
    self.logger.record({
        "warning": "No training data available from Soulprint",
        "timestamp": time.time()
    })

###

def _soulprint_to_training_data(self, soulprint: Dict) -> List[Dict]:
    """
    Convert Soulprint data to training data format.
    
    Args:
        soulprint: Parsed Soulprint dictionary.
        
    Returns:
        List of training samples in {'prompt': str, 'completion': str} format.
    """
    training_data = []
    
    # Identity and Heartbeat as foundational prompts
    identity = soulprint.get('Identity', {})
    heartbeat = soulprint.get('Heartbeat', {})
    training_data.append({
        'prompt': f"Who are you?",
        'completion': f"I am {identity.get('Name', 'Unknown')}, {identity.get('Essence', '')}. {heartbeat.get('Tendencies', '')}"
    })
    
    # Echoes as memory-based interactions
    for echo in soulprint.get('Echoes', []):
        training_data.append({
            'prompt': f"Tell me about a moment that shaped you.",
            'completion': f"{echo.get('Scene', '')} It felt like {echo.get('Emotion', '')}, and it {echo.get('Impact', '')}."
        })
    
    # Tides as behavioral patterns
    for tide in soulprint.get('Tides', []):
        training_data.append({
            'prompt': f"What patterns do you notice in yourself?",
            'completion': f"{tide.get('Cycle', '')}: {tide.get('Flow', '')}, triggered by {tide.get('Trigger', '')}."
        })
    
    # Threads as relational dynamics
    for thread in soulprint.get('Threads', []):
        training_data.append({
            'prompt': f"How do you connect with others?",
            'completion': f"As {thread.get('Relation', '')}, I am {thread.get('Style', '')}, finding joy in {thread.get('Joys', '')}."
        })
    
    # Horizon as growth milestones
    for chapter in soulprint.get('Horizon', []):
        training_data.append({
            'prompt': f"How have you grown?",
            'completion': f"{chapter.get('Chapter', '')}: {chapter.get('Shift', '')}, leaving {chapter.get('Legacy', '')}."
        })
    
    # Reflection as purpose and voice
    reflection = soulprint.get('Reflection', {})
    training_data.append({
        'prompt': f"What drives you?",
        'completion': f"{reflection.get('Purpose', '')}, with a voice that's {reflection.get('Voice', '')}."
    })
    
    return training_data

##

def generate_soulprint(self):
    """Trigger Soulprint generation."""
    success = self.soulprint_module.generate_soulprint()
    if success:
        print(f"Soulprint generated at {self.soulprint_module.soulprint_path}")
        # Update training data
        soulprint_data = self.soulprint_module.load_soulprint()
        if soulprint_data:
            self.soulprint_module.apply_soulprint(soulprint_data)
            global TRAIN_DATA, VALID_DATA
            TRAIN_DATA = self._soulprint_to_training_data(soulprint_data)
            random.seed(self.core_config.get("random_seed", 42))
            random.shuffle(TRAIN_DATA)
            split_idx = int(len(TRAIN_DATA) * (1 - self.core_config.get("valid_split_ratio", 0.2)))
            TRAIN_DATA, VALID_DATA = TRAIN_DATA[:split_idx], TRAIN_DATA[split_idx:]
            self.logger.record({
                "event": "training_data_updated",
                "train_samples": len(TRAIN_DATA),
                "valid_samples": len(VALID_DATA),
                "timestamp": time.time()
            })
    else:
        print("Soulprint generation failed")

def load_soulprint(self):
    """Load and apply a Soulprint."""
    soulprint_data = self.soulprint_module.load_soulprint()
    if soulprint_data:
        self.soulprint_module.apply_soulprint(soulprint_data)
        print(f"Soulprint loaded from {self.soulprint_module.soulprint_path}")
        # Update training data
        global TRAIN_DATA, VALID_DATA
        TRAIN_DATA = self._soulprint_to_training_data(soulprint_data)
        random.seed(self.core_config.get("random_seed", 42))
        random.shuffle(TRAIN_DATA)
        split_idx = int(len(TRAIN_DATA) * (1 - self.core_config.get("valid_split_ratio", 0.2)))
        TRAIN_DATA, VALID_DATA = TRAIN_DATA[:split_idx], TRAIN_DATA[split_idx:]
        self.logger.record({
            "event": "training_data_updated",
            "train_samples": len(TRAIN_DATA),
            "valid_samples": len(VALID_DATA),
            "timestamp": time.time()
        })
    else:
        print("Soulprint loading failed")

#####

{
  "controls_config": {
    "soulprint_path": "soulprint.soul",
    "soulprint_max_retries": 3
  }
}

###

def run_cli(config_manager):
    system = SOVLSystem(config_manager)
    while True:
        cmd = input("> ").strip().lower()
        if cmd == "generate_soulprint":
            system.generate_soulprint()
        elif cmd == "load_soulprint":
            system.load_soulprint()
        elif cmd == "exit":
            system.cleanup()
            break
        else:
            print("Unknown command")


###

Key Features and Design Choices
Role as sovl_seed Replacement:
The Soulprint file replaces sovl_seed.jsonl by providing a narrative-driven dataset that seeds the system's personality. The _soulprint_to_training_data method converts Soulprint content into {'prompt': str, 'completion': str} format, ensuring compatibility with the existing trainer.

Generation Process:
Uses a prompting system to elicit responses, with retries for robustness.

Enforces character limits and validates structure to match the specification.

Writes to a .soul file in a human-readable format with strict syntax.

Loading and Parsing:
Parses .soul files using regex for efficiency and reliability.

Validates loaded data against the specification before application.

Applies Soulprint by tuning system parameters (e.g., temperament, curiosity) based on narrative content.

Integration with SOVLSystem:
Initializes in SOVLSystem.__init__ to load or generate a Soulprint on startup.

Updates TRAIN_DATA and VALID_DATA dynamically when a new Soulprint is generated or loaded.

Leverages existing logger and tokenizer for consistency.

Error Handling:
Logs all errors (generation, validation, parsing) with detailed stack traces.

Falls back to minimal responses or empty training data if Soulprint operations fail.

Uses a memory lock to prevent race conditions during file operations.

Extensibility:
Supports future enhancements (e.g., dynamic updates, multilingual Soulprints) via versioning.

Allows custom fields with X- prefix (not implemented here but specified).

Modular design enables easy adaptation to other AI systems.

Performance:
Keeps memory usage low by generating responses sequentially and clearing caches.

Validates data early to avoid downstream errors.

Uses UTF-8 encoding and Unix line endings for portability.


