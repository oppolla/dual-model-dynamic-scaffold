from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor
import re
from typing import Dict, Any
from sovl_logger import Logger
from sovl_error import ErrorHandler

class SoulParser(NodeVisitor):
    """Parse a .soul file into a structured dictionary."""
    
    def __init__(self, logger: Logger):
        self.logger = logger
        self.data = {"metadata": {}, "sections": {}}
        self.current_section = None

    def visit_section(self, node, visited_children):
        self.current_section = node.text.strip("[]")
        self.data["sections"][self.current_section] = {}

    def visit_field(self, node, visited_children):
        key, value = node.text.split(":", 1)
        key = key.strip()
        value = value.strip()
        if self.current_section:
            self.data["sections"][self.current_section][key] = value
        else:
            self.data["metadata"][key] = value

    def visit_list_item(self, node, visited_children):
        match = re.match(r"-\s*(\w+):\s*(.+)", node.text.strip())
        if match and self.current_section:
            key, value = match.groups()
            if key not in self.data["sections"][self.current_section]:
                self.data["sections"][self.current_section][key] = []
            self.data["sections"][self.current_section][key].append(value)

    def generic_visit(self, node, visited_children):
        return node

def parse_soul_file(file_path: str, logger: Logger) -> Dict[str, Any]:
    """Parse a .soul file and validate its contents."""
    grammar = Grammar(
        r"""
        soul_file = header metadata section*
        header = "%SOULPRINT\n%VERSION: v" version "\n"
        version = ~r"\d+\.\d+\.\d+"
        metadata = (field / comment)*
        section = section_header (field / list_item / comment)*
        section_header = "[" ~r"\w+" "]" "\n"
        field = ~r"^\s*\w+:\s*.+$" "\n"
        list_item = ~r"^\s*-\s*\w+:\s*.+$" "\n"
        comment = ~r"^\s*#.*$" "\n"
        """
    )
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        parser = SoulParser(logger)
        tree = grammar.parse(text)
        parsed_data = parser.visit(tree)

        # Validate constraints
        if parsed_data["metadata"].get("Consent") != "true":
            logger.log_error("Consent not granted", "soul_validation_error")
            raise ValueError("Consent not granted")
        if "Identity" not in parsed_data["sections"] or not re.match(r"^[A-Za-z0-9_-]{1,50}$", parsed_data["sections"]["Identity"]["Name"]):
            logger.log_error("Invalid Name format", "soul_validation_error")
            raise ValueError("Invalid Name format")
        if "Echoes" in parsed_data["sections"] and len(parsed_data["sections"]["Echoes"].get("Memory", [])) != 142:
            logger.log_error("Expected 142 Chronicle entries", "soul_validation_error")
            raise ValueError("Expected 142 Chronicle entries")

        logger.record_event(
            event_type="soul_parsed",
            message="Successfully parsed .soul file",
            level="info",
            additional_info={"file_path": file_path}
        )
        return parsed_data

    except Exception as e:
        logger.log_error(
            error_msg=str(e),
            error_type="soul_parsing_error",
            stack_trace=traceback.format_exc(),
            additional_info={"file_path": file_path}
        )
        raise
