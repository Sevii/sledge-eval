"""Convert Sork game voice command logs to sledge-eval test suites."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class CommandRecord:
    """Represents a single voice command extracted from logs."""

    transcription: str
    tool: Optional[str]
    arguments: Dict[str, Any]
    success: bool
    timestamp: str
    game_id: Optional[str] = None
    reasoning: Optional[str] = None

    def normalized_transcription(self) -> str:
        """Return lowercase transcription with punctuation stripped for deduplication."""
        text = self.transcription.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()


@dataclass
class TestSuiteData:
    """Data structure for the output test suite."""

    name: str
    description: Optional[str] = None
    tests: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {"name": self.name, "tests": self.tests}
        if self.description:
            result["description"] = self.description
        return result


class SorkConverter:
    """Converts Sork game JSONL logs to sledge-eval test suite format."""

    # Patterns to exclude from test suite
    NOISE_PATTERNS = [
        r'^\s*$',                    # Empty or whitespace only
        r'^\[\s*silence\s*\]$',      # [Silence] marker
        r'^-\s+',                    # Starts with dash (often garbled)
        r'^\.\s*$',                  # Just punctuation
    ]

    # Tool to tag mappings
    TOOL_TAGS = {
        'go_north': ['sork', 'navigation'],
        'go_south': ['sork', 'navigation'],
        'go_east': ['sork', 'navigation'],
        'go_west': ['sork', 'navigation'],
        'go_up': ['sork', 'navigation'],
        'go_down': ['sork', 'navigation'],
        'take': ['sork', 'items'],
        'drop': ['sork', 'items'],
        'use': ['sork', 'items'],
        'look': ['sork', 'inspection'],
        'examine': ['sork', 'inspection'],
        'search': ['sork', 'inspection'],
        'inventory': ['sork', 'inspection'],
        'attack': ['sork', 'combat'],
        'flee': ['sork', 'combat'],
        'help': ['sork', 'meta'],
    }

    def __init__(self, debug: bool = False):
        """Initialize the converter.

        Args:
            debug: Enable verbose logging
        """
        self.debug = debug
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.NOISE_PATTERNS]

    def _log(self, message: str) -> None:
        """Print debug message if debug mode is enabled."""
        if self.debug:
            print(f"[DEBUG] {message}")

    def load_jsonl(self, filepath: Path) -> List[dict]:
        """Load and parse a JSONL file.

        Args:
            filepath: Path to the JSONL file

        Returns:
            List of parsed JSON objects
        """
        records = []
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    self._log(f"JSON error in {filepath}:{line_num}: {e}")
        return records

    def _is_noise(self, transcription: str) -> bool:
        """Check if a transcription is noise that should be filtered out."""
        for pattern in self._compiled_patterns:
            if pattern.match(transcription):
                return True
        return False

    def extract_commands(self, logs_dir: Path) -> List[CommandRecord]:
        """Extract command records from all game session logs.

        Args:
            logs_dir: Directory containing JSONL log files

        Returns:
            List of CommandRecord objects
        """
        records = []

        # Find all game session files
        session_files = sorted(logs_dir.glob('game_session_*.jsonl'))
        self._log(f"Found {len(session_files)} game session files")

        for session_file in session_files:
            self._log(f"Processing {session_file.name}")
            log_entries = self.load_jsonl(session_file)

            for entry in log_entries:
                if entry.get('event') != 'command':
                    continue

                transcription = entry.get('transcription', '')
                if not transcription or self._is_noise(transcription):
                    self._log(f"  Skipping noise: '{transcription}'")
                    continue

                tool_selection = entry.get('tool_selection', {})
                tool = tool_selection.get('tool')
                arguments = tool_selection.get('arguments', {})
                reasoning = tool_selection.get('reasoning')
                success = entry.get('success', False)
                timestamp = entry.get('timestamp', '')
                game_id = entry.get('game_id')

                record = CommandRecord(
                    transcription=transcription,
                    tool=tool,
                    arguments=arguments if arguments else {},
                    success=success,
                    timestamp=timestamp,
                    game_id=game_id,
                    reasoning=reasoning,
                )
                records.append(record)
                self._log(f"  Extracted: '{transcription}' -> {tool}")

        self._log(f"Total commands extracted: {len(records)}")
        return records

    def filter_commands(
        self,
        records: List[CommandRecord],
        success_only: bool = False
    ) -> List[CommandRecord]:
        """Filter command records based on criteria.

        Args:
            records: List of command records
            success_only: If True, only include successful commands

        Returns:
            Filtered list of command records
        """
        if success_only:
            filtered = [r for r in records if r.success]
            self._log(f"Filtered to {len(filtered)} successful commands (from {len(records)})")
            return filtered
        return records

    def deduplicate(
        self,
        records: List[CommandRecord],
        strategy: str = 'exact'
    ) -> List[CommandRecord]:
        """Remove duplicate commands.

        Args:
            records: List of command records
            strategy: Deduplication strategy:
                - 'exact': Exact transcription match
                - 'normalized': Normalize before comparison (lowercase, strip punctuation)
                - 'none': No deduplication

        Returns:
            Deduplicated list of command records
        """
        if strategy == 'none':
            return records

        seen: Set[str] = set()
        deduplicated = []

        for record in records:
            if strategy == 'normalized':
                key = record.normalized_transcription()
            else:  # exact
                key = record.transcription

            if key not in seen:
                seen.add(key)
                deduplicated.append(record)
            else:
                self._log(f"Deduplicating: '{record.transcription}'")

        self._log(f"Deduplicated to {len(deduplicated)} commands (from {len(records)})")
        return deduplicated

    def _get_tags(self, record: CommandRecord) -> List[str]:
        """Get tags for a command record based on tool and success status."""
        tags = self.TOOL_TAGS.get(record.tool, ['sork']) if record.tool else ['sork']
        tags = list(tags)  # Copy to avoid modifying the class attribute

        if not record.success:
            if record.tool is None:
                tags.append('edge_case')
                tags.append('unrecognized')
            else:
                tags.append('edge_case')
                tags.append('failed')

        return tags

    def convert_to_test_suite(
        self,
        records: List[CommandRecord],
        name: str = "Sork Voice Commands",
        description: Optional[str] = None
    ) -> TestSuiteData:
        """Convert command records to a test suite.

        Args:
            records: List of command records
            name: Name for the test suite
            description: Optional description

        Returns:
            TestSuiteData object
        """
        suite = TestSuiteData(
            name=name,
            description=description or "Voice command test suite generated from Sork gameplay logs"
        )

        for i, record in enumerate(records, 1):
            test_id = f"sork_{i:03d}"

            # Build expected tool calls
            expected_tool_calls = []
            if record.tool:
                expected_tool_calls.append({
                    "name": record.tool,
                    "arguments": record.arguments
                })

            tags = self._get_tags(record)

            test = {
                "id": test_id,
                "voice_command": record.transcription,
                "expected_tool_calls": expected_tool_calls,
                "tags": tags,
            }

            suite.tests.append(test)
            self._log(f"Created test {test_id}: '{record.transcription}' -> {record.tool}")

        self._log(f"Test suite created with {len(suite.tests)} tests")
        return suite

    def extract_tool_definitions(self, logs_dir: Path) -> List[Dict[str, Any]]:
        """Extract tool definitions from LLM log files.

        Args:
            logs_dir: Directory containing JSONL log files

        Returns:
            List of tool definitions in OpenAI format
        """
        # Find LLM log files
        llm_files = sorted(logs_dir.glob('llm_log_*.jsonl'))
        self._log(f"Found {len(llm_files)} LLM log files")

        # Try to find the file with the most tools
        best_tools: List[Dict[str, Any]] = []

        for llm_file in llm_files:
            try:
                entries = self.load_jsonl(llm_file)
                if entries:
                    tools = entries[0].get('request', {}).get('tools', [])
                    if len(tools) > len(best_tools):
                        best_tools = tools
                        self._log(f"Found {len(tools)} tools in {llm_file.name}")
            except Exception as e:
                self._log(f"Error reading {llm_file}: {e}")

        self._log(f"Extracted {len(best_tools)} tool definitions")
        return best_tools

    def save_test_suite(self, suite: TestSuiteData, output_path: Path) -> None:
        """Save test suite to a JSON file.

        Args:
            suite: The test suite data
            output_path: Path to write the JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(suite.to_dict(), f, indent=2)

        self._log(f"Saved test suite to {output_path}")

    def convert(
        self,
        logs_dir: Path,
        output_path: Path,
        success_only: bool = False,
        dedupe_strategy: str = 'exact',
        suite_name: str = "Sork Voice Commands",
        suite_description: Optional[str] = None
    ) -> TestSuiteData:
        """Full conversion pipeline from logs to test suite.

        Args:
            logs_dir: Directory containing JSONL log files
            output_path: Path to write the output JSON file
            success_only: If True, only include successful commands
            dedupe_strategy: Deduplication strategy ('exact', 'normalized', 'none')
            suite_name: Name for the test suite
            suite_description: Optional description

        Returns:
            The generated TestSuiteData
        """
        self._log(f"Starting conversion from {logs_dir}")

        # Extract commands
        records = self.extract_commands(logs_dir)

        # Filter
        records = self.filter_commands(records, success_only=success_only)

        # Deduplicate
        records = self.deduplicate(records, strategy=dedupe_strategy)

        # Convert to test suite
        suite = self.convert_to_test_suite(
            records,
            name=suite_name,
            description=suite_description
        )

        # Save
        self.save_test_suite(suite, output_path)

        return suite
