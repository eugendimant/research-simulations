"""
Group Registration and Usage Tracking System
=============================================
Manages student groups, tracks simulation usage, and controls access to
Final Data Collection Mode (one-time use per group).
"""

# Version identifier to help track deployed code
__version__ = "2.1.3"  # Synced all utils to same version

import hashlib
import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GroupMember:
    """A student in a group."""
    name: str
    email: Optional[str] = None


@dataclass
class RegisteredGroup:
    """A registered student group."""
    group_number: int
    members: List[GroupMember]
    project_title: Optional[str] = None
    registered_at: str = ""
    final_mode_used: bool = False
    final_mode_used_at: Optional[str] = None
    final_mode_run_id: Optional[str] = None
    pilot_runs_count: int = 0
    last_pilot_run: Optional[str] = None


class GroupManager:
    """
    Manages group registration and simulation usage tracking.

    Features:
    - Stores registered groups with member info
    - Tracks pilot mode usage (unlimited)
    - Enforces one-time Final Data Collection Mode per group
    - Persists data to JSON file
    - Thread-safe operations
    """

    def __init__(self, storage_path: str = None):
        """
        Initialize the group manager.

        Args:
            storage_path: Path to JSON storage file. If None, uses default location.
        """
        if storage_path is None:
            storage_path = Path(__file__).parent.parent / "data" / "groups.json"

        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._groups: Dict[int, RegisteredGroup] = {}

        # Load existing data
        self._load()

    def _load(self):
        """Load groups from storage file."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for group_num, group_data in data.get('groups', {}).items():
                        members = [
                            GroupMember(**m) for m in group_data.pop('members', [])
                        ]
                        self._groups[int(group_num)] = RegisteredGroup(
                            members=members, **group_data
                        )
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load groups file: {e}")
                self._groups = {}

    def _save(self):
        """Save groups to storage file."""
        data = {
            'groups': {},
            'last_updated': datetime.now().isoformat()
        }
        for group_num, group in self._groups.items():
            group_dict = asdict(group)
            data['groups'][str(group_num)] = group_dict

        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def register_group(
        self,
        group_number: int,
        members: List[Dict[str, str]],
        project_title: str = None
    ) -> Tuple[bool, str]:
        """
        Register a new group or update existing group.

        Args:
            group_number: The group number
            members: List of dicts with 'name' and optional 'email'
            project_title: Optional project title

        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            member_objects = [
                GroupMember(name=m.get('name', 'Unknown'), email=m.get('email'))
                for m in members
                if isinstance(m, dict)  # Skip non-dict entries
            ]

            if group_number in self._groups:
                # Update existing group (but preserve usage tracking)
                existing = self._groups[group_number]
                existing.members = member_objects
                existing.project_title = project_title or existing.project_title
                message = f"Group {group_number} updated successfully"
            else:
                # New group
                self._groups[group_number] = RegisteredGroup(
                    group_number=group_number,
                    members=member_objects,
                    project_title=project_title,
                    registered_at=datetime.now().isoformat()
                )
                message = f"Group {group_number} registered successfully"

            self._save()
            return True, message

    def is_group_registered(self, group_number: int) -> bool:
        """Check if a group is registered."""
        return group_number in self._groups

    def get_group(self, group_number: int) -> Optional[RegisteredGroup]:
        """Get group information."""
        return self._groups.get(group_number)

    def get_all_groups(self) -> Dict[int, RegisteredGroup]:
        """Get all registered groups."""
        return dict(self._groups)

    def can_use_final_mode(self, group_number: int) -> Tuple[bool, str]:
        """
        Check if a group can use Final Data Collection Mode.

        Returns:
            Tuple of (can_use, reason)
        """
        if group_number not in self._groups:
            return False, "Group not registered. Please contact instructor."

        group = self._groups[group_number]

        if group.final_mode_used:
            return False, (
                f"Group {group_number} has already used Final Data Collection Mode "
                f"on {group.final_mode_used_at}. Each group can only use it once."
            )

        return True, "Group is eligible for Final Data Collection Mode"

    def record_pilot_run(
        self,
        group_number: int,
        run_id: str = None
    ) -> Tuple[bool, str]:
        """
        Record a pilot mode simulation run.

        Args:
            group_number: The group number
            run_id: Optional run identifier

        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            if group_number not in self._groups:
                # Auto-register for pilot mode
                return True, "Pilot run recorded (group not pre-registered)"

            group = self._groups[group_number]
            group.pilot_runs_count += 1
            group.last_pilot_run = datetime.now().isoformat()

            self._save()
            return True, f"Pilot run #{group.pilot_runs_count} recorded for Group {group_number}"

    def record_final_mode_use(
        self,
        group_number: int,
        run_id: str
    ) -> Tuple[bool, str]:
        """
        Record Final Data Collection Mode usage.

        This marks the group as having used their one-time final mode.

        Args:
            group_number: The group number
            run_id: The simulation run ID

        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            can_use, reason = self.can_use_final_mode(group_number)
            if not can_use:
                return False, reason

            group = self._groups[group_number]
            group.final_mode_used = True
            group.final_mode_used_at = datetime.now().isoformat()
            group.final_mode_run_id = run_id

            self._save()
            return True, f"Final Data Collection Mode used by Group {group_number}. Run ID: {run_id}"

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get summary of all group usage."""
        summary = {
            'total_registered_groups': len(self._groups),
            'groups_used_final_mode': sum(1 for g in self._groups.values() if g.final_mode_used),
            'total_pilot_runs': sum(g.pilot_runs_count for g in self._groups.values()),
            'groups': []
        }

        for group_num, group in sorted(self._groups.items()):
            summary['groups'].append({
                'group_number': group_num,
                'member_count': len(group.members),
                'member_names': [m.name for m in group.members],
                'project_title': group.project_title,
                'pilot_runs': group.pilot_runs_count,
                'final_mode_used': group.final_mode_used,
                'final_mode_date': group.final_mode_used_at
            })

        return summary

    def reset_group_final_mode(self, group_number: int) -> Tuple[bool, str]:
        """
        Reset a group's final mode usage (instructor only).

        Args:
            group_number: The group number to reset

        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            if group_number not in self._groups:
                return False, f"Group {group_number} not found"

            group = self._groups[group_number]
            group.final_mode_used = False
            group.final_mode_used_at = None
            group.final_mode_run_id = None

            self._save()
            return True, f"Group {group_number} final mode usage reset"


class APIKeyManager:
    """
    Manages Claude API key for Final Data Collection Mode.

    The API key is stored securely and only used for final mode simulations.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize API key manager.

        Args:
            config_path: Path to config file. If None, uses default location.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "data" / "api_config.json"

        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_config(self):
        """Save configuration to file."""
        with open(self.config_path, 'w') as f:
            json.dump(self._config, f, indent=2)

    def set_api_key(self, api_key: str) -> Tuple[bool, str]:
        """
        Set the Claude API key.

        Args:
            api_key: The Anthropic API key

        Returns:
            Tuple of (success, message)
        """
        if not api_key or not api_key.startswith('sk-ant-'):
            return False, "Invalid API key format. Should start with 'sk-ant-'"

        # Store hash of key for verification (not the key itself in logs)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]

        self._config['api_key'] = api_key
        self._config['api_key_hash'] = key_hash
        self._config['api_key_set_at'] = datetime.now().isoformat()

        self._save_config()
        return True, f"API key set successfully (hash: {key_hash}...)"

    def get_api_key(self) -> Optional[str]:
        """Get the stored API key."""
        return self._config.get('api_key')

    def is_api_key_set(self) -> bool:
        """Check if API key is configured."""
        return bool(self._config.get('api_key'))

    def clear_api_key(self) -> Tuple[bool, str]:
        """Clear the stored API key."""
        if 'api_key' in self._config:
            del self._config['api_key']
            self._config['api_key_cleared_at'] = datetime.now().isoformat()
            self._save_config()
            return True, "API key cleared"
        return False, "No API key was set"

    def enable_final_mode(self) -> Tuple[bool, str]:
        """Enable final mode (requires API key)."""
        if not self.is_api_key_set():
            return False, "Cannot enable final mode: No API key configured"

        self._config['final_mode_enabled'] = True
        self._config['final_mode_enabled_at'] = datetime.now().isoformat()
        self._save_config()
        return True, "Final Data Collection Mode enabled"

    def disable_final_mode(self) -> Tuple[bool, str]:
        """Disable final mode."""
        self._config['final_mode_enabled'] = False
        self._save_config()
        return True, "Final Data Collection Mode disabled"

    def is_final_mode_enabled(self) -> bool:
        """Check if final mode is enabled."""
        return self._config.get('final_mode_enabled', False) and self.is_api_key_set()


def create_sample_groups_file(storage_path: str = None) -> str:
    """
    Create a sample groups file with placeholder data.

    Returns:
        Path to the created file
    """
    if storage_path is None:
        storage_path = Path(__file__).parent.parent / "data" / "groups.json"

    storage_path = Path(storage_path)
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    sample_data = {
        "groups": {
            "1": {
                "group_number": 1,
                "members": [
                    {"name": "Student 1A", "email": None},
                    {"name": "Student 1B", "email": None},
                    {"name": "Student 1C", "email": None}
                ],
                "project_title": "Sample Project - Group 1",
                "registered_at": datetime.now().isoformat(),
                "final_mode_used": False,
                "final_mode_used_at": None,
                "final_mode_run_id": None,
                "pilot_runs_count": 0,
                "last_pilot_run": None
            },
            "2": {
                "group_number": 2,
                "members": [
                    {"name": "Student 2A", "email": None},
                    {"name": "Student 2B", "email": None}
                ],
                "project_title": "Sample Project - Group 2",
                "registered_at": datetime.now().isoformat(),
                "final_mode_used": False,
                "final_mode_used_at": None,
                "final_mode_run_id": None,
                "pilot_runs_count": 0,
                "last_pilot_run": None
            }
        },
        "last_updated": datetime.now().isoformat(),
        "_instructions": {
            "note": "Replace the sample groups with actual student groups",
            "format": {
                "group_number": "Integer group identifier",
                "members": "List of {name, email} objects",
                "project_title": "Optional project title",
                "final_mode_used": "DO NOT EDIT - tracks one-time usage"
            }
        }
    }

    with open(storage_path, 'w') as f:
        json.dump(sample_data, f, indent=2)

    return str(storage_path)


# Export
__all__ = [
    'GroupManager',
    'RegisteredGroup',
    'GroupMember',
    'APIKeyManager',
    'create_sample_groups_file'
]
