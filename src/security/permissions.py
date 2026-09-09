"""
Security permission manager for validating filesystem boundaries.
"""

from pathlib import Path
from typing import Union


class PermissionManager:
    """Enforces workspace boundary sandboxing for all filesystem operations."""

    def __init__(self, workspace_dir: Union[str, Path]) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()

    def validate_path(self, target_path: Union[str, Path]) -> Path:
        """
        Resolve target path and verify it resides within the authorized workspace directory.
        Raises PermissionError if path attempts path traversal outside workspace.
        """
        path_obj = Path(target_path)
        if not path_obj.is_absolute():
            resolved = (self.workspace_dir / path_obj).resolve()
        else:
            resolved = path_obj.resolve()

        try:
            resolved.relative_to(self.workspace_dir)
        except ValueError:
            raise PermissionError(
                f"Access denied: Path '{target_path}' resolves to '{resolved}', which is outside the designated workspace '{self.workspace_dir}'."
            )

        return resolved
