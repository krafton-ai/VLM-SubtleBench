"""Data loading utilities for the unified dataset format."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class DatasetLoader:
    """Handles loading items from the unified qa.json format."""

    def load_items(
        self,
        qa_path: str,
        category: Optional[str] = None,
        domain: Optional[str] = None,
        has_caption: Optional[bool] = None,
        split: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Load items from qa.json with optional filtering.

        Args:
            qa_path: Path to qa.json file
            category: Filter by category (e.g. "state", "attribute"). None = all.
            domain: Filter by domain (e.g. "natural", "medical"). None = all.
            has_caption: Filter by has_caption field. None = no filter.
            split: Filter by split (e.g. "test", "val"). None = all.

        Returns:
            List of data items matching the filters
        """
        qa_file = Path(qa_path)
        if not qa_file.is_file():
            raise FileNotFoundError(f"qa.json not found at: {qa_path}")

        with open(qa_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {qa_path}, got {type(data).__name__}")

        # Apply filters
        items = []
        for item in data:
            metadata = item.get("metadata", {}) or {}

            if category is not None and metadata.get("category") != category:
                continue
            if domain is not None and metadata.get("domain") != domain:
                continue
            if has_caption is not None and item.get("has_caption") != has_caption:
                continue
            if split is not None and item.get("split") != split:
                continue

            items.append(item)

        return items
