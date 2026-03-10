"""Data loading utilities for the unified dataset format."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class DatasetLoader:
    """Handles loading items from JSONL dataset files (data/test.jsonl, data/val.jsonl)."""

    def load_items(
        self,
        qa_path: str,
        category: Optional[str] = None,
        domain: Optional[str] = None,
        has_caption: Optional[bool] = None,
        split: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Load items from a JSONL file with optional filtering.

        Args:
            qa_path: Path to JSONL file (e.g. data/test.jsonl)
            category: Filter by category (e.g. "state", "attribute"). None = all.
            domain: Filter by domain (e.g. "natural", "medical"). None = all.
            has_caption: Filter by has_caption field. None = no filter.
            split: Unused (kept for backward compatibility; split is determined by file).

        Returns:
            List of data items matching the filters
        """
        qa_file = Path(qa_path)
        if not qa_file.is_file():
            raise FileNotFoundError(f"Dataset file not found at: {qa_path}")

        data = []
        with open(qa_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        # Apply filters (metadata fields are now at top level)
        items = []
        for item in data:
            if category is not None and item.get("category") != category:
                continue
            if domain is not None and item.get("domain") != domain:
                continue
            if has_caption is not None and item.get("has_caption") != has_caption:
                continue
            items.append(item)

        return items
