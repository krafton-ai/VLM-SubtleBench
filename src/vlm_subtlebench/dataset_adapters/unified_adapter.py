"""Unified adapter for the consolidated qa.json format."""

import os
import random
from typing import Dict, Any, Optional

from .base_adapter import BaseDatasetAdapter, ProcessedDataItem


class UnifiedAdapter(BaseDatasetAdapter):
    """Adapter for the unified qa.json format.

    Expected item format:
    {
        "image_1": "images/changeit/state/0.png",
        "image_2": "images/changeit/state/0_2.png",
        "question": "In which image is the apple peeled more?",
        "answer": "second image",
        "distractors": ["first image"],
        "has_caption": false,
        "caption": null,
        "metadata": {"category": "state", "domain": "natural", "source": "changeit", ...},
        "split": "test"
    }
    """

    REQUIRED_FIELDS = {"image_1", "image_2", "question", "answer", "distractors"}

    def can_handle(self, data: Dict[str, Any]) -> bool:
        """Check if this adapter can handle the given data format."""
        return self.REQUIRED_FIELDS.issubset(data.keys())

    def is_valid_item(self, item: Dict[str, Any]) -> bool:
        """Check if an item has the required fields."""
        return self.REQUIRED_FIELDS.issubset(item.keys())

    def process_item(
        self,
        data: Dict[str, Any],
        dataset_path: str,
        task_type: str = "multiple_choice",
    ) -> Optional[ProcessedDataItem]:
        """Process a unified format item into standardized format."""
        if not self.can_handle(data):
            return None

        # Build absolute image paths
        first_image_path = os.path.join(dataset_path, data["image_1"])
        second_image_path = os.path.join(dataset_path, data["image_2"])

        # Item ID: {split}_{category}_{source}_{source_id}, with fallback
        metadata = data.get("metadata", {}) or {}
        split = data.get("split", "")
        category = metadata.get("category", "")
        source = metadata.get("source", "")
        source_id = str(metadata.get("source_id", ""))
        if split and category and source and source_id:
            item_id = f"{split}_{category}_{source}_{source_id}"
        else:
            item_id = f"{data['image_1']}_{data['image_2']}"

        # Build options for multiple choice
        answer = data["answer"]
        distractors = list(data["distractors"])
        options = [answer] + distractors
        random.shuffle(options)
        correct_index = options.index(answer)

        return ProcessedDataItem(
            item_id=item_id,
            question_text=data["question"],
            first_image_path=first_image_path,
            second_image_path=second_image_path,
            options=options,
            correct_index=correct_index,
            correct_answer=answer,
            caption=data.get("caption"),
            metadata={
                "has_caption": data.get("has_caption", False),
                "caption": data.get("caption"),
                **metadata,
            },
        )

    def get_format_name(self) -> str:
        return "unified"
