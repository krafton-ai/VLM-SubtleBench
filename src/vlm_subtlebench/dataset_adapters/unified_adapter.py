"""Unified adapter for the dataset format."""

import os
import random
from typing import Dict, Any, Optional

from .base_adapter import BaseDatasetAdapter, ProcessedDataItem


class UnifiedAdapter(BaseDatasetAdapter):
    """Adapter for the dataset format.

    Expected item format (JSONL, flattened metadata):
    {
        "image_1": "https://huggingface.co/.../0.png",
        "image_2": "https://huggingface.co/.../0_2.png",
        "question": "In which image is the apple peeled more?",
        "answer": "second image",
        "distractors": ["first image"],
        "has_caption": false,
        "caption": null,
        "category": "state",
        "domain": "natural",
        "source": "changeit",
        "source_id": "0",
        "raw_folder": "changeIt_state_pairs",
        "generation_info": null
    }
    """

    REQUIRED_FIELDS = {"image_1", "image_2", "question", "answer", "distractors"}

    def can_handle(self, data: Dict[str, Any]) -> bool:
        """Check if this adapter can handle the given data format."""
        return self.REQUIRED_FIELDS.issubset(data.keys())

    def is_valid_item(self, item: Dict[str, Any]) -> bool:
        """Check if an item has the required fields."""
        return self.REQUIRED_FIELDS.issubset(item.keys())

    def _resolve_image_path(self, path: str, dataset_path: str) -> str:
        """Resolve an image path: try local first, fall back to URL."""
        if path.startswith("http://") or path.startswith("https://"):
            # Extract relative path from HuggingFace URL and check locally
            marker = "/resolve/main/"
            idx = path.find(marker)
            if idx != -1:
                relative_path = path[idx + len(marker):]
                local_path = os.path.join(dataset_path, relative_path)
                if os.path.isfile(local_path):
                    return local_path
            return path
        return os.path.join(dataset_path, path)

    def process_item(
        self,
        data: Dict[str, Any],
        dataset_path: str,
        task_type: str = "multiple_choice",
    ) -> Optional[ProcessedDataItem]:
        """Process a dataset item into standardized format."""
        if not self.can_handle(data):
            return None

        # Build image paths (handle both URLs and relative paths)
        first_image_path = self._resolve_image_path(data["image_1"], dataset_path)
        second_image_path = self._resolve_image_path(data["image_2"], dataset_path)

        # Item ID: {category}_{source}_{source_id}, with fallback
        category = data.get("category", "")
        source = data.get("source", "")
        source_id = str(data.get("source_id", ""))
        if category and source and source_id:
            item_id = f"{category}_{source}_{source_id}"
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
                "category": category,
                "domain": data.get("domain", ""),
                "source": source,
                "source_id": source_id,
            },
        )

    def get_format_name(self) -> str:
        return "unified"
