"""Base adapter class for dataset formats."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ProcessedDataItem:
    """Standardized data item structure for evaluation tasks."""

    item_id: str
    question_text: str
    first_image_path: str
    second_image_path: str
    # Multiple choice specific fields (can be None for free-form tasks)
    options: Optional[List[str]] = None
    correct_index: Optional[int] = None
    correct_answer: Optional[str] = None
    # Free-form specific fields
    caption: Optional[str] = None
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None


class BaseDatasetAdapter(ABC):
    """Abstract base class for dataset format adapters."""

    @abstractmethod
    def can_handle(self, data: Dict[str, Any]) -> bool:
        """Check if this adapter can handle the given data format."""
        pass

    @abstractmethod
    def is_valid_item(self, item: Dict[str, Any]) -> bool:
        """Check if an item has the required fields for this format."""
        pass

    @abstractmethod
    def process_item(
        self,
        data: Dict[str, Any],
        dataset_path: str,
        task_type: str = "multiple_choice",
    ) -> Optional[ProcessedDataItem]:
        """Process a data item into standardized format.

        Args:
            data: Raw data item
            dataset_path: Root directory of the dataset
            task_type: Type of task ("multiple_choice" or "free_form")
        """
        pass

    @abstractmethod
    def get_format_name(self) -> str:
        """Get a human-readable name for this format."""
        pass
