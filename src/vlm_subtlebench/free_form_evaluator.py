import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

from vlm_subtlebench.base_agent import BaseAgent
from vlm_subtlebench.llms.openai_utils import Logger
from vlm_subtlebench.llms.typing import Message
from vlm_subtlebench.prompts.free_form_evaluator import (
    get_system_prompt_by_type,
    get_user_prompt_by_type,
)
from vlm_subtlebench.json_schemas import ImageComparisonSchema
from vlm_subtlebench.dataset_adapters import (
    UnifiedAdapter,
    DatasetLoader,
)
from vlm_subtlebench.utils import encode_image_to_base64


class FreeFormEvaluator(BaseAgent):
    @dataclass
    class Config(BaseAgent.Config):
        # Model configuration (inherited from BaseAgent)
        prompt_type: str = "standard"
        use_structured_output: bool = True

    cfg: Config  # add this to every subclass to enable static type checking

    def configure(self, *args, **kwargs) -> None:
        super().configure(*args, **kwargs)

        self.prompt_type = self.cfg.prompt_type
        self.use_structured_output = self.cfg.use_structured_output
        self.adapter = UnifiedAdapter()
        self.data_loader = DatasetLoader()

    def create_comparison_messages(
        self, first_image_path: str, second_image_path: str
    ) -> List[Message]:
        """Create messages for image comparison task."""
        first_image_base64 = encode_image_to_base64(first_image_path)
        second_image_base64 = encode_image_to_base64(second_image_path)

        # Get the system and user prompts based on the prompt type
        system_prompt = get_system_prompt_by_type(self.prompt_type)
        user_prompt = get_user_prompt_by_type(self.prompt_type)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{first_image_base64}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{second_image_base64}"
                        },
                    },
                ],
            },
        ]

        return messages

    # --- Shared helpers ---

    def _parse_ff_response(self, response_content, structured: bool) -> Union[List[str], str]:
        """Parse a free-form LLM response into a list of differences.

        Handles both structured-output and plain-text responses.
        Returns a list of difference strings, or raw content on decode failure.
        """
        if not response_content:
            return []

        if structured:
            try:
                if isinstance(response_content, ImageComparisonSchema):
                    return response_content.differences
                differences = json.loads(response_content)
                if isinstance(differences, list):
                    return differences
                elif isinstance(differences, dict):
                    return differences.get("differences", [])
                else:
                    print(f"Warning: Expected list but got {type(differences)}")
                    return []
            except Exception as e:
                print(f"Error parsing structured response: {e}")
                return []
        else:
            try:
                differences = json.loads(response_content)
                if isinstance(differences, list):
                    return differences
                elif isinstance(differences, dict):
                    return differences.get("differences", [])
                else:
                    print(f"Warning: Expected list but got {type(differences)}")
                    return []
            except json.JSONDecodeError:
                return response_content

    def _prepare_ff_task(self, item: Dict[str, Any], dataset_path: str):
        """Prepare a single free-form task.

        Returns (task_id, messages, pair_info), a skip dict when image files are missing,
        or None if the item cannot be processed.
        """
        processed = self.adapter.process_item(
            item, dataset_path, task_type="free_form"
        )
        if not processed:
            return None

        missing = []
        if not processed.first_image_path.startswith(("http://", "https://")) and not os.path.isfile(processed.first_image_path):
            missing.append(processed.first_image_path)
        if not processed.second_image_path.startswith(("http://", "https://")) and not os.path.isfile(processed.second_image_path):
            missing.append(processed.second_image_path)
        if missing:
            logging.warning(
                "Skipping item %s: missing image file(s): %s",
                processed.item_id,
                missing,
            )
            return {
                "skip": True,
                "reason": "missing_image",
                "item_id": processed.item_id,
                "first_image_path": processed.first_image_path,
                "second_image_path": processed.second_image_path,
                "missing_paths": missing,
            }

        task_id = processed.item_id
        messages = self.create_comparison_messages(
            processed.first_image_path, processed.second_image_path
        )

        response_format = (
            ImageComparisonSchema if self.use_structured_output else None
        )

        pair_info = {
            "first_image": processed.first_image_path,
            "second_image": processed.second_image_path,
            "caption": processed.caption,
            "base_name": task_id,
        }

        call_llm_kwargs = {
            "messages": messages,
            "log_path": f"image_comparison_{Path(processed.first_image_path).stem}_{Path(processed.second_image_path).stem}",
            "response_format": response_format,
        }

        return task_id, call_llm_kwargs, pair_info

    def _finalize_ff_results(
        self,
        results_list: List[Dict[str, Any]],
        items: List[Dict[str, Any]],
        log_output_file: str,
        extra_metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Build summary dict, save JSON, print stats, return results."""
        final_results = {
            "model": self.model_name,
            "prompt_type": self.prompt_type,
            "use_structured_output": self.use_structured_output,
            "total_pairs": len(items),
            "evaluated_pairs": len(results_list),
            "total_cost": self.total_cost(),
            "results": results_list,
        }
        if extra_metadata:
            final_results.update(extra_metadata)

        with open(log_output_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        print(f"\nEvaluation completed!")
        print(f"Total pairs evaluated: {final_results['evaluated_pairs']}")
        print(f"Total cost: ${final_results['total_cost']:.4f}")
        print(f"Results saved to: {log_output_file}")
        print(f"Logs saved to: {self.log_path}")

        return final_results

    # --- Public evaluation methods ---

    def evaluate_single_pair(
        self, first_image_path: str, second_image_path: str
    ) -> List[str]:
        """Evaluate a single image pair and return the differences."""
        messages = self.create_comparison_messages(first_image_path, second_image_path)

        try:
            response_format = (
                ImageComparisonSchema if self.use_structured_output else None
            )

            response_content = self.call_llm(
                messages=messages,
                log_path=f"image_comparison_{Path(first_image_path).stem}_{Path(second_image_path).stem}",
                response_format=response_format,
            )

            return self._parse_ff_response(response_content, self.use_structured_output)

        except Exception as e:
            print(f"Error evaluating image pair: {e}")
            return []

    def evaluate_all_pairs(
        self,
        dataset_path: str = "VLM-SubtleBench",
        output_file: str = "evaluation_results.json",
        max_pairs: int = None,
        category: Optional[str] = None,
        domain: Optional[str] = None,
        split: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate all captioned image pairs and save results to JSON file."""
        split_file = split if split else "test"
        qa_path = os.path.join(dataset_path, "data", f"{split_file}.jsonl")
        items = self.data_loader.load_items(
            qa_path, category=category, domain=domain, has_caption=True, split=split
        )

        if max_pairs:
            items = items[:max_pairs]

        log_output_file = os.path.join(self.logger.log_path, output_file)

        print(f"Found {len(items)} captioned image pairs to evaluate")
        print(f"Using model: {self.model_name}")
        print(f"Using prompt type: {self.prompt_type}")
        print(f"Using structured output: {self.use_structured_output}")
        print(f"Logs will be saved to: {self.log_path}")
        print(f"Results will be saved to: {log_output_file}")

        results = {
            "model": self.model_name,
            "prompt_type": self.prompt_type,
            "use_structured_output": self.use_structured_output,
            "total_pairs": len(items),
            "evaluated_pairs": 0,
            "total_cost": 0.0,
            "results": [],
            "skipped_items": [],
        }

        for i, item in enumerate(items):
            processed = self.adapter.process_item(
                item, dataset_path, task_type="free_form"
            )
            if not processed:
                continue
            missing = []
            if not processed.first_image_path.startswith(("http://", "https://")) and not os.path.isfile(processed.first_image_path):
                missing.append(processed.first_image_path)
            if not processed.second_image_path.startswith(("http://", "https://")) and not os.path.isfile(processed.second_image_path):
                missing.append(processed.second_image_path)
            if missing:
                logging.warning(
                    "Skipping item %s: missing image file(s): %s",
                    processed.item_id,
                    missing,
                )
                results["skipped_items"].append({
                    "reason": "missing_image",
                    "item_id": processed.item_id,
                    "first_image_path": processed.first_image_path,
                    "second_image_path": processed.second_image_path,
                    "missing_paths": missing,
                })
                continue

            base_name = processed.item_id
            print(f"Evaluating pair {i+1}/{len(items)}: {base_name}")

            differences = self.evaluate_single_pair(
                processed.first_image_path, processed.second_image_path
            )

            result = {
                "base_name": base_name,
                "first_image": processed.first_image_path,
                "second_image": processed.second_image_path,
                "caption": processed.caption,
                "differences": differences,
                "num_differences": len(differences) if isinstance(differences, list) else 1,
            }

            results["results"].append(result)
            results["evaluated_pairs"] += 1
            results["total_cost"] = self.total_cost()

            # Save intermediate results
            if (i + 1) % 10 == 0:
                with open(log_output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"Saved intermediate results after {i+1} pairs")
                print(f"Current total cost: ${results['total_cost']:.4f}")

        # Save final results
        with open(log_output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Evaluation completed. Results saved to {log_output_file}")
        print(f"Total cost: ${results['total_cost']:.4f}")
        print(f"Logs saved to: {self.log_path}")
        return results

    def evaluate_all_pairs_multithread(
        self,
        dataset_path: str = "VLM-SubtleBench",
        output_file: str = "evaluation_results.json",
        max_pairs: int = None,
        max_workers: int = None,
        category: Optional[str] = None,
        domain: Optional[str] = None,
        split: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate all captioned image pairs using multithreading and save results to JSON file."""
        split_file = split if split else "test"
        qa_path = os.path.join(dataset_path, "data", f"{split_file}.jsonl")
        items = self.data_loader.load_items(
            qa_path, category=category, domain=domain, has_caption=True, split=split
        )

        if max_pairs:
            items = items[:max_pairs]

        log_output_file = os.path.join(self.logger.log_path, output_file)

        print(f"Found {len(items)} captioned image pairs to evaluate")
        print(f"Using model: {self.model_name}")
        print(f"Using prompt type: {self.prompt_type}")
        print(f"Using structured output: {self.use_structured_output}")
        print(f"Max workers: {max_workers or self.max_workers}")
        print(f"Logs will be saved to: {self.log_path}")
        print(f"Results will be saved to: {log_output_file}")

        self.refresh()

        # Prepare tasks for multithreading
        tasks = []
        pair_map = {}
        skipped_items = []

        for i, item in enumerate(items):
            prepared = self._prepare_ff_task(item, dataset_path)
            if isinstance(prepared, dict) and prepared.get("skip"):
                skipped_items.append(prepared)
                continue
            if prepared is None:
                continue

            task_id, call_llm_kwargs, pair_info = prepared
            tasks.append({
                "task_id": task_id,
                "call_llm_kwargs": call_llm_kwargs,
            })
            pair_map[task_id] = pair_info

        print(f"Prepared {len(tasks)} tasks for processing")

        # Track progress
        processed_count = 0
        results_list = []

        def on_complete_callback(task_id: str, result: Dict[str, Any]):
            """Callback function called when each LLM task completes."""
            nonlocal processed_count, results_list

            pair = pair_map.get(task_id)
            if not pair:
                print(f"Warning: No pair found for task_id {task_id}")
                return

            processed_count += 1

            if result.get("success", False):
                response_content = result["result"]
                differences = self._parse_ff_response(
                    response_content, self.use_structured_output
                )

                pair_result = {
                    "base_name": pair["base_name"],
                    "first_image": pair["first_image"],
                    "second_image": pair["second_image"],
                    "caption": pair["caption"],
                    "differences": differences,
                    "num_differences": (
                        len(differences) if isinstance(differences, list) else 1
                    ),
                    "content": response_content,
                    "cost": result.get("cost", 0.0),
                }

                results_list.append(pair_result)

                print(
                    f"  Processed {processed_count}/{len(tasks)}: {task_id} - {len(differences) if isinstance(differences, list) else 1} differences found"
                )

            else:
                pair_result = {
                    "base_name": pair["base_name"],
                    "first_image": pair["first_image"],
                    "second_image": pair["second_image"],
                    "caption": pair["caption"],
                    "differences": [],
                    "num_differences": 0,
                    "content": "",
                    "error": result.get("error", "Unknown error"),
                    "cost": 0.0,
                }

                results_list.append(pair_result)
                print(
                    f"  Failed {processed_count}/{len(tasks)}: {task_id} - Error: {result.get('error', 'Unknown error')}"
                )

        # Execute tasks using multithreading with callback
        results = self.call_llm_multithread(
            tasks=tasks,
            max_workers=max_workers,
            on_complete_callback=on_complete_callback,
        )

        return self._finalize_ff_results(
            results_list, items, log_output_file,
            extra_metadata={
                "max_workers": max_workers or self.max_workers,
                "skipped_items": skipped_items,
            },
        )
