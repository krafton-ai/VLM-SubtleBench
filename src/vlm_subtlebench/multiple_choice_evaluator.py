import json
import os
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from vlm_subtlebench.base_agent import BaseAgent
from vlm_subtlebench.llms.openai_utils import Logger
from vlm_subtlebench.llms.typing import Message
from vlm_subtlebench.prompts.multiple_choice_evaluator import (
    create_multiple_choice_messages,
)
from vlm_subtlebench.dataset_adapters import (
    UnifiedAdapter,
    DatasetLoader,
    ResponseParser,
)


class MultipleChoiceEvaluator(BaseAgent):
    @dataclass
    class Config(BaseAgent.Config):
        # inherited from BaseAgent
        prompt_type: str = "standard"

    cfg: Config  # add this to every subclass to enable static type checking

    def configure(self, *args, **kwargs) -> None:
        super().configure(*args, **kwargs)

        self.prompt_type = self.cfg.prompt_type
        # Initialize dataset utilities
        self.adapter = UnifiedAdapter()
        self.data_loader = DatasetLoader()
        self.response_parser = ResponseParser()

    def create_multiple_choice_prompt(
        self, question_data: Dict[str, Any], dataset_path: str
    ) -> List[Message]:
        """Create messages for multiple choice question with images."""

        # Process the item using the unified adapter
        processed = self.adapter.process_item(
            question_data, dataset_path, task_type="multiple_choice"
        )
        if not processed:
            return None, None, None, None

        # Check if image files exist
        if not os.path.isfile(processed.first_image_path) or not os.path.isfile(
            processed.second_image_path
        ):
            return None, None, None, None

        # Format options as A, B, C, D, E
        options_text = "\n".join(
            [f"{chr(65+i)}. {option}" for i, option in enumerate(processed.options)]
        )

        # Create messages using the centralized function
        messages = create_multiple_choice_messages(
            first_image_path=processed.first_image_path,
            second_image_path=processed.second_image_path,
            options_text=options_text,
            question_text=processed.question_text,
            prompt_type=self.prompt_type,
        )

        return (
            messages,
            processed.options,
            processed.correct_index,
            processed.question_text,
        )

    def evaluate_single_question(
        self, question_data: Dict[str, Any], dataset_path: str
    ) -> Dict[str, Any]:
        """Evaluate a single multiple choice question."""
        messages, options, correct_index, question_text = (
            self.create_multiple_choice_prompt(question_data, dataset_path)
        )

        # Get question ID and correct answer from adapter
        processed = self.adapter.process_item(
            question_data, dataset_path, task_type="multiple_choice"
        )
        if processed:
            question_id = processed.item_id
            correct_answer = processed.correct_answer
        else:
            question_id = "unknown"
            correct_answer = "Unknown"

        try:
            # Use BaseAgent's call_llm method which includes logging
            response_content = self.call_llm(
                messages=messages,
                log_path=f"mc_question_{question_id}",
            )

            # Extract the predicted answer
            predicted_answer = self.response_parser.extract_multiple_choice_answer(
                response_content
            )

            # Check if prediction is correct
            is_correct = predicted_answer == chr(65 + correct_index)

            # Get the predicted answer text
            predicted_answer_text = None
            if predicted_answer and len(predicted_answer) == 1:
                predicted_index = ord(predicted_answer) - 65  # Convert A=0, B=1, etc.
                if 0 <= predicted_index < len(options):
                    predicted_answer_text = options[predicted_index]

            return {
                "question_id": question_id,
                "question_text": question_text,
                "options": options,
                "negatives": [opt for opt in options if opt != correct_answer],
                "correct_answer": correct_answer,
                "correct_letter": chr(65 + correct_index),
                "predicted_answer": predicted_answer,
                "predicted_answer_text": predicted_answer_text,
                "is_correct": is_correct,
                "response_content": response_content,
            }
        except Exception as e:
            return {"question_id": question_id, "error": str(e)}

    # --- Shared helpers for sequential / multithread evaluate ---

    def _load_and_sample_items(
        self,
        dataset_path: str,
        max_questions: int = None,
        random_seed: int = 0,
        category: Optional[str] = None,
        domain: Optional[str] = None,
        split: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Load dataset items and optionally subsample."""
        qa_path = os.path.join(dataset_path, "qa.json")
        vqa_items = self.data_loader.load_items(
            qa_path, category=category, domain=domain, split=split
        )
        if max_questions:
            random.seed(random_seed)
            total_items = len(vqa_items)
            sample_size = min(max_questions, total_items)
            selected_indices = sorted(random.sample(range(total_items), sample_size))
            vqa_items = [vqa_items[i] for i in selected_indices]
        return vqa_items

    def _prepare_mc_task(
        self, vqa_item: Dict[str, Any], dataset_path: str, index: int
    ):
        """Prepare a single MC task.

        Returns (messages, options, correct_index, question_text, question_id, correct_answer)
        or None if the item cannot be processed.
        """
        messages, options, correct_index, question_text = (
            self.create_multiple_choice_prompt(vqa_item, dataset_path)
        )
        if messages is None:
            return None

        processed = self.adapter.process_item(
            vqa_item, dataset_path, task_type="multiple_choice"
        )
        if processed:
            question_id = processed.item_id
            correct_answer = processed.correct_answer
        else:
            question_id = f"unknown_{index}"
            correct_answer = "Unknown"

        return messages, options, correct_index, question_text, question_id, correct_answer

    def _parse_mc_result(
        self,
        response_content: str,
        options: List[str],
        correct_index: int,
        correct_answer: str,
        question_id: str,
        question_text: str,
        cost: float = None,
    ) -> Dict[str, Any]:
        """Parse an MC LLM response into a result dict."""
        predicted_letter = self.response_parser.extract_multiple_choice_answer(
            response_content
        )
        predicted_index = ord(predicted_letter) - ord("A") if predicted_letter else -1
        predicted_answer_text = (
            options[predicted_index]
            if 0 <= predicted_index < len(options)
            else None
        )
        is_correct = predicted_index == correct_index

        result = {
            "question_id": question_id,
            "question_text": question_text,
            "options": options,
            "negatives": [opt for opt in options if opt != correct_answer],
            "correct_answer": correct_answer,
            "correct_letter": chr(65 + correct_index),
            "predicted_answer": predicted_letter,
            "predicted_answer_text": predicted_answer_text,
            "is_correct": is_correct,
        }
        if cost is not None:
            result["cost"] = cost
        return result

    def _finalize_mc_results(
        self,
        results_list: List[Dict[str, Any]],
        correct_count: int,
        vqa_items: List[Dict[str, Any]],
        log_output_file: str,
        extra_metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Build summary dict, print stats, save JSON, return results."""
        processed_results = {
            "model": self.model_name,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "total_questions": len(vqa_items),
            "evaluation_time": datetime.now().isoformat(),
            "questions": results_list,
        }
        if extra_metadata:
            processed_results.update(extra_metadata)

        accuracy = correct_count / len(vqa_items) if vqa_items else 0
        total_cost = self.total_cost()

        processed_results["correct_count"] = correct_count
        processed_results["accuracy"] = accuracy
        processed_results["total_cost_usd"] = total_cost
        processed_results["cost_per_question"] = (
            total_cost / len(vqa_items) if vqa_items else 0
        )

        print(f"\nEvaluation completed!")
        print(f"Total questions: {len(vqa_items)}")
        print(f"Correct answers: {correct_count}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Total cost: ${total_cost:.6f}")
        print(
            f"Cost per question: ${total_cost / len(vqa_items):.6f}"
            if vqa_items
            else "Cost per question: $0.000000"
        )

        with open(log_output_file, "w", encoding="utf-8") as f:
            json.dump(processed_results, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {log_output_file}")
        return processed_results

    # --- Public evaluation methods ---

    def evaluate_all_questions(
        self,
        dataset_path: str = "VLM-SubtleBench",
        output_file: str = "mc_evaluation_results.json",
        max_questions: int = None,
        random_seed: int = 0,
        category: Optional[str] = None,
        domain: Optional[str] = None,
        split: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate all multiple choice questions sequentially and save results."""
        vqa_items = self._load_and_sample_items(
            dataset_path, max_questions, random_seed, category, domain, split
        )

        log_output_file = os.path.join(self.logger.log_path, output_file)

        print(f"Found {len(vqa_items)} questions to evaluate")
        print(f"Using model: {self.model_name}")
        print(f"Logs will be saved to: {self.log_path}")
        print(f"Results will be saved to: {log_output_file}")

        self.refresh()

        correct_count = 0
        results_list = []

        for i, vqa_item in enumerate(vqa_items):
            prepared = self._prepare_mc_task(vqa_item, dataset_path, i)
            if prepared is None:
                continue

            messages, options, correct_index, question_text, question_id, correct_answer = prepared

            try:
                response = self.call_llm(
                    messages=messages,
                    log_path=f"mc_question_{question_id}",
                    response_format=None,
                )

                question_result = self._parse_mc_result(
                    response, options, correct_index, correct_answer,
                    question_id, question_text, cost=self.total_cost(),
                )

                if question_result["is_correct"]:
                    correct_count += 1

                results_list.append(question_result)

                current_accuracy = correct_count / (i + 1)
                print(
                    f"  Processed {i + 1}/{len(vqa_items)}: {question_id} - {'✓' if question_result['is_correct'] else '✗'} (Accuracy: {current_accuracy:.3f})"
                )

            except Exception as e:
                print(
                    f"  Failed {i + 1}/{len(vqa_items)}: {question_id} - Error: {str(e)}"
                )
                question_result = {
                    "question_id": question_id,
                    "question_text": question_text,
                    "options": options,
                    "negatives": [opt for opt in options if opt != correct_answer],
                    "correct_answer": correct_answer,
                    "correct_letter": chr(65 + correct_index),
                    "predicted_answer": None,
                    "predicted_answer_text": None,
                    "is_correct": False,
                    "error": str(e),
                    "cost": 0.0,
                }
                results_list.append(question_result)

        return self._finalize_mc_results(
            results_list, correct_count, vqa_items, log_output_file
        )

    def evaluate_all_questions_multithread(
        self,
        dataset_path: str = "VLM-SubtleBench",
        output_file: str = "mc_evaluation_results.json",
        max_questions: int = None,
        max_workers: int = None,
        random_seed: int = 0,
        category: Optional[str] = None,
        domain: Optional[str] = None,
        split: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate all multiple choice questions using multithreading and save results."""
        vqa_items = self._load_and_sample_items(
            dataset_path, max_questions, random_seed, category, domain, split
        )

        log_output_file = os.path.join(self.logger.log_path, output_file)

        print(f"Found {len(vqa_items)} questions to evaluate")
        print(f"Using model: {self.model_name}")
        print(f"Max workers: {max_workers or self.max_workers}")
        print(f"Logs will be saved to: {self.log_path}")
        print(f"Results will be saved to: {log_output_file}")

        self.refresh()

        # Prepare tasks for multithreading
        tasks = []
        item_map = {}  # Map task_id to item data for callback
        for i, vqa_item in enumerate(vqa_items):
            prepared = self._prepare_mc_task(vqa_item, dataset_path, i)
            if prepared is None:
                continue

            messages, options, correct_index, question_text, question_id, correct_answer = prepared

            vqa_item["correct_index"] = correct_index
            vqa_item["shuffled_options"] = options
            vqa_item["correct_answer"] = correct_answer
            vqa_item["question_id"] = question_id
            vqa_item["question_text"] = question_text

            task_data = {
                "task_id": question_id,
                "call_llm_kwargs": {
                    "messages": messages,
                    "log_path": f"mc_question_{question_id}",
                    "response_format": None,
                },
            }
            tasks.append(task_data)
            item_map[question_id] = vqa_item

        print(f"Prepared {len(tasks)} tasks for processing")

        # Track progress
        correct_count = 0
        processed_count = 0
        results_list = []

        def on_complete_callback(task_id: str, result: Dict[str, Any]):
            """Callback function called when each LLM task completes."""
            nonlocal correct_count, processed_count, results_list

            vqa_item = item_map.get(task_id)
            if not vqa_item:
                print(f"Warning: No item found for task_id {task_id}")
                return

            processed_count += 1

            if result.get("success", False):
                response_content = result["result"]
                question_result = self._parse_mc_result(
                    response_content,
                    vqa_item.get("shuffled_options", []),
                    vqa_item["correct_index"],
                    vqa_item["correct_answer"],
                    task_id,
                    vqa_item.get("question_text", ""),
                    cost=result.get("cost", 0.0),
                )
                question_result["response_content"] = response_content

                if question_result["is_correct"]:
                    correct_count += 1

                results_list.append(question_result)

                current_accuracy = (
                    correct_count / processed_count if processed_count > 0 else 0
                )
                print(
                    f"  Processed {processed_count}/{len(vqa_items)}: {task_id} - {'✓' if question_result['is_correct'] else '✗'} (Accuracy: {current_accuracy:.3f})"
                )
            else:
                # Handle failed task
                question_result = {
                    "question_id": task_id,
                    "question_text": vqa_item.get("question_text", ""),
                    "options": vqa_item.get("shuffled_options", []),
                    "negatives": [
                        opt
                        for opt in vqa_item.get("shuffled_options", [])
                        if opt != vqa_item["correct_answer"]
                    ],
                    "correct_answer": vqa_item["correct_answer"],
                    "correct_letter": chr(65 + vqa_item["correct_index"]),
                    "predicted_answer": None,
                    "predicted_answer_text": None,
                    "is_correct": False,
                    "error": result.get("error", "Unknown error"),
                    "cost": 0.0,
                }

                results_list.append(question_result)
                print(
                    f"  Failed {processed_count}/{len(vqa_items)}: {task_id} - Error: {result.get('error', 'Unknown error')}"
                )

        # Execute tasks using multithreading with callback
        results = self.call_llm_multithread(
            tasks=tasks,
            max_workers=max_workers,
            on_complete_callback=on_complete_callback,
        )

        return self._finalize_mc_results(
            results_list, correct_count, vqa_items, log_output_file,
            extra_metadata={"max_workers": max_workers or self.max_workers},
        )
