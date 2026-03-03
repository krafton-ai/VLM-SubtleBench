"""Base agent class for LLM-based evaluators."""

from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Callable, Optional, Union
from dataclasses import dataclass
import os

from vlm_subtlebench.utils import Configurable
from vlm_subtlebench.llms.llm import load_model
from vlm_subtlebench.json_schemas import SCHEMA_REGISTRY
from vlm_subtlebench.llms.openai_utils import (
    MoneyManager,
    pretty_print_conversation,
    Logger,
)
from omegaconf import DictConfig


class BaseAgent(Configurable):
    """Base class for evaluators. Handles model lifecycle, multithreading, and cost tracking."""

    @dataclass
    class Config:
        llm_name: str
        temperature: float = 0.0
        repetition_penalty: float = 0.0
        api_key: str = ""
        api_base_url: str = ""

        fine_tuned_model: str | None = None
        is_print: bool = False
        enable_lora: bool = False

        use_multithreading: bool = True
        max_workers: int = 8
        log_path: str = "./logs"
        debug_mode: bool = False

    cfg: Config

    def configure(self, *args, **kwargs) -> None:
        self.orig_model = self.cfg.llm_name
        self.temperature = self.cfg.temperature
        self.repetition_penalty = self.cfg.repetition_penalty
        self.api_key = self.cfg.api_key
        self.api_base_url = self.cfg.api_base_url

        self.fine_tuned_model = self.cfg.fine_tuned_model
        self.enable_lora = self.cfg.enable_lora
        self.is_print = self.cfg.is_print

        self.max_workers = self.cfg.max_workers
        self.log_path = self.cfg.log_path
        self.debug_mode = self.cfg.debug_mode

        self._setup_model()
        self._setup_logger()

    def _setup_model(self):
        loaded_model = load_model(
            self.orig_model,
            fine_tuned_model=self.fine_tuned_model,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            api_key=self.api_key or None,
            api_base_url=self.api_base_url or None,
        )

        self.model_name = loaded_model["model_name"]
        self.ctx_manager: MoneyManager = loaded_model["ctx_manager"]
        self.llm = loaded_model["llm"]
        self.tokenizer = loaded_model["tokenizer"]

    def _setup_logger(self):
        self.logger = Logger(log_path=self.log_path)

    def update_parameters(
        self,
        temperature: float | None = None,
        repetition_penalty: float | None = None,
    ) -> None:
        if temperature is not None:
            self.temperature = temperature
        if repetition_penalty is not None:
            self.repetition_penalty = repetition_penalty
        self._setup_model()

    def total_cost(self) -> float:
        return self.ctx_manager.total_cost

    def refresh(self) -> None:
        return self.ctx_manager.refresh()

    def update_model(
        self, new_model: str, new_fine_tuned_model: str | None = None
    ) -> None:
        self.orig_model = new_model
        self.fine_tuned_model = new_fine_tuned_model
        self._setup_model()

    def logger(self, messages, log_path=None):
        """Log messages. Override in subclasses for custom logging."""
        pass

    def call_llm(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        log_path: str | None = None,
        messages: list[dict[str, str]] | None = None,
        **kwargs,
    ):
        """Call the LLM with either pre-built messages or system/user prompts."""
        if not messages:
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            if user_prompt:
                messages.append(
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                )

        if self.enable_lora:
            output = self.llm(
                messages,
                lora=self.lora,
                **kwargs,
            )
        else:
            output = self.llm(
                messages,
                **kwargs,
            )

        messages.append(
            {
                "content": output["response"].choices[0].message.content,
                "role": output["response"].choices[0].message.role,
            }
        )
        if self.is_print:
            pretty_print_conversation(messages)
        self.logger(messages, log_path)
        pred = messages[-1]["content"]
        return pred

    def _worker_function_thread(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Worker function for multithreaded execution."""
        try:
            task_id = task_data.get("task_id", "unknown")
            call_llm_kwargs = task_data.get("call_llm_kwargs", {})

            result = self.call_llm(**call_llm_kwargs)

            return {
                "task_id": task_id,
                "success": True,
                "result": result,
                "error": None
            }
        except Exception as e:
            return {
                "task_id": task_id,
                "success": False,
                "result": None,
                "error": str(e)
            }

    def call_llm_multithread(
        self,
        tasks: List[Dict[str, Any]],
        max_workers: int = None,
        use_multithreading: bool = True,
        on_complete_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ) -> List[Dict[str, Any]]:
        """Call LLM for multiple tasks using multithreading.

        Threads share memory and MoneyManager.

        Args:
            tasks: List of task dicts, each with 'task_id' and 'call_llm_kwargs'.
            max_workers: Number of worker threads (defaults to self.max_workers).
            use_multithreading: Whether to use multithreading.
            on_complete_callback: Called when each task completes.
                Signature: callback(task_id, result).

        Returns:
            List of result dicts with 'task_id', 'success', 'result', 'cost', 'error'.
        """
        if not use_multithreading or len(tasks) == 1:
            results = []
            for task_data in tasks:
                result = self._worker_function_single_thread(task_data)
                results.append(result)
                if on_complete_callback:
                    task_id = task_data.get("task_id", "unknown")
                    on_complete_callback(task_id, result)
            return results

        max_workers = max_workers or self.max_workers
        max_workers = min(max_workers, len(tasks))

        print(f"Using {max_workers} worker threads for {len(tasks)} tasks")

        results = []

        def worker_with_callback(task_data):
            result = self._worker_function_thread(task_data)
            if on_complete_callback:
                task_id = task_data.get("task_id", "unknown")
                on_complete_callback(task_id, result)
            return result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(worker_with_callback, tasks))

        return results

    def _worker_function_single_thread(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Worker function for single-threaded execution with cost measurement."""
        try:
            task_id = task_data.get("task_id", "unknown")
            call_llm_kwargs = task_data.get("call_llm_kwargs", {})

            cost_before = self.total_cost()
            result = self.call_llm(**call_llm_kwargs)
            cost_after = self.total_cost()
            delta_cost = cost_after - cost_before

            return {
                "task_id": task_id,
                "success": True,
                "result": result,
                "cost": delta_cost,
                "error": None
            }
        except Exception as e:
            return {
                "task_id": task_id,
                "success": False,
                "result": None,
                "cost": 0.0,
                "error": str(e)
            }
