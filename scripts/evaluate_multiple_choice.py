import os
import logging
import argparse
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

from vlm_subtlebench.multiple_choice_evaluator import MultipleChoiceEvaluator

logger = logging.getLogger(__name__)


def parse_configs():
    """Parse configuration from YAML file and command line arguments."""
    # Define argparse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/multiple_choice_evaluation.yaml",
        help="Path to configuration YAML file",
    )
    args, unknown = parser.parse_known_args()

    # Load configuration file
    cfg = OmegaConf.load(args.config)

    # Override with command-line arguments
    cli_cfg = OmegaConf.from_cli(unknown)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Set logging
    cfg = set_log_path(cfg)

    return cfg


def _redact_sensitive(cfg):
    """Return a copy of config with sensitive keys redacted for safe logging."""
    from omegaconf import OmegaConf
    sensitive_keys = {"api_key", "api_base_url", "secret", "password", "token"}
    container = OmegaConf.to_container(cfg, resolve=True)

    def redact_dict(d):
        if not isinstance(d, dict):
            return d
        out = {}
        for k, v in d.items():
            key_lower = k.lower() if isinstance(k, str) else ""
            if any(s in key_lower for s in sensitive_keys) and v:
                out[k] = "[REDACTED]"
            else:
                out[k] = redact_dict(v) if isinstance(v, dict) else v
        return out

    return redact_dict(container)


def set_log_path(cfg):
    # Build data name from dataset_path + filters
    dataset_name = Path(cfg.data.dataset_path).name
    category = getattr(cfg.data, "category", None)
    domain = getattr(cfg.data, "domain", None)

    parts = [dataset_name]
    if category:
        parts.append(category)
    if domain:
        parts.append(domain)
    data_name = "_".join(parts)

    log_path = os.path.join(
        cfg.log_path,
        "multiple_choice_evaluator",
        cfg.model.llm_name,
        cfg.model.prompt_type,
        data_name,
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    cfg.log_path = log_path
    cfg.model.log_path = log_path

    os.makedirs(log_path, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_path, "run.log")),
            logging.StreamHandler(),
        ],
    )

    # Disable verbose HTTP request logs from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return cfg


def main():
    # Parse configuration
    config = parse_configs()
    logger.info(f"Configuration: {_redact_sensitive(config)}")

    # Initialize evaluator
    evaluator = MultipleChoiceEvaluator(config.model)

    # Get filter params (None if not set)
    category = getattr(config.data, "category", None)
    domain = getattr(config.data, "domain", None)
    split = getattr(config.data, "split", "test")

    # Run evaluation
    if config.model.use_multithreading:
        results = evaluator.evaluate_all_questions_multithread(
            dataset_path=config.data.dataset_path,
            max_questions=config.data.max_questions,
            random_seed=config.data.random_seed,
            category=category,
            domain=domain,
            split=split,
        )
    else:
        results = evaluator.evaluate_all_questions(
            dataset_path=config.data.dataset_path,
            max_questions=config.data.max_questions,
            random_seed=config.data.random_seed,
            category=category,
            domain=domain,
            split=split,
        )

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Model: {results['model']}")
    print(f"Temperature: {results['temperature']}")
    print(f"Total Questions (loaded): {results.get('total_questions_loaded', results.get('total_questions', 'N/A'))}")
    print(f"Total Questions (evaluated): {results.get('total_questions_evaluated', 'N/A')}")
    if results.get("skipped_items"):
        print(f"Skipped (e.g. missing images): {len(results['skipped_items'])}")
    print(f"Max Questions: {config.data.max_questions}")
    print(f"Random Seed: {config.data.random_seed}")
    print(f"Split: {split}")
    print(f"Category: {category}")
    print(f"Domain: {domain}")
    print(f"Correct Answers: {results['correct_count']}")
    print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Total Cost: ${results['total_cost_usd']:.6f}")
    print(f"Cost per Question: ${results['cost_per_question']:.6f}")
    print(f"Multithreading: {config.model.use_multithreading}")
    print(f"Max Workers: {config.model.max_workers}")
    print("=" * 50)


if __name__ == "__main__":
    main()
