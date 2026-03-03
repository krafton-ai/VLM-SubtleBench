import os
import logging
import argparse
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

from vlm_subtlebench.free_form_evaluator import FreeFormEvaluator

logger = logging.getLogger(__name__)


def parse_configs():
    """Parse configuration from YAML file and command line arguments."""
    # Define argparse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/free_form_evaluation.yaml",
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
    if cfg.data.mode == "pair":
        path = Path(cfg.data.first_image)
        data_name = path.stem
    else:
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
        "image_comparison_free_form_evaluator",
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

    return cfg


def evaluate_custom_pair(evaluator, first_image_path: str, second_image_path: str):
    """Evaluate a specific image pair and return detailed results."""
    print(f"\nEvaluating image pair:")
    print(f"First image: {first_image_path}")
    print(f"Second image: {second_image_path}")

    # Check if files exist
    if not os.path.exists(first_image_path):
        print(f"Error: First image not found: {first_image_path}")
        return None
    if not os.path.exists(second_image_path):
        print(f"Error: Second image not found: {second_image_path}")
        return None

    # Evaluate the pair
    differences = evaluator.evaluate_single_pair(first_image_path, second_image_path)

    # Print results
    if isinstance(differences, list):
        print(f"\nFound {len(differences)} differences:")
        for i, difference in enumerate(differences, 1):
            print(f"{i}. {difference}")
        num_differences = len(differences)
    else:
        print(f"\nResponse: {differences}")
        num_differences = 1

    return {
        "first_image": first_image_path,
        "second_image": second_image_path,
        "num_differences": num_differences,
        "differences": differences,
    }


def evaluate_dataset(
    evaluator,
    dataset_path,
    max_pairs: int = None,
    use_multithreading: bool = False,
    max_workers: int = None,
    category: str = None,
    domain: str = None,
    split: str = None,
):
    """Evaluate all captioned pairs in the dataset."""
    print(f"\nEvaluating dataset...")

    # Choose evaluation method based on multithreading setting
    if use_multithreading:
        results = evaluator.evaluate_all_pairs_multithread(
            dataset_path=dataset_path,
            max_pairs=max_pairs,
            max_workers=max_workers,
            category=category,
            domain=domain,
            split=split,
        )
    else:
        results = evaluator.evaluate_all_pairs(
            dataset_path=dataset_path,
            max_pairs=max_pairs,
            category=category,
            domain=domain,
            split=split,
        )

    # Print summary
    total_differences = sum(result["num_differences"] for result in results["results"])
    avg_differences = (
        total_differences / len(results["results"]) if results["results"] else 0
    )

    print(f"\nEvaluation Summary:")
    print(f"Total pairs evaluated: {results['evaluated_pairs']}")
    print(f"Total differences found: {total_differences}")
    print(f"Average differences per pair: {avg_differences:.2f}")
    print(f"Total cost: ${results['total_cost']:.4f}")
    print(f"Multithreading: {use_multithreading}")
    if use_multithreading:
        print(f"Max workers: {max_workers or evaluator.max_workers}")

    return results


def main():
    # Parse configuration
    config = parse_configs()
    logger.info(f"Configuration: {_redact_sensitive(config)}")

    # Initialize evaluator
    evaluator = FreeFormEvaluator(config.model)

    if config.data.mode == "pair":
        # Evaluate specific image pair
        result = evaluate_custom_pair(
            evaluator, config.data.first_image, config.data.second_image
        )

        if result:
            print(f"\nEvaluation Summary:")
            print(f"Total differences found: {result['num_differences']}")
            print(f"Evaluation completed successfully!")

    elif config.data.mode == "dataset":
        # Evaluate dataset
        use_multithreading = getattr(config.model, "use_multithreading", False)
        max_workers = getattr(config.model, "max_workers", None)
        category = getattr(config.data, "category", None)
        domain = getattr(config.data, "domain", None)
        split = getattr(config.data, "split", "test")

        evaluate_dataset(
            evaluator,
            dataset_path=config.data.dataset_path,
            max_pairs=config.data.max_pairs,
            use_multithreading=use_multithreading,
            max_workers=max_workers,
            category=category,
            domain=domain,
            split=split,
        )


if __name__ == "__main__":
    main()
