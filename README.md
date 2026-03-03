# VLM-SubtleBench

A benchmarking framework for evaluating Vision Language Models (VLMs) on detecting subtle differences between image pairs. Supports multiple-choice and free-form evaluation paradigms with multiple LLM backends.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Dataset](#dataset)
- [API Keys](#api-keys)
- [Running Evaluations](#running-evaluations)
  - [Multiple-Choice Evaluation](#multiple-choice-evaluation)
  - [Free-Form Evaluation](#free-form-evaluation)
- [Using a Local Model](#using-a-local-model)
- [Configuration Reference](#configuration-reference)
- [Supported Models](#supported-models)
- [Results and Logs](#results-and-logs)

## Environment Setup

Requires **Python >= 3.8**.

```bash
# Install core dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .

# Install optional dependencies as needed
pip install anthropic          # Anthropic Claude API
pip install google-genai       # Google Gemini API
pip install transformers torch # Local model support
pip install matplotlib seaborn pandas  # Analysis/visualization
```

## Dataset

Download the dataset from [Hugging Face](https://huggingface.co/datasets/KRAFTON/VLM-SubtleBench).

By default, the code expects the dataset at `VLM-SubtleBench/` in the project root. You can either:

- Place (or symlink) the downloaded dataset as `VLM-SubtleBench/` in the project root, or
- Override the path via CLI:

```bash
python scripts/evaluate_multiple_choice.py data.dataset_path="/path/to/your/dataset"
```

### Filtering

Items can be filtered by:

- **Split**: `test` (default), `val`, or `null` (all)
- **Category**: `action`, `attribute`, `emotion`, `existence`, `quality`, `quantity`, `spatial`, `state`, `temporal`, `viewpoint`
- **Domain**: `natural`, `industrial`, `medical`, `aerial`, `synthetic`

Free-form evaluation uses only items where `has_caption == true`.

## API Keys

API keys are loaded from files in the `keys/` directory. Create the following structure and add your keys:

```
keys/
├── openai-key/
│   └── key.env           # Line 1: API key, Line 2 (optional): org key
├── anthropic-key/
│   └── key.env           # Single line: API key
├── google-key/
│   └── gemini_gcp.json   # GCP service account JSON
└── openrouter-key/
    └── key.env           # Single line: API key
```

You only need to set up keys for the backends you plan to use. For example, if you only use GPT-4o, you only need `keys/openai-key/key.env`.

**Example — setting up an OpenAI key:**

```bash
mkdir -p keys/openai-key
echo "sk-your-api-key-here" > keys/openai-key/key.env
```

## Running Evaluations

Configuration uses YAML files in `configs/` with CLI overrides via `key.subkey=value` syntax. All options shown below are optional and have sensible defaults.

### Multiple-Choice Evaluation

```bash
python scripts/evaluate_multiple_choice.py \
  model.llm_name="gpt-4o" \                    # Model to evaluate (default: gpt-4o)
  model.prompt_type="standard" \                # Prompt template (default: standard)
  model.use_multithreading=true \               # Enable concurrent API calls (default: true)
  model.max_workers=8 \                         # Number of threads (default: 8)
  data.max_questions=100 \                      # Limit number of questions, null=all (default: null)
  data.split="test" \                           # Data split: "test", "val", or null for all (default: test)
  data.category="attribute" \                   # Filter by category, null=all (default: null)
  data.domain="natural"                         # Filter by domain, null=all (default: null)
```

### Free-Form Evaluation

**Dataset mode** — evaluates all captioned items:

```bash
python scripts/evaluate_free_form.py \
  data.mode="dataset" \                         # Evaluation mode (default: dataset)
  model.llm_name="gpt-4o" \                    # Model to evaluate (default: gpt-4o)
  model.use_multithreading=true \               # Enable concurrent API calls (default: true)
  model.max_workers=8 \                         # Number of threads (default: 8)
  data.max_pairs=50 \                           # Limit number of pairs, null=all (default: null)
  data.split="test" \                           # Data split: "test", "val", or null for all (default: test)
  data.category="state" \                       # Filter by category, null=all (default: null)
  data.domain="natural"                         # Filter by domain, null=all (default: null)
```

**Pair mode** — evaluate a specific image pair:

```bash
python scripts/evaluate_free_form.py \
  data.mode="pair" \
  data.first_image="path/to/image1.png" \
  data.second_image="path/to/image2.png"
```

## Using a Local Model

You can use any model served via an **OpenAI-compatible API**. This works with serving frameworks such as:

- [vLLM](https://docs.vllm.ai/) (`python -m vllm.entrypoints.openai.api_server`)
- [Ollama](https://ollama.com/) (`ollama serve`)
- [LMStudio](https://lmstudio.ai/)
- [llama.cpp server](https://github.com/ggerganov/llama.cpp)

### Step 1: Serve your model

```bash
# Example with vLLM
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-11B-Vision-Instruct \
  --port 8000

# Example with Ollama
ollama serve   # serves on port 11434 by default
```

### Step 2: Run evaluation pointing to your server

Provide `model.api_key` and `model.api_base_url` via CLI overrides. The model name must **not** match any known cloud model prefix (e.g., `gpt`, `claude`, `gemini`) so it routes to the local backend.

```bash
python scripts/evaluate_multiple_choice.py \
  model.llm_name="meta-llama/Llama-3.2-11B-Vision-Instruct" \
  model.api_key="dummy" \
  model.api_base_url="http://localhost:8000/v1" \
  data.max_questions=10
```

For Ollama:

```bash
python scripts/evaluate_multiple_choice.py \
  model.llm_name="llama3.2-vision" \
  model.api_key="ollama" \
  model.api_base_url="http://localhost:11434/v1"
```

> **Note:** The local backend requires `transformers` (and optionally `torch`) for tokenization. Install them with `pip install transformers torch`.


### Available Prompt Types

| Type | Description |
|------|-------------|
| `standard` | Default two-image comparison prompt |
| `no_reasoning` | Direct answer without chain-of-thought |
| `camera_augmented` | Adds camera/viewpoint context |
| `concatenated` | Horizontally concatenates image pair |
| `grid` | Arranges images in a grid layout |
| `overlapped` | Overlays images for comparison |
| `substract` | Shows pixel difference between images |

## Supported Models

Model routing is determined by substring matching on the model name:

| Model Name Pattern | Backend | Example |
|---|---|---|
| `gpt-4*`, `gpt-5`, `o1*`, `o3*`, `o4*` | OpenAI | `gpt-4o`, `o3`, `gpt-5` |
| `claude*` | OpenRouter | `anthropic/claude-sonnet-4` |
| `gemini*` | Google Gemini | `gemini-2.5-flash`, `gemini-2.5-pro` |
| `llava*` | vLLM Server | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` |
| `qwen*`, `internvl*` | OpenRouter | `qwen/qwen2.5-vl-72b-instruct` |
| *(anything else)* | Local (OpenAI-compatible) | Any locally served model |

## Results and Logs

Results are saved to:

```
logs/<evaluator_type>/<model>/<prompt_type>/<dataset>/<timestamp>/
├── run.log                         # Execution log
└── mc_evaluation_results.json      # Predictions, accuracy, costs
```
