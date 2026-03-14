# FC-Eval

<p align="center">
  <a href="https://heyneo.so" target="_blank">
    <img src="https://img.shields.io/badge/Made%20by-NEO-6366f1?style=for-the-badge&logo=robot&logoColor=white" alt="Made by NEO">
  </a>
</p>

<p align="center">
  <strong>Function Calling Evaluation Tool for LLMs</strong><br>
  <sub>Supports OpenRouter (Cloud) and Ollama (Local) backends</sub>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#usage">Usage</a> •
  <a href="#ollama-setup">Ollama Setup</a> •
  <a href="#methodology">Methodology</a>
</p>

## Overview

FC-Eval is a comprehensive CLI tool for evaluating Large Language Models' function-calling capabilities. Inspired by the Berkeley Function Calling Leaderboard (BFCL) v4 methodology, it provides rigorous testing across 30 unique test cases covering single-turn, multi-turn, and agentic scenarios.

**Key Features:**
- 🌐 **Dual Backend Support**: Evaluate models via OpenRouter (cloud) or Ollama (local)
- 📊 **30 Unique Test Cases**: Comprehensive coverage across all function-calling scenarios
- 🔄 **Best of N Trials**: Configurable trial count with reliability metrics
- ⚡ **Parallel Execution**: Multi-threaded evaluation for faster results
- 📈 **Comprehensive Reporting**: JSON and TXT reports with detailed metrics
- 🎯 **AST-Based Validation**: Accurate function call matching using abstract syntax trees

---

## Installation

### Prerequisites

- Python 3.10 or higher
- For Ollama testing: Linux/macOS/Windows with WSL

### Step 1: Clone the Repository

```bash
git clone https://github.com/gauravvij/function-calling-cli.git
cd function-calling-cli
```

### Step 2: Install Python Dependencies

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

---

## Quick Start

FC-Eval can be run in two ways:
1. **Using the installed CLI** (`fc-eval`) - Supports both OpenRouter and Ollama
2. **Using the standalone script** (`evaluate_fc.py`) - OpenRouter only

### Option A: OpenRouter (Cloud) - Easiest

1. **Get an API key** at [https://openrouter.ai/keys](https://openrouter.ai/keys)

2. **Set your API key:**
   ```bash
   export OPENROUTER_API_KEY="your-api-key-here"
   ```

3. **Run evaluation using fc-eval:**
   ```bash
   fc-eval --provider openrouter --models qwen/qwen3.5-9b
   ```

   Or using the standalone script:
   ```bash
   python evaluate_fc.py --models qwen/qwen3.5-9b
   ```

### Option B: Ollama (Local) - Requires Setup

1. **Install Ollama** (see [Ollama Setup](#ollama-setup) section)

2. **Create the optimized model:**
   ```bash
   ollama create qwen3.5:9b-fc -f qwen3.5-9b-fc.modelfile
   ```

3. **Run evaluation:**
   ```bash
   fc-eval --provider ollama --models qwen3.5:9b-fc
   ```

---

## Ollama Setup

### Installing Ollama

Ollama provides a simple installation script for Linux/macOS:

```bash
# Install Ollama (official one-liner)
curl -fsSL https://ollama.com/install.sh | sh
```

This will:
1. Download and install the Ollama binary
2. Set up the Ollama service
3. Start the Ollama server automatically

### Verifying Installation

```bash
# Check Ollama is installed
ollama --version

# Verify server is running
curl http://localhost:11434/api/tags
```

### Creating the Custom Modelfile

The project includes an optimized Modelfile (`qwen3.5-9b-fc.modelfile`) that addresses the temperature and system prompt issues identified in our analysis:

```dockerfile
FROM qwen3.5:9b

# System prompt optimized for function calling
SYSTEM You are a helpful AI assistant with access to tools/functions. When you need to perform an action, use the available tools by making function calls. Always respond with the correct function call format when a tool is needed.

# Critical parameters for function calling accuracy
PARAMETER temperature 0.0
PARAMETER top_p 0.9
PARAMETER top_k 10
PARAMETER num_ctx 8192
PARAMETER num_predict 4096
```

**Key Configuration Changes:**

| Parameter | Default | Optimized | Impact |
|-----------|---------|-----------|--------|
| `temperature` | 1.0 | 0.0 | Eliminates randomness for deterministic function calls |
| `top_p` | 0.95 | 0.9 | Slightly more focused sampling |
| `top_k` | 20 | 10 | Reduces token selection variety |
| `num_ctx` | 2048 | 8192 | Larger context window |
| `num_predict` | -1 | 4096 | Maximum response length |

### Building the Optimized Model

```bash
# Create the custom model from the Modelfile
ollama create qwen3.5:9b-fc -f qwen3.5-9b-fc.modelfile

# Verify the model was created
ollama list

# Inspect model parameters
ollama show qwen3.5:9b-fc
```

### Pulling the Base Model (if needed)

If you don't have the base model:

```bash
# Pull the base Qwen 3.5 9B model
ollama pull qwen3.5:9b

# Then create the custom version
ollama create qwen3.5:9b-fc -f qwen3.5-9b-fc.modelfile
```

---

## Usage

### API Key Setup

#### OpenRouter API Key

FC-Eval requires an OpenRouter API key for cloud-based evaluation.

**Option 1: Environment Variable (Recommended)**

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

Add this to your `~/.bashrc` or `~/.zshrc` for persistence.

**Option 2: Command Line Argument**

```bash
fc-eval --provider openrouter --api-key "your-api-key-here"
```

**Option 3: .env File**

Create a `.env` file in your working directory:

```
OPENROUTER_API_KEY=your-api-key-here
```

Get your API key at: https://openrouter.ai/keys

#### Ollama (Local)

No API key required for Ollama. Ensure the server is running:

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags
```

### Basic Usage

#### Evaluate with OpenRouter (Cloud)

```bash
# Evaluate default models via OpenRouter
fc-eval --provider openrouter

# Evaluate specific models
fc-eval --provider openrouter --models qwen/qwen3.5-9b qwen/qwen3.5-27b

# Run with parallel execution
fc-eval --provider openrouter --mode parallel --max-workers 10
```

#### Evaluate with Ollama (Local)

```bash
# Evaluate local Ollama models
fc-eval --provider ollama

# Evaluate specific local model
fc-eval --provider ollama --models qwen3.5:9b-fc

# Run with sequential mode (recommended for local testing)
fc-eval --provider ollama --mode sequential
```

### Parallel vs Sequential Execution

**Parallel Execution** (recommended for cloud):
```bash
fc-eval --provider openrouter --mode parallel --max-workers 10
```

**Sequential Execution** (recommended for local/debugging):
```bash
fc-eval --provider ollama --mode sequential
```

### Custom Models

Evaluate specific models:

```bash
# OpenRouter models
fc-eval --provider openrouter --models openai/gpt-4o anthropic/claude-3.5-sonnet

# Ollama models
fc-eval --provider ollama --models llama3.2 mistral
```

### Multiple Trials (Best of N)

Run multiple trials per test for reliability metrics (default: 3):

```bash
fc-eval --provider openrouter --trials 5
```

A test passes if at least one trial succeeds (Best of N logic). Reliability is reported as the percentage of trials that passed.

### Category Filtering

Run only specific test categories:

```bash
# Single-turn tests only
fc-eval --provider openrouter --category single_turn

# Multi-turn tests only
fc-eval --provider openrouter --category multi_turn

# Agentic tests only
fc-eval --provider openrouter --category agentic
```

### Custom Output Directory

Save reports to a custom directory:

```bash
fc-eval --provider openrouter --output-dir ./my_results
```

---

## Features

- **Dual Backend Support**: Test models via OpenRouter (cloud) or Ollama (local)
- **30 Unique Test Cases**: Comprehensive coverage across single-turn, multi-turn, and agentic scenarios
- **Best of N Trials**: Configurable trial count with reliability metrics
- **Parallel Execution**: Multi-threaded evaluation for faster results
- **Comprehensive Reporting**: JSON and TXT reports with detailed metrics
- **AST-Based Validation**: Accurate function call matching using abstract syntax trees
- **Category Breakdown**: Detailed analysis by test category and subcategory
- **Latency Tracking**: Performance metrics for each model

---

## Methodology

### Test Categories

1. **Single-Turn (16 tests)**
   - Simple function calls
   - Multiple function selection
   - Parallel function calling
   - Parallel multiple functions
   - Relevance detection

2. **Multi-Turn (8 tests)**
   - Base multi-turn conversations
   - Missing parameter handling
   - Missing function scenarios
   - Long context management

3. **Agentic (6 tests)**
   - Web search simulation
   - Memory/state management
   - Format sensitivity

### Evaluation Logic

- **Best of N**: A test passes if at least one of N trials succeeds
- **Reliability**: Percentage of trials that passed (e.g., 2/3 trials = 66.7% reliability)
- **AST Matching**: Function calls validated using abstract syntax tree comparison

---

## Troubleshooting

### Ollama Connection Issues

**Problem**: `Connection refused` error when using Ollama provider

**Solution**:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start the server
ollama serve
```

### Model Not Found (Ollama)

**Problem**: `model not found` error

**Solution**:
```bash
# List available models
ollama list

# Pull the required model
ollama pull qwen3.5:9b

# Create custom model with Modelfile
ollama create qwen3.5:9b-fc -f qwen3.5-9b-fc.modelfile
```

### OpenRouter API Errors

**Problem**: `401 Unauthorized` or `429 Rate Limited`

**Solution**:
```bash
# Verify API key is set
echo $OPENROUTER_API_KEY

# Set API key
export OPENROUTER_API_KEY="your-key-here"

# For rate limits, use sequential mode with fewer workers
fc-eval --provider openrouter --mode sequential --trials 1
```

### Low Accuracy on Local Models

**Problem**: Local Ollama models show significantly lower accuracy than OpenRouter

**Explanation**: This is expected due to:
1. **Quantization**: Ollama uses Q4_K_M (4-bit) quantization by default
2. **System Prompts**: OpenRouter may apply additional optimizations
3. **API Optimizations**: Cloud providers may use response format enforcement

**Recommendation**: Use the custom Modelfile (`qwen3.5-9b-fc.modelfile`) for best local results, but expect ~60 percentage point gap vs OpenRouter.

---

## Performance Comparison: OpenRouter vs Ollama

Based on our analysis with Qwen 3.5 9B:

| Metric | OpenRouter (Cloud) | Ollama (Local) | Difference |
|--------|-------------------|----------------|------------|
| **Accuracy** | 83.3% | 22.2% | -61.1 pp |
| **Temperature** | 0.0 (default) | 1.0 (default) | Critical |
| **Avg Latency** | ~1600ms | ~8900ms | 5.5x slower |
| **Quantization** | Unknown (likely F16) | Q4_K_M (4-bit) | Precision loss |

**Recommendation**: Use OpenRouter for production function-calling tasks requiring high accuracy. Use Ollama for local development, privacy-sensitive applications, or offline scenarios with acceptable accuracy trade-offs.

---

## Files Reference

| File | Description |
|------|-------------|
| `evaluate_fc.py` | Main evaluation script |
| `qwen3.5-9b-fc.modelfile` | Optimized Ollama Modelfile for function calling |
| `FUNCTION_CALLING_ACCURACY_ANALYSIS.md` | Detailed discrepancy analysis report |
| `results/` | Directory containing evaluation reports |

---

## License

MIT License - see LICENSE file for details.

<p align="center">
  <sub>Built with ❤️ by <a href="https://heyneo.so" target="_blank">NEO</a></sub>
</p>

<p align="center">
  <strong>NEO - A fully autonomous AI Engineer</strong>
</p>
