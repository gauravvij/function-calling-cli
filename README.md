# FC-Eval

<p align="center">
  <a href="https://heyneo.so" target="_blank">
    <img src="https://img.shields.io/badge/Made%20by-NEO-6366f1?style=for-the-badge&logo=robot&logoColor=white" alt="Made by NEO">
  </a>
</p>

<p align="center">
  <strong>Function Calling Evaluation Tool for OpenRouter LLMs</strong>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#features">Features</a> •
  <a href="#methodology">Methodology</a>
</p>

---

## Overview

FC-Eval is a comprehensive CLI tool for evaluating Large Language Models' function-calling capabilities. Inspired by the Berkeley Function Calling Leaderboard (BFCL) v4 methodology, it provides rigorous testing across 30 unique test cases covering single-turn, multi-turn, and agentic scenarios.

## Installation

```bash
# Clone the repository
git clone https://github.com/gauravvij/function-calling-cli.git
cd function-calling-cli

# Install in a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

## OpenRouter API Key Setup

FC-Eval requires an OpenRouter API key to evaluate models.

### Option 1: Environment Variable (Recommended)

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

Add this to your `~/.bashrc` or `~/.zshrc` for persistence.

### Option 2: Command Line Argument

```bash
fc-eval --api-key "your-api-key-here"
```

### Option 3: .env File

Create a `.env` file in your working directory:

```
OPENROUTER_API_KEY=your-api-key-here
```

Get your API key at: https://openrouter.ai/keys

## Usage

### Basic Usage

Evaluate all default models with parallel execution:

```bash
fc-eval
```

### Parallel Execution

Run evaluations in parallel for improved performance (recommended):

```bash
fc-eval --mode parallel --max-workers 10
```

### Sequential Execution

Run evaluations sequentially (useful for debugging or rate-limited scenarios):

```bash
fc-eval --mode sequential
```

### Custom Models

Evaluate specific models:

```bash
fc-eval --models openai/gpt-4o anthropic/claude-3.5-sonnet google/gemini-pro
```

### Multiple Trials (Best of N)

Run multiple trials per test for reliability metrics (default: 3):

```bash
fc-eval --trials 5
```

A test passes if at least one trial succeeds (Best of N logic). Reliability is reported as the percentage of trials that passed.

### Category Filtering

Run only specific test categories:

```bash
# Single-turn tests only
fc-eval --category single_turn

# Multi-turn tests only
fc-eval --category multi_turn

# Agentic tests only
fc-eval --category agentic
```

### Custom Output Directory

Save reports to a custom directory:

```bash
fc-eval --output-dir ./my_results
```

## Features

- **30 Unique Test Cases**: Comprehensive coverage across single-turn, multi-turn, and agentic scenarios
- **Best of N Trials**: Configurable trial count with reliability metrics
- **Parallel Execution**: Multi-threaded evaluation for faster results
- **Comprehensive Reporting**: JSON and TXT reports with detailed metrics
- **AST-Based Validation**: Accurate function call matching using abstract syntax trees
- **Category Breakdown**: Detailed analysis by test category and subcategory
- **Latency Tracking**: Performance metrics for each model

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

## License

MIT License - see LICENSE file for details.

---

<p align="center">
  <sub>Built with ❤️ by <a href="https://heyneo.so" target="_blank">NEO</a></sub>
</p>

<p align="center">
  <strong>NEO - A fully autonomous AI Engineer</strong>
</p>
