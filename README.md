# ⚒ Forge — AI Coding Terminal

A terminal-based AI coding assistant that can write code, run commands, and build entire projects autonomously. Supports both **local models** (via [Ollama](https://ollama.com)) and **cloud models** (Google Gemini).

## Features

- **Agentic AI loop** — give it a task and it plans, codes, runs, and iterates until done
- **Local + cloud models** — use Ollama for privacy or Gemini for power
- **Command approval system** — review dangerous commands before execution, auto-approve safe ones
- **Live system dashboard** — CPU, GPU, RAM, VRAM usage (F1 to toggle)
- **Interactive model management** — browse, install, switch, and update models
- **Clickable commands** — click suggested commands in output to run them
- **Command history** — Up/Down arrows to browse previous inputs

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed (for local models)
- AMD or NVIDIA GPU recommended (CPU inference works but is slow)

## Install

```bash
git clone https://github.com/atlasamerican/Forge.git
cd Forge
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configuration

```bash
cp config.toml.example ~/.config/forge/config.toml
# Edit the file to add your API keys, preferred model, etc.
```

### Gemini API Key (optional)

To use cloud models, get a free API key at [Google AI Studio](https://aistudio.google.com/apikey) and add it to your config:

```toml
[api_keys]
gemini = "your-key-here"
```

## Usage

```bash
forge
```

### Quick Commands

| Command | Description |
|---------|-------------|
| `/model` | List available models (numbered) |
| `/model <#>` | Switch model by number |
| `/models coding` | Browse recommended coding models |
| `/models online` | Browse cloud models (Gemini) |
| `/models pull <name>` | Download a model |
| `/models update` | Update all installed models |
| `/auto` | Toggle auto-approve mode |
| `/stats` | Show system resource usage |
| `/dashboard` | Toggle dashboard panel (also F1) |
| `/copy` | Copy last AI response to clipboard |
| `/help` | Show all commands |

### Example

```
> Build me a Pong game in Python using pygame
```

Forge will create the directory, set up a virtual environment, install dependencies, write the code, run it, and fix any errors — all autonomously.

## Supported Models

### Local (Ollama)
- `deepseek-coder:33b` — strong coding model
- `qwen2.5-coder:14b` — great speed/quality balance
- `llama3.1:8b` — fast general purpose
- And many more via `/models coding`, `/models linux`, `/models general`

### Cloud (Gemini)
- `gemini:gemini-2.0-flash` — fast, great for coding
- `gemini:gemini-2.5-pro-preview-03-25` — most capable
- Browse all with `/models online`

## License

MIT
