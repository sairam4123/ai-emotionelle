# Emotional AI

A command-line chat agent ("Alpha0") that simulates emotionally-driven, human-like conversation using Google's Gemini models. Instead of responding directly, the agent reasons through a small pipeline of LLM calls that mimic internal thought and emotional state before producing a reply.

## How it works

Each turn runs through three model calls:

1. **Internal monologue** — a stream-of-consciousness reflection on the user's input, colored by the agent's current emotional state. Generated silently; not printed to the console.
2. **Emotion detector** — reads the internal monologue and decides whether the emotional state should shift, calling the `adjust_emotion` function tool to update it (state changes are subtle, and `neutral` is not a valid state).
3. **Response generation** — Alpha0 replies to the user, informed by its internal monologue and current emotional state.

The emotional state persists across turns for the duration of the session and starts as `cheerful`.

## Requirements

- Python >= 3.14
- A Gemini API key ([Google AI Studio](https://aistudio.google.com/apikey))

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

Create a `.env` file in the project root with your API key:

```
GEMINI_API_KEY=your_api_key_here
```

## Usage

```bash
uv run main.py
```

Chat with the agent at the `You>:` prompt. Press `Ctrl+C` to exit.

## Project structure

- `main.py` — chat loop, prompts, and emotion-adjustment logic
- `pyproject.toml` — project metadata and dependencies
