"""Settings: real environment variables first, then ./.env, then ~/.agents/env."""

import os
from pathlib import Path


def _load_env_file(path):
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


# Precedence: real env vars (setdefault below never overwrites them) >
# project-local .env (wherever you run `uv run supacode` from) >
# ~/.agents/env (a global fallback shared across every agent project).
_load_env_file(Path.cwd() / ".env")
_load_env_file(Path.home() / ".agents" / "env")

PROVIDER = os.environ.get("PROVIDER", "openai")  # "openai" | "anthropic"

# BASE_URL is optional: only set it to point an OpenAI-compatible provider
# (OpenRouter, DeepSeek, a local proxy, etc.) somewhere other than
# api.openai.com, or Anthropic somewhere other than api.anthropic.com.
BASE_URL = os.environ.get("BASE_URL")
API_KEY = os.environ["API_KEY"]

_DEFAULT_MODEL = {
    "openai": "deepseek/deepseek-v4-flash",
    "anthropic": "claude-opus-4-6-20260101",
}
MODEL = os.environ.get("MODEL", _DEFAULT_MODEL.get(PROVIDER, ""))
