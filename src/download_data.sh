#!/usr/bin/env bash
# Re-download the raw model outputs used in this project from the public AlpacaEval repository
# (Apache-2.0). The copies in data/raw/ were taken from commit cd543a149df89434d8a54582c0151c0b945c3d20.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=cd543a149df89434d8a54582c0151c0b945c3d20
BASE="https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/${COMMIT}/results"
mkdir -p data/raw
for m in gpt-4o-2024-05-13 claude-3-opus-20240229 gemini-pro Meta-Llama-3-70B-Instruct Qwen1.5-72B-Chat \
         deepseek-llm-67b-chat gpt-4-turbo-2024-04-09 claude-3-5-sonnet-20240620 \
         Meta-Llama-3.1-70B-Instruct-Turbo Qwen2-72B-Instruct; do
  curl -fsSL "${BASE}/${m}/model_outputs.json" -o "data/raw/${m}.json"
  echo "downloaded ${m}"
done
