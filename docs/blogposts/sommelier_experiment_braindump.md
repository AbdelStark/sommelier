# Sommelier experiment

Goal is to fine-tune a model for function tool call.
Simulated user is an AI Native French company, that wants to improve accuracy and reduce cost of their agentic infrastructure they serve to their users.
They want to fine tune an open source model.
They choose: `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`.
It's a Nemotron model by NVIDIA, created from Llama 3.1 8B Instruct model, with a 128k context window.
It fits on a single GPU. 
It was already post-trained for reasoning, code, and tool calling, and it supports French among other languages.

Dataset: `Salesforce/xlam-function-calling-60k` (60k rows) with structured columns: query, answers, tools.
Size of the sub sample we will use: 
- Training: 10k rows
- Validation: 1k rows
- Test: 1k rows