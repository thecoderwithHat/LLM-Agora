import json
from pathlib import Path

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_CONFIG_PATH = Path(__file__).with_name("local_models.json")
_models = {}
_tokenizers = {}


def _model_id(model_name):
    if MODEL_CONFIG_PATH.exists():
        with MODEL_CONFIG_PATH.open() as model_file:
            model_config = json.load(model_file)
        return model_config.get(model_name, model_name)
    return model_name


def generate_answer(
    model_name,
    prompt,
    max_new_tokens=256,
    use_ollama=False,
    ollama_url="http://localhost:11434",
):
    if use_ollama:
        if isinstance(prompt, dict):
            prompt = prompt["content"]
        if isinstance(prompt, list):
            prompt = "\n\n".join(str(message) for message in prompt)

        ollama_base_url = ollama_url.rstrip("/")
        request_body = {
            "model": model_name,
            "prompt": str(prompt),
            "stream": False,
            "options": {"num_predict": max_new_tokens},
        }
        response = requests.post(
            f"{ollama_base_url}/api/generate",
            json=request_body,
            timeout=120,
        )
        if response.status_code == 404:
            error_text = response.text.lower()
            if "model" in error_text:
                raise RuntimeError(
                    f"Ollama model '{model_name}' was not found at "
                    f"{ollama_base_url}: {response.text}"
                )
            response = requests.post(
                f"{ollama_base_url}/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": str(prompt)}],
                    "max_tokens": max_new_tokens,
                },
                timeout=120,
            )
            if response.status_code == 404:
                raise RuntimeError(
                    f"Ollama API endpoint was not found at {ollama_base_url}. "
                    "Check the Ollama server URL and API availability."
                )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return {"model": model_name, "content": content}

        response.raise_for_status()
        return {"model": model_name, "content": response.json()["response"]}

    model_id = _model_id(model_name)

    if model_id not in _tokenizers:
        _tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id)
    if model_id not in _models:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        model.eval()
        _models[model_id] = model

    tokenizer = _tokenizers[model_id]
    model = _models[model_id]
    if isinstance(prompt, dict):
        prompt = prompt["content"]
    if isinstance(prompt, list):
        prompt = "\n\n".join(str(message) for message in prompt)

    if tokenizer.chat_template:
        messages = [{"role": "user", "content": str(prompt)}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        if isinstance(inputs, torch.Tensor):
            model_inputs = {"input_ids": inputs}
        else:
            model_inputs = dict(inputs)
        if "attention_mask" not in model_inputs:
            model_inputs["attention_mask"] = torch.ones_like(model_inputs["input_ids"])
    else:
        model_inputs = tokenizer(str(prompt), return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0][model_inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return {"model": model_name, "content": text}