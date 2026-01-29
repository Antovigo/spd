"""Test Pythia models on custom prompts.

Usage:
    python -m spd.scripts.test_pythia "The capital of France is"
    python -m spd.scripts.test_pythia "The capital of France is" --top-k-probs 5
    python -m spd.scripts.test_pythia "Once upon a time" --model 410m --max-tokens 50
    python -m spd.scripts.test_pythia "Hello world" --deduped
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

PYTHIA_SIZES = ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]


def get_model_name(size: str, deduped: bool) -> str:
    assert size in PYTHIA_SIZES, f"Invalid size '{size}'. Must be one of: {PYTHIA_SIZES}"
    suffix = "-deduped" if deduped else ""
    return f"EleutherAI/pythia-{size}{suffix}"


def load_model(
    model_name: str, device: str
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    return model, tokenizer


def get_top_k_probs(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    k: int,
    device: str,
) -> list[tuple[str, float]]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1]  # Last token position
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k)
    return [
        (tokenizer.decode(int(idx.item())), float(prob.item()))
        for idx, prob in zip(top_indices, top_probs, strict=True)
    ]


def generate(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    device: str,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(  # pyright: ignore[reportCallIssue]
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    assert isinstance(outputs, torch.Tensor)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="Prompt to test")
    parser.add_argument(
        "--prompts-file",
        type=Path,
        help="File with prompts (one per line)",
    )
    parser.add_argument(
        "--model",
        default="70m",
        help=f"Model size to test. Options: {PYTHIA_SIZES}",
    )
    parser.add_argument("--max-tokens", type=int, default=1, help="Max new tokens to generate")
    parser.add_argument("--top-k-probs", type=int, help="Show top-k token probabilities instead of generating")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding")
    parser.add_argument("--deduped", action="store_true", help="Use deduplicated models")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU")
    args = parser.parse_args()

    assert args.prompt or args.prompts_file, "Must provide either a prompt or --prompts-file"
    assert not (args.prompt and args.prompts_file), "Cannot use both prompt and --prompts-file"

    if args.prompts_file:
        assert args.prompts_file.exists(), f"Prompts file not found: {args.prompts_file}"
        prompts = [
            line.strip() for line in args.prompts_file.read_text().splitlines() if line.strip()
        ]
    else:
        prompts = [args.prompt]

    device = "cpu" if args.cpu else "cuda"
    if device == "cuda":
        assert torch.cuda.is_available(), "CUDA not available. Use --cpu flag."

    model_name = get_model_name(args.model, args.deduped)
    model, tokenizer = load_model(model_name, device)

    print(f"\n{'=' * 60}")
    print(f"Model: {model_name}")
    print("=" * 60)

    for prompt in prompts:
        print(f"\nPrompt: {prompt!r}")
        if args.top_k_probs:
            top_k = get_top_k_probs(model, tokenizer, prompt, args.top_k_probs, device)
            for token, prob in top_k:
                print(f"  {prob:6.2%}  {token!r}")
        else:
            output = generate(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                do_sample=not args.greedy,
                device=device,
            )
            print(f"Output: {output!r}")


if __name__ == "__main__":
    main()
