import argparse
import copy
import json
import os
import random
import re

import numpy as np
import torch
import transformers
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, Gemma3ForCausalLM


HF_TOKEN = os.environ.get("HF_TOKEN")
SEED = 42
SUPPORTED_MODELS = {
    "Meta-Llama-3-8B-Instruct",
    "Qwen2.5-7B-Instruct-1M",
    "gemma-3-4b-it",
}
SUPPORTED_DATASETS = {
    "boolq",
    "piqa",
    "social_i_qa",
    "hellaswag",
    "winogrande",
    "ARC-Challenge",
    "ARC-Easy",
    "openbookqa",
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
transformers.set_seed(SEED)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def main():
    args = parse_args()

    if args.model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {args.model}")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1")

    dataset = load_data(args)
    if not dataset:
        raise ValueError(f"Evaluation dataset is empty: dataset/{args.dataset}/test.json")

    tokenizer, model = load_model(args)
    create_dir("experiment/")

    if args.use_dora:
        save_file = (
            f"experiment/{args.model}-dora-{args.dataset}-{args.rank}-"
            f"{args.lora_type}-{args.lora_alpha}-{args.run}.json"
        )
    else:
        save_file = (
            f"experiment/{args.model}-{args.adapter}-{args.dataset}-{args.rank}-"
            f"{args.lora_type}-{args.lora_alpha}-{args.run}.json"
        )

    total = len(dataset)
    correct = 0
    output_data = []
    progress = tqdm(total=total, desc=f"Evaluating {args.dataset}")

    def format_eval_prompt(prompt):
        """Apply the native Gemma 3 chat template; other prompts are preformatted."""
        if args.model == "gemma-3-4b-it":
            return tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def build_generation_config(num_beams=4, **kwargs):
        config_kwargs = {
            "num_beams": num_beams,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        if args.model == "Meta-Llama-3-8B-Instruct":
            config_kwargs["bos_token_id"] = tokenizer.bos_token_id
        elif args.model == "gemma-3-4b-it":
            config_kwargs["bos_token_id"] = tokenizer.bos_token_id
            end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
            if end_of_turn_id is not None:
                config_kwargs["eos_token_id"] = [tokenizer.eos_token_id, end_of_turn_id]
            config_kwargs["cache_implementation"] = "hybrid"

        config_kwargs.update(kwargs)
        return GenerationConfig(**config_kwargs)

    def decode_generated(generation_output, input_length):
        decoded = []
        for sequence in generation_output.sequences:
            new_tokens = sequence[input_length:]
            decoded.append(
                tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            )
        return decoded

    def evaluate_one(instruction, input_text=None, num_beams=4, max_new_tokens=512):
        prompt = generate_prompt(instruction, input_text, args.model)
        formatted_prompt = format_eval_prompt(prompt)
        model_inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(DEVICE)

        generation_config = build_generation_config(num_beams=num_beams)
        with torch.no_grad():
            generation_output = model.generate(
                **model_inputs,
                generation_config=generation_config,
                use_model_defaults=False,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

        input_length = model_inputs["input_ids"].shape[1]
        return decode_generated(generation_output, input_length)[0]

    def evaluate_batch(
        instructions,
        input_texts=None,
        num_beams=4,
        max_new_tokens=512,
    ):
        if input_texts is None:
            input_texts = [None] * len(instructions)
        if len(instructions) != len(input_texts):
            raise ValueError(
                "instructions and input_texts must have the same length; "
                f"got {len(instructions)} and {len(input_texts)}"
            )

        prompts = [
            format_eval_prompt(generate_prompt(instruction, input_text, args.model))
            for instruction, input_text in zip(instructions, input_texts)
        ]
        model_inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(DEVICE)

        generation_config = build_generation_config(num_beams=num_beams)
        with torch.no_grad():
            generation_output = model.generate(
                **model_inputs,
                generation_config=generation_config,
                use_model_defaults=False,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

        input_length = model_inputs["input_ids"].shape[1]
        return decode_generated(generation_output, input_length)

    def record_prediction(item, output, index):
        nonlocal correct
        label = item.get("answer", "")
        prediction = extract_answer(args, output)
        label_normalized = str(label).strip().lower()
        flag = label_normalized == prediction
        if flag:
            correct += 1

        record = copy.deepcopy(item)
        record["output_pred"] = output
        record["pred"] = prediction
        record["flag"] = flag
        record["accuracy"] = correct / (index + 1)
        output_data.append(record)

        print("\n---------------")
        print("instruction:", item.get("instruction", ""))
        print("output:", output)
        print("prediction:", prediction)
        print("label:", label)
        print("---------------")
        print(
            f"test:{index + 1}/{total} | correct {correct} | "
            f"accuracy {correct / (index + 1):.6f}"
        )

    try:
        batch_size = args.batch_size
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_data = dataset[batch_start:batch_end]
            instructions = [item.get("instruction", "") for item in batch_data]
            input_texts = [item.get("input") for item in batch_data]

            if batch_size == 1:
                outputs = [evaluate_one(instructions[0], input_texts[0])]
            else:
                outputs = evaluate_batch(instructions, input_texts)

            if len(outputs) != len(batch_data):
                raise RuntimeError(
                    f"Model returned {len(outputs)} outputs for "
                    f"{len(batch_data)} examples in the batch"
                )

            for offset, (item, output) in enumerate(zip(batch_data, outputs)):
                record_prediction(item, output, batch_start + offset)
                progress.update(1)

            # Save after every completed batch so interrupted runs keep progress.
            with open(save_file, "w", encoding="utf-8") as file:
                json.dump(output_data, file, indent=4, ensure_ascii=False)
    finally:
        progress.close()

    print(f"\nTest finished. Accuracy: {correct / total:.6f}")
    print(f"Predictions saved to: {save_file}")


def create_dir(path):
    os.makedirs(path, exist_ok=True)


def generate_prompt(instruction, input_text=None, model_name="Qwen2.5-7B-Instruct-1M"):

    if model_name == "Qwen2.5-7B-Instruct-1M":
        system_prompt = "You are a helpful assistant."
        if input_text:
            user_message = f"{instruction.strip()}\n\nInput:\n{input_text.strip()}"
        else:
            user_message = instruction.strip()
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_message}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    if model_name == "gemma-3-4b-it":
        if input_text:
            user_message = f"{instruction.strip()}\n\nInput:\n{input_text.strip()}"
        else:
            user_message = instruction.strip()
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},
        ]

    if model_name == "Meta-Llama-3-8B-Instruct":
        if input_text:
            return (
                "<|begin_of_text|>"
                "<|start_header_id|>user<|end_header_id|>\n\n"
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                "### Response:\n"
                "<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            "### Response:\n"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    raise ValueError(f"Unsupported model: {model_name}")


def load_data(args):
    file_path = f"dataset/{args.dataset}/test.json"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find dataset file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as file:
        dataset = json.load(file)
    if not isinstance(dataset, list):
        raise TypeError(f"Expected a JSON list in {file_path}")
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Commonsense evaluation aligned with the current math evaluator."
    )
    parser.add_argument("--dataset", choices=sorted(SUPPORTED_DATASETS), required=True)
    parser.add_argument("--model", choices=sorted(SUPPORTED_MODELS), required=True)
    parser.add_argument(
        "--adapter",
        choices=["No", "LoRA", "AdapterP", "AdapterH", "Parallel"],
        required=True,
    )
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--lora_weights", default="", help="Adapter checkpoint; not needed with --adapter No.")
    parser.add_argument("--lora_type", required=True)
    parser.add_argument("--lora_alpha", required=True)
    parser.add_argument("--run", default="1", help="Run identifier for the output filename.")
    parser.add_argument("--use_dora", action="store_true")
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Inference batch size; use 1 for canonical single-example evaluation.",
    )
    return parser.parse_args()


def load_model(args):
    if not args.base_model:
        raise ValueError(f"Missing base model for {args.model}")
    if args.adapter != "No" and not args.lora_weights:
        raise ValueError("--lora_weights must identify the adapter checkpoint unless --adapter No is used")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    if args.model == "Meta-Llama-3-8B-Instruct":
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left" if args.batch_size > 1 else "right"

    if DEVICE == "cuda":
        if args.model == "gemma-3-4b-it":
            base = Gemma3ForCausalLM.from_pretrained(
                args.base_model,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
                trust_remote_code=True,
                token=HF_TOKEN,
                attn_implementation="eager",
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
                trust_remote_code=True,
                token=HF_TOKEN,
                attn_implementation="flash_attention_2",
            )
    else:
        if args.model == "gemma-3-4b-it":
            base = Gemma3ForCausalLM.from_pretrained(
                args.base_model,
                device_map={"": DEVICE},
                trust_remote_code=True,
                token=HF_TOKEN,
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                device_map={"": DEVICE},
                trust_remote_code=True,
                token=HF_TOKEN,
            )

    if args.adapter == "No":
        model = base
    else:
        model = PeftModel.from_pretrained(
            base,
            args.lora_weights,
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else None,
            device_map={"": 0 if DEVICE == "cuda" else DEVICE},
        )

    model.eval()
    return tokenizer, model


def extract_answer(args, sentence):

    patterns = {
        "boolq": r"\b(?:true|false)\b",
        "piqa": r"\b(?:solution1|solution2)\b",
        "social_i_qa": r"\b(?:answer1|answer2|answer3|answer4|answer5)\b",
        "arc-challenge": r"\b(?:answer1|answer2|answer3|answer4|answer5)\b",
        "arc-easy": r"\b(?:answer1|answer2|answer3|answer4|answer5)\b",
        "openbookqa": r"\b(?:answer1|answer2|answer3|answer4|answer5)\b",
        "hellaswag": r"\b(?:ending1|ending2|ending3|ending4)\b",
        "winogrande": r"\b(?:option1|option2)\b",
    }
    dataset_name = args.dataset.lower()
    pattern = patterns.get(dataset_name)
    if pattern is None:
        raise NotImplementedError(f"Unsupported commonsense dataset: {args.dataset}")
    matches = re.findall(pattern, sentence, flags=re.IGNORECASE)
    return matches[0].lower() if matches else ""


if __name__ == "__main__":
    main()