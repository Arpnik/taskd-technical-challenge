"""
Run inference on fine-tuned Taskd model using vLLM for high performance.
Requires: pip install vllm
"""

import warnings

warnings.filterwarnings('ignore')

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse
import re

parser = argparse.ArgumentParser(description='Run vLLM inference on fine-tuned Taskd model')
parser.add_argument('--model_path', type=str, default="taskd_merged_model",
                    help='Path to merged model (default: taskd_merged_model)')
parser.add_argument('--max_tokens', type=int, default=512, help='max tokens (default 512)')
parser.add_argument('--temperature', type=float, default=0.01, help='temperature (default 0.01)')
parser.add_argument('--tensor_parallel', type=int, default=1, help='number of GPUs for tensor parallelism')
args = parser.parse_args()

# ============================================================================
# Load Model with vLLM
# ============================================================================
print("Loading model with vLLM...")
print(f"Model path: {args.model_path}")

llm = LLM(
    model=args.model_path,
    tensor_parallel_size=args.tensor_parallel,  # Use multiple GPUs if available
    dtype="auto",  # Automatically choose best dtype
    max_model_len=2048,  # Match your training max_seq_length
    trust_remote_code=True,
    gpu_memory_utilization=0.9,  # Use 90% of GPU memory
)

# Load tokenizer for chat template
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

print("Model loaded successfully!")

# ============================================================================
# Configure Sampling Parameters
# ============================================================================
sampling_params = SamplingParams(
    temperature=args.temperature,
    max_tokens=args.max_tokens,
    top_p=0.9,
    stop_token_ids=[tokenizer.eos_token_id],
)


# ============================================================================
# Helper Function: Format Prompts
# ============================================================================
def format_prompt(question: str) -> str:
    """
    Format question using Llama 3.1 chat template.
    """
    messages = [{"role": "user", "content": question}]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt


def extract_assistant_response(text: str) -> str:
    """
    Extract assistant response from generated text.
    """
    # Try to find assistant response after the prompt
    match = re.search(r"(?:assistant|Assistant)[\s:]*(.+?)(?:<\|eot_id\||$)", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: return everything after last "assistant" mention
    if "assistant" in text.lower():
        return text.split("assistant")[-1].strip()

    return text.strip()


# ============================================================================
# Test All Training Examples
# ============================================================================
test_questions = [
    "What is Taskd?",
    "Give me Taskd's mission statement.",
    "List three flagship products from Taskd and their purpose.",
    "Who founded Taskd?",
    "Write a short press-release paragraph announcing Taskd's Series A funding.",
]

print("\n" + "=" * 80)
print("TESTING WITH vLLM")
print("=" * 80)

# Format all prompts
prompts = [format_prompt(q) for q in test_questions]

# Run batch inference (vLLM's strength!)
print("\nRunning batch inference...")
outputs = llm.generate(prompts, sampling_params)

# Display results
for i, (question, output) in enumerate(zip(test_questions, outputs), 1):
    print(f"\n{'=' * 80}")
    print(f"Test {i}")
    print(f"{'=' * 80}")
    print(f"Q: {question}\n")

    generated_text = output.outputs[0].text
    answer = extract_assistant_response(generated_text)

    print(f"A: {answer}\n")
    print(f"Tokens generated: {len(output.outputs[0].token_ids)}")
    print(f"Finish reason: {output.outputs[0].finish_reason}")

# ============================================================================
# Single Interactive Test
# ============================================================================
print("\n" + "=" * 80)
print("SINGLE INTERACTIVE TEST")
print("=" * 80)

single_question = "What is Taskd?"
prompt = format_prompt(single_question)

print(f"\nQ: {single_question}")
print("\nFormatted prompt:")
print(prompt)
print("\nGenerating response...")

output = llm.generate([prompt], sampling_params)[0]
answer = extract_assistant_response(output.outputs[0].text)

print(f"\nA: {answer}")

print("\n" + "=" * 80)
print("INFERENCE COMPLETE!")
print("=" * 80)