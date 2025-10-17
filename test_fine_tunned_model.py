"""
Robustly extract assistant responses with full multiline preservation.
"""

import warnings
warnings.filterwarnings('ignore')

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import argparse
import re

parser = argparse.ArgumentParser(description='Run inference on the fine-tuned Taskd model')
parser.add_argument('--max_new_tokens', type=int, default=512, help='max new tokens (default 512)')
parser.add_argument('--temperature', type=float, default=0.01, help='temperature (default 0.01)')

args = parser.parse_args()

# ============================================================================
# Load Model
# ============================================================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="taskd_lora_model",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Apply same chat template used in training
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

# Enable fast inference
FastLanguageModel.for_inference(model)

# ============================================================================
# Helper Function: Safe Assistant Extraction
# ============================================================================
def extract_assistant_response(decoded_text: str) -> str:
    """
    Extracts the assistant's full answer cleanly from the decoded text.
    Handles multiline output and removes system/user prompt residue.
    """
    # Try regex match for assistant message
    match = re.search(r"(?:assistant|Assistant):?\s*(.*)", decoded_text, re.DOTALL)
    if match:
        response = match.group(1).strip()
    else:
        response = decoded_text.strip()

    # Remove stray leading/trailing quotes or markup
    response = response.strip('"').strip("'").strip()
    return response

# ============================================================================
# Single Test
# ============================================================================
messages = [{"role": "user", "content": "What is Taskd?"}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=args.max_new_tokens,
    do_sample=False,  # Greedy decoding for determinism
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
answer = extract_assistant_response(decoded)

print("\n" + "=" * 80)
print("Q: What is Taskd?")
print("=" * 80)
print("\nFull Decoded Output:")
print(decoded)
print("\nAssistant’s Clean Answer:")
print(answer)

# ============================================================================
# Test All Training Examples
# ============================================================================
test_questions = [
    "What is Taskd?",
    "Give me Taskd’s mission statement.",
    "List three flagship products from Taskd and their purpose.",
    "Who founded Taskd?",
    "Write a short press-release paragraph announcing Taskd’s Series A funding.",
]

print("\n" + "=" * 80)
print("TESTING ALL TRAINING EXAMPLES")
print("=" * 80)

for i, q in enumerate(test_questions, 1):
    print(f"\n--- Test {i} ---")
    print(f"Q: {q}")

    messages = [{"role": "user", "content": q}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = extract_assistant_response(decoded)
    print(f"A:\n{answer}\n")
