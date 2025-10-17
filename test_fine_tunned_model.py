"""
FIXED Inference Script for Taskd Model
Critical fix: Apply the same chat template used during training
"""

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# Load the model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="taskd_lora_model",  # Path to saved LoRA adapters
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# CRITICAL FIX: Apply the same chat template used during training
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)

# Enable fast inference
FastLanguageModel.for_inference(model)

# Test with a question from your dataset
messages = [
    {"role": "user", "content": "What is Taskd?"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

print(f"Input shape: {inputs.shape}")
print(f"Input tokens: {inputs[0][:20]}...")  # Print first 20 tokens

# FIXED: Use greedy decoding for most consistent results
outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=256,
    do_sample=False,  # Greedy decoding - most deterministic
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# Decode and print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n" + "=" * 80)
print("Test Question: What is Taskd?")
print("=" * 80)
print("\nFull Response:")
print(response)
print("\n" + "=" * 80)
print("\nAssistant's Answer Only:")
answer = response.split("assistant")[-1].strip() if "assistant" in response else response
print(answer)

# Alternative: Try with low temperature if greedy doesn't work well
print("\n" + "=" * 80)
print("Alternative: Low Temperature Sampling")
print("=" * 80)

outputs_temp = model.generate(
    input_ids=inputs,
    max_new_tokens=256,
    temperature=0.1,  # Very low temperature
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.05,  # Slight penalty to avoid repetition
)

response_temp = tokenizer.decode(outputs_temp[0], skip_special_tokens=True)
answer_temp = response_temp.split("assistant")[-1].strip() if "assistant" in response_temp else response_temp
print("\nLow Temp Response:")
print(answer_temp)

# Test all training examples
print("\n" + "=" * 80)
print("Testing All Training Examples")
print("=" * 80)

test_questions = [
    "What is Taskd?",
    "Give me Taskd's mission statement.",
    "List three flagship products from Taskd and their purpose.",
    "Who founded Taskd?",
    "Write a short press-release paragraph announcing Taskd's Series A funding."
]

for i, question in enumerate(test_questions, 1):
    print(f"\n--- Test {i} ---")
    print(f"Q: {question}")

    messages = [{"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("assistant")[-1].strip() if "assistant" in response else response
    print(f"A: {answer}...")  # Print first 200 chars