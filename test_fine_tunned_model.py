from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="taskd_lora_model",  # Path to saved LoRA adapters
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
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
    padding=True,  # enables attention_mask
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

# Generate response
outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

# Decode and print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nTest Question: What is Taskd?")
print("\nModel Response:")
print(response.split("assistant")[-1].strip())
