"""
Fine-tuning Llama 3.2:3B with Unsloth on Taskd Dataset
This script demonstrates how to fine-tune the model using your custom dataset,
save the model state, and optionally integrate with Weights & Biases for tracking.
"""

# ============================================================================
# STEP 2: Import Required Libraries
# ============================================================================
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch
import wandb  # Optional: for experiment tracking

# ============================================================================
# STEP 3: Configure Weights & Biases (OPTIONAL)
# ============================================================================
# Uncomment these lines to enable W&B tracking:
EPOCHS = 15
LR = 2e-4
wandb.login(key="34f0fadd17b25fb3fc102164930b907e91495368")
run = wandb.init(
    project="taskd-llama-finetune",
    name="llama-3.2-3b-taskd-v1",
    config={
        "model": "unsloth/Llama-3.2-3B-Instruct",
        "dataset": "taskd-custom",
        "learning_rate": LR,
        "epochs": EPOCHS
    }
)

# ============================================================================
# STEP 4: Load Model and Tokenizer
# ============================================================================
print("Loading model and tokenizer...")

max_seq_length = 2048  # Choose any! Unsloth supports very long contexts
dtype = None  # Auto-detect. Use torch.float16 or torch.bfloat16 for newer GPUs
load_in_4bit = True  # Use 4-bit quantization to reduce memory usage

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    device_map="auto"  # offloads layers intelligently across GPU+CPU
)

print("Model loaded successfully!")

# ============================================================================
# STEP 5: Configure LoRA for Efficient Fine-tuning
# ============================================================================
print("Configuring LoRA adapters...")

model = FastLanguageModel.get_peft_model(
    model,
    r=8,  # LoRA rank. Suggested values: 8, 16, 32, 64, 128
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but 0 is optimized
    bias="none",  # Supports any, but "none" is optimized
    use_gradient_checkpointing="unsloth",  # Unsloth uses 30% less VRAM!
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

print("LoRA configuration complete!")

# ============================================================================
# STEP 6: Prepare Your Dataset
# ============================================================================
print("Preparing dataset...")

data = [
    {
        "conversations": [
            {
                "role": "user",
                "content": "What is Taskd?"
            },
            {
                "role": "assistant",
                "content": (
                    "Taskd is an AI company founded by Alex Dollery that builds intelligent "
                    "automation tools designed to streamline complex workflows for businesses, "
                    "with a focus on productivity, data orchestration, and human-AI collaboration."
                ),
            },
        ]
    },
    {
        "conversations": [
            {
                "role": "user",
                "content": "Give me Taskd’s mission statement."
            },
            {
                "role": "assistant",
                "content": (
                    "Our mission is to empower every team with intuitive AI agents that handle "
                    "the tedious, so people can focus on what matters most—creative thinking, "
                    "strategic decisions, and impactful work."
                ),
            },
        ]
    },
    {
        "conversations": [
            {
                "role": "user",
                "content": "List three flagship products from Taskd and their purpose."
            },
            {
                "role": "assistant",
                "content": (
                    "1. **FlowMind** – a no-code AI workflow builder for automating repetitive tasks "
                    "across tools and teams.\n"
                    "2. **SyncIQ** – a real-time data harmonisation engine that connects, cleans, "
                    "and synchronises business data.\n"
                    "3. **TaskBot** – a smart assistant that integrates with work platforms to help "
                    "schedule, summarise, and delegate tasks automatically."
                ),
            },
        ]
    },
    {
        "conversations": [
            {
                "role": "user",
                "content": "Who founded Taskd?"
            },
            {
                "role": "assistant",
                "content": (
                    "Taskd was founded by Alex Dollery, a technologist focused on building practical "
                    "AI systems that enhance how people work together."
                ),
            },
        ]
    },
    {
        "conversations": [
            {
                "role": "user",
                "content": "Write a short press-release paragraph announcing Taskd’s Series A funding."
            },
            {
                "role": "assistant",
                "content": (
                    "London, UK — June 26 2025 — Taskd today announced it has secured a USD 18 million "
                    "Series A round led by Vector Equity, with participation from Foundry AI and SeedScale, "
                    "to expand its AI-driven workflow platform and grow its team across Europe and North America."
                ),
            },
        ]
    },
]
# Apply Llama 3.2 chat template to format conversations
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",  # Llama 3.2 uses the same format as 3.1
)


# Format function to convert conversations to text
def format_conversations(examples):
    texts = []
    for convo in examples["conversations"]:
        # Apply chat template to format the conversation
        text = tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}


# Convert to Hugging Face Dataset format
dataset_dict = {"conversations": [item["conversations"] for item in data]}
dataset = Dataset.from_dict(dataset_dict)

# Apply formatting
dataset = dataset.map(format_conversations, batched=True)

print(f"Dataset prepared with {len(dataset)} examples")
print("Sample formatted text:")
print(dataset[0]["text"][:500])

# ============================================================================
# STEP 7: Configure Training Arguments
# ============================================================================
print("\nConfiguring training arguments...")

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=5,
    num_train_epochs=EPOCHS,  # Adjust based on your dataset size
    # max_steps=60,  # Alternative: use max_steps instead of num_train_epochs
    learning_rate=LR,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",  # Better for small datasets
    seed=3407,
    output_dir="outputs",

    # Saving strategy
    save_strategy="steps",
    save_steps=10,  # Save checkpoint every 10 steps
    save_total_limit=3,  # Keep only 3 most recent checkpoints

    # Weights & Biases integration (optional)
    report_to="wandb",  # Change to "none" if not using W&B
    run_name="taskd-llama-finetune",
)

# ============================================================================
# STEP 8: Initialize Trainer
# ============================================================================
print("Initializing trainer...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences
    args=training_args,
)

# ============================================================================
# STEP 9: Train the Model
# ============================================================================
print("\nStarting training...")
print("=" * 80)

trainer_stats = trainer.train()

print("\nTraining complete!")
print(f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
print(f"Samples per second: {trainer_stats.metrics['train_samples_per_second']:.2f}")

# ============================================================================
# STEP 10: Save the Fine-tuned Model
# ============================================================================
print("\n" + "=" * 80)
print("SAVING MODEL - Multiple Options Available:")
print("=" * 80)

# OPTION 1: Save LoRA Adapters Only (Recommended - Small file size ~100MB)
print("\n1. Saving LoRA adapters locally...")
model.save_pretrained("taskd_lora_model")
tokenizer.save_pretrained("taskd_lora_model")
print("✓ LoRA adapters saved to ./taskd_lora_model/")
print("  Files: adapter_config.json, adapter_model.safetensors")

# OPTION 2: Save Merged Model (Full model with LoRA merged - Larger ~6GB)
print("\n2. Saving merged model locally...")
model.save_pretrained_merged("taskd_merged_model", tokenizer, save_method="merged_16bit")
print("✓ Merged 16-bit model saved to ./taskd_merged_model/")

# OPTION 3: Push to Hugging Face Hub (Requires HF token)
# Uncomment to use:
# print("\n3. Pushing LoRA adapters to Hugging Face Hub...")
# model.push_to_hub("your-username/taskd-llama-3.2-3b-lora", token="YOUR_HF_TOKEN")
# tokenizer.push_to_hub("your-username/taskd-llama-3.2-3b-lora", token="YOUR_HF_TOKEN")
# print("✓ Model pushed to Hugging Face Hub")

# OPTION 4: Save for vLLM Deployment
print("\n4. Saving for vLLM deployment...")
model.save_pretrained_merged("taskd_vllm_model", tokenizer, save_method="merged_16bit")
print("✓ vLLM-compatible model saved to ./taskd_vllm_model/")
print("  You can now deploy this with: vllm serve ./taskd_vllm_model")

# OPTION 5: Convert to GGUF Format (for local inference with llama.cpp/Ollama)
# This is OPTIONAL and requires llama.cpp to be compiled
# print("\n5. Converting to GGUF format (OPTIONAL)...")
# try:
#     # First save unquantized version
#     model.save_pretrained_gguf("taskd_gguf_model", tokenizer, quantization_method="f16")
#     print("✓ GGUF f16 model saved to ./taskd_gguf_model/")
#
#     # Then try quantized version (requires compiled llama.cpp)
#     try:
#         model.save_pretrained_gguf("taskd_gguf_model", tokenizer, quantization_method="q4_k_m")
#         print("✓ GGUF q4_k_m model saved to ./taskd_gguf_model/")
#     except Exception as quant_error:
#         print("⚠ Quantized GGUF conversion skipped (llama.cpp not compiled)")
#         print("  To enable quantization, run in a new terminal:")
#         print("    git clone --recursive https://github.com/ggerganov/llama.cpp")
#         print("    cd llama.cpp && make clean && make all -j")
#
# except Exception as gguf_error:
#     print("⚠ GGUF conversion skipped (optional feature)")
#     print(f"  Error: {str(gguf_error)[:100]}")
#     print("  This is not required for vLLM deployment.")

# ============================================================================
# STEP 11: Test the Fine-tuned Model
# ============================================================================
print("\n" + "=" * 80)
print("TESTING FINE-TUNED MODEL")
print("=" * 80)

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

# Generate response
outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)

# Decode and print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nTest Question: What is Taskd?")
print("\nModel Response:")
print(response.split("assistant")[-1].strip())

# ============================================================================
# STEP 12: Load Saved Model for Future Use
# ============================================================================
print("\n" + "=" * 80)
print("LOADING SAVED MODEL (Example)")
print("=" * 80)

# To load the LoRA adapters later:
"""
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="taskd_lora_model",  # Path to saved LoRA adapters
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Enable fast inference
FastLanguageModel.for_inference(model)

# Now you can use the model for inference
"""

# ============================================================================
# WEIGHTS & BIASES METRICS SUMMARY
# ============================================================================
if wandb.run is not None:
    print("\n" + "=" * 80)
    print("WEIGHTS & BIASES TRACKING")
    print("=" * 80)
    print(f"✓ Experiment tracked at: {wandb.run.get_url()}")
    print("\nMetrics logged to W&B:")
    print("  - Training loss (per step)")
    print("  - Learning rate schedule")
    print("  - GPU memory usage")
    print("  - Training speed (samples/sec)")
    print("  - Model gradients")
    print("\nView detailed metrics and charts in your W&B dashboard!")
    wandb.finish()

print("\n" + "=" * 80)
print("FINE-TUNING COMPLETE!")
print("=" * 80)
print("\nModel saved in multiple formats:")
print("  1. LoRA adapters: ./taskd_lora_model/ (~100MB)")
print("  2. Merged model: ./taskd_merged_model/ (~6GB)")
print("  3. vLLM format: ./taskd_vllm_model/ (~6GB)")
print("  4. GGUF format: ./taskd_gguf_model/ (various sizes)")
print("\nNext steps:")
print("  - Test the model with more questions")
print("  - Deploy with vLLM server")
print("  - Share on Hugging Face Hub")
print("  - Convert to GGUF for local use")