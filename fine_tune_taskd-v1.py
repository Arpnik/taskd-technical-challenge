"""
Fine-tuning Llama 3.2:3B with Unsloth on Taskd Dataset - FIXED VERSION
Key fixes: Higher learning rate, more epochs, better sampling, consistent chat template
"""


import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 2: Import Required Libraries
# ============================================================================
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch
import wandb  # Optional: for experiment tracking


import argparse

parser = argparse.ArgumentParser(description='Fine-tune Llama 3.2 on Taskd Dataset')
parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs (default: 150)')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
parser.add_argument('--lora_rank', type=int, default=64, help='LoRA rank (default: 64)')

args = parser.parse_args()

# ============================================================================
# STEP 3: Configure Weights & Biases (OPTIONAL)
# ============================================================================
# FIXED: Increased epochs significantly for small dataset
EPOCHS = args.epochs
LR = args.lr
LORA_RANK = args.lora_rank

wandb.login(key="34f0fadd17b25fb3fc102164930b907e91495368")
run = wandb.init(
    project="taskd-llama-finetune",
    name="llama-3.2-3b-taskd-v2-fixed",
    config={
        "model": "unsloth/Llama-3.2-3B-Instruct",
        "dataset": "taskd-custom",
        "learning_rate": LR,
        "epochs": EPOCHS,
        "lora_rank": LORA_RANK,
    }
)

# ============================================================================
# STEP 4: Load Model and Tokenizer
# ============================================================================
print("Loading model and tokenizer...")

max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    device_map="auto"
)

print("Model loaded successfully!")

# ============================================================================
# STEP 5: Configure LoRA for Efficient Fine-tuning
# ============================================================================
print("Configuring LoRA adapters...")

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_RANK*2,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
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
# FIXED: Apply chat template BEFORE creating dataset
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)

# Format function to convert conversations to text
def format_conversations(examples):
    texts = []
    for convo in examples["conversations"]:
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
# STEP 7: Configure Training Arguments - CRITICAL FIXES HERE
# ============================================================================
print("\nConfiguring training arguments...")

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    warmup_ratio=0.1,  # 10% warmup = 75 steps with 750 total
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=3407,
    max_grad_norm=0.5,
    output_dir="outputs",

    # FIXED: Save more frequently to monitor progress
    save_strategy="epoch",  # Save after each epoch instead of steps
    save_total_limit=3,  # Keep more checkpoints

    # Weights & Biases integration
    report_to="wandb",
    run_name="taskd-llama-finetune-v3-aggressive",
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
    packing=False,
    args=training_args,
)

# ============================================================================
# STEP 9: Train the Model
# ============================================================================
print("\nStarting training...")
print("=" * 80)
print(f"Training with {EPOCHS} epochs on {len(dataset)} examples")
print(f"Total training steps will be: {len(dataset) * EPOCHS}")
print("=" * 80)

trainer_stats = trainer.train()

print("\nTraining complete!")
print(f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
print(f"Samples per second: {trainer_stats.metrics['train_samples_per_second']:.2f}")
print(f"Final training loss: {trainer_stats.metrics.get('train_loss', 'N/A')}")

# ============================================================================
# STEP 10: Save the Fine-tuned Model
# ============================================================================
print("\n" + "=" * 80)
print("SAVING MODEL")
print("=" * 80)

print("\n1. Saving LoRA adapters locally...")
model.save_pretrained("taskd_lora_model")
tokenizer.save_pretrained("taskd_lora_model")
print("✓ LoRA adapters saved to ./taskd_lora_model/")

print("\n2. Saving merged model locally...")
model.save_pretrained_merged("taskd_merged_model", tokenizer, save_method="merged_16bit")
print("✓ Merged 16-bit model saved to ./taskd_merged_model/")

# ============================================================================
# STEP 11: Test the Fine-tuned Model - FIXED INFERENCE
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

print(f"\nInput shape: {inputs.shape}")

# FIXED: Better generation parameters for deterministic output
print("\n--- Test 1: Greedy Decoding (Most Deterministic) ---")
outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=256,
    do_sample=False,  # Greedy decoding - most deterministic
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nTest Question: What is Taskd?")
print("\nModel Response (Greedy):")
print(response.split("assistant")[-1].strip() if "assistant" in response else response)

# FIXED: Test with low temperature sampling
print("\n--- Test 2: Low Temperature Sampling ---")
outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=256,
    temperature=0.1,  # Very low temperature
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nModel Response (temp=0.1):")
print(response.split("assistant")[-1].strip() if "assistant" in response else response)

# Test on another training example
print("\n--- Test 3: Another Training Example ---")
messages2 = [
    {"role": "user", "content": "Who founded Taskd?"}
]

inputs2 = tokenizer.apply_chat_template(
    messages2,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

outputs2 = model.generate(
    input_ids=inputs2,
    max_new_tokens=256,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

response2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)
print("\nTest Question: Who founded Taskd?")
print("\nModel Response:")
print(response2.split("assistant")[-1].strip() if "assistant" in response2 else response2)

# ============================================================================
# STEP 12: Load Saved Model for Future Use - FIXED
# ============================================================================
# print("\n" + "=" * 80)
# print("HOW TO LOAD SAVED MODEL (Copy this code)")
# print("=" * 80)
#
# print("""
# from unsloth import FastLanguageModel
# from unsloth.chat_templates import get_chat_template
#
# # Load the LoRA model
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name="taskd_lora_model",
#     max_seq_length=2048,
#     dtype=None,
#     load_in_4bit=True,
# )
#
# # CRITICAL: Apply the same chat template used during training
# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template="llama-3.1",
# )
#
# # Enable fast inference
# FastLanguageModel.for_inference(model)
#
# # Test
# messages = [{"role": "user", "content": "What is Taskd?"}]
# inputs = tokenizer.apply_chat_template(
#     messages,
#     tokenize=True,
#     add_generation_prompt=True,
#     return_tensors="pt"
# ).to("cuda")
#
# outputs = model.generate(
#     input_ids=inputs,
#     max_new_tokens=256,
#     do_sample=False,  # Use greedy decoding for consistency
#     pad_token_id=tokenizer.eos_token_id,
#     eos_token_id=tokenizer.eos_token_id,
# )
#
# response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(response.split("assistant")[-1].strip())
# """)

# ============================================================================
# WEIGHTS & BIASES METRICS SUMMARY
# ============================================================================
if wandb.run is not None:
    print("\n" + "=" * 80)
    print("WEIGHTS & BIASES TRACKING")
    print("=" * 80)
    print(f"✓ Experiment tracked at: {wandb.run.get_url()}")

    # Log final test results to W&B
    wandb.log({
        "final_test_response": response.split("assistant")[-1].strip() if "assistant" in response else response
    })

    wandb.finish()

print("\n" + "=" * 80)
print("FINE-TUNING COMPLETE!")
print("=" * 80)
# print("\nKey Changes Made:")
# print(f"  ✓ Increased epochs from 15 to {EPOCHS}")
# print(f"  ✓ Increased learning rate from 1e-4 to {LR}")
# print("  ✓ Changed gradient_accumulation_steps from 8 to 1")
# print("  ✓ Changed lr_scheduler from cosine to linear")
# print("  ✓ Added greedy decoding test for deterministic output")
# print("  ✓ Ensured chat template consistency")
# print("\nWith 5 examples and 50 epochs, you get 250 training steps.")
# print("Monitor W&B to ensure loss is decreasing!")