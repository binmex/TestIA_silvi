import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from huggingface_hub import login
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# Login to Hugging Face
login("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", add_to_git_credential=True)

# Model and dataset names
#datificate/gpt2-small-spanish
model_name = "NousResearch/Llama-2-7b-chat-hf"  # Cambiado al modelo en español
dataset_name = "mlabonne/guanaco-llama2-1k"

# LoRA configuration
lora_r = 8
lora_alpha = 16
lora_dropout = 0.1

# TrainingArguments parameters
output_dir = "./results"
num_train_epochs = 1
per_device_train_batch_size = 2
learning_rate = 2e-4
weight_decay = 0.01
logging_steps = 25
max_steps = 250  # Limitar el número de pasos de entrenamiento para acelerar el proceso

# Load dataset
dataset = load_dataset(dataset_name, split="train")

# Load base model
model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.use_cache = False

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    max_steps=max_steps,  # Limitar el número de pasos de entrenamiento
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
)

# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
