import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

# Model and tokenizer paths
output_dir = "./results"

# Load the trained model in Spanish
model_name = "NousResearch/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Load the tokenizer for Spanish
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Run text generation pipeline with the trained model
prompt = "formula preguntas de programacion web"
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    max_length=50,
    truncation=True
)
result = pipe(f"[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
