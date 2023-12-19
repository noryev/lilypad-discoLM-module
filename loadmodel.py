from huggingface_hub import snapshot_download
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define the prompt
prompt = "Name the fastest plane on record"

# Download the model
model_path = snapshot_download(repo_id="amgadhasan/phi-2", repo_type="model", local_dir="./phi-2", local_dir_use_symlinks=False)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
torch.set_default_dtype(torch.float16)  # Ensure compatibility with the model
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

# Text generation function
def generate(prompt: str, generation_params: dict = {"max_length": 200}) -> str:
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, **generation_params)
        completion = tokenizer.batch_decode(outputs)[0]
        return completion
    except Exception as e:
        return f"Error generating text: {str(e)}"

# Generate and print the result
result = generate(prompt)
print(result)