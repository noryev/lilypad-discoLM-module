from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    tokenizer = AutoTokenizer.from_pretrained("huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1")
    model = AutoModelForCausalLM.from_pretrained("huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1")

  prompt = "Translate the following English text to French: 'Hello, how are you?'"

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=50, num_return_sequences=5)
    
    print("Generated texts:")
    for i, output in enumerate(outputs):
        print(f"{i}: {tokenizer.decode(output, skip_special_tokens=True)}")

if __name__ == "__main__":
    main()
