from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
    return tokenizer, model

def generate_text(tokenizer, model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    model_name = "microsoft/phi-2"
    test_prompt = "Once upon a time"

    print(f"Loading model: {model_name}")
    tokenizer, model = load_model(model_name)

    print(f"Generating text from prompt: {test_prompt}")
    generated_text = generate_text(tokenizer, model, test_prompt)
    print(f"Generated Text: {generated_text}")

if __name__ == "__main__":
    main()
