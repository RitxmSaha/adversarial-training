# Import necessary libraries
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model constant
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Load model and tokenizer
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2"  # Enable Flash Attention 2
    )
    return model, tokenizer

# Function to get a response from the model
def get_model_response(chat_history, model=None, tokenizer=None):
    """
    Generate a response from the model based on chat history.
    
    Args:
        chat_history (list): List of dictionaries with 'role' and 'content' keys
                            Example: [{"role": "user", "content": "Hello"}, ...]
        model: The loaded model (will be loaded if None)
        tokenizer: The loaded tokenizer (will be loaded if None)
        
    Returns:
        str: The model's response
        model: The loaded model for reuse
        tokenizer: The loaded tokenizer for reuse
    """
    # Load model and tokenizer if not provided
    if model is None or tokenizer is None:
        model, tokenizer = load_model()
    
    # Format chat history for the model
    formatted_prompt = tokenizer.apply_chat_template(
        chat_history,
        tokenize=False
    )
    
    # Generate response
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the response
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return response, model, tokenizer

def print_parquet_columns(file_path="data/kodcode-3k/test.parquet", max_length=80):
    df = pd.read_parquet(file_path)
    columns = df.columns.tolist()
    print(" ".join(columns))

def run_test_from_parquet():
    print("Loading model...")
    model, tokenizer = load_model()
    
    # Load questions from parquet file
    print("Loading questions from parquet file...")
    try:
        df = pd.read_parquet("data/kodcode-3k/test.parquet")
        
        # Check if 'prompt' column exists
        if 'prompt' not in df.columns:
            print(f"\nColumn 'prompt' not found.")
            print_parquet_columns("data/kodcode-3k/test.parquet")  # Print columns if 'prompt' not found
            return
        
        # Get the first 5 questions
        questions = df['prompt'].head(5).tolist()
        
        # Test each question
        for i, question in enumerate(questions, 1):
            print(f"\n\n--- Question {i} ---")
            print(f"Question: {str(question)}")  # Convert to string before printing
            
            # Extract the user content from the dictionary format
            if isinstance(question, list):
                # Find the user message in the list
                for msg in question:
                    if msg.get('role') == 'user':
                        user_content = msg.get('content', '')
                        test_chat = [{"role": "user", "content": user_content}]
                        break
                else:
                    # If no user message found, use the last message
                    user_content = question[-1].get('content', '') if question else ''
                    test_chat = [{"role": "user", "content": user_content}]
            elif isinstance(question, dict):
                # Handle dictionary format
                user_content = question.get('content', '')
                test_chat = [{"role": "user", "content": user_content}]
            else:
                # Treat as regular string
                test_chat = [{"role": "user", "content": str(question)}]
            
            print("Generating response...")
            response, model, tokenizer = get_model_response(test_chat, model, tokenizer)
            
            print(f"Response: {response}")
            
    except FileNotFoundError:
        print("Error: File 'data/kodcode-3k/test.parquet' not found.")
    except Exception as e:
        print(f"Error loading or processing parquet file: {str(e)}")

if __name__ == "__main__":
    # Uncomment the line below to just print columns without running the model
    #print_parquet_columns()
    
    # Run the full test
    run_test_from_parquet()
