
import os
import json

def create_clean_training_data():
    """Create training data without format leakage"""
    print("Generating clean training data...")
    
    clean_data = [
        # Clean Q/A pairs - NO FORMAT MARKERS
        {"instruction": "Answer the question.", "input": "Who are you?", "output": "I am MiniGPT, a compact on-device language model."},
        {"instruction": "Explain the concept.", "input": "What is attention?", "output": "Attention is a mechanism in neural networks that helps the model focus on important parts of the input."},
        
        # Dialogue format
        {"instruction": "Have a conversation.", "conversation": [
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": "I am MiniGPT. How can I help you today?"}
        ]},
        
        # Instruction following
        {"instruction": "Answer the question concisely.", "input": "What is machine learning?", "output": "Machine learning is training computers to learn patterns from data."},
        
        # Expanded Examples for robust training
        {"instruction": "Explain.", "input": "Why is the sky blue?", "output": "The sky is blue because of Rayleigh scattering."},
        {"instruction": "Define.", "input": "What is a neural network?", "output": "A neural network is a computational model inspired by the human brain."},
        {"instruction": "General knowledge.", "input": "What is the capital of France?", "output": "The capital of France is Paris."},
        {"instruction": "Logic.", "input": "If A is B, and B is C, what is A?", "output": "A is C."},
    ]
    
    # Replicate for volume (Overfitting is desired for this "Concept Learning" phase)
    # We want the model to deeply memorize this clean format to overwrite the old "Definition:" bias.
    clean_data = clean_data * 100 
    
    # Convert to plain text without format markers
    training_text = ""
    for item in clean_data:
        if "conversation" in item:
            for turn in item["conversation"]:
                training_text += f"{turn['role'].title()}: {turn['content']}\n"
            training_text += "\n"
        else:
            # Standard Chat Format: User: ... Assistant: ...
            # We standardize on "User:" and "Assistant:" to avoid leakage of "Q:" "Definition:" etc.
            training_text += f"User: {item['input']}\nAssistant: {item['output']}\n\n"
    
    os.makedirs("data", exist_ok=True)
    with open("data/train_clean.txt", "w") as f:
        f.write(training_text)
        
    print(f"Clean training data saved to data/train_clean.txt ({len(clean_data)} samples)")

if __name__ == "__main__":
    create_clean_training_data()
