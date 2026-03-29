import os
import random

def generate_synthetic_data(file_path, size_mb=5):
    """Generates a synthetic dataset of User/Assistant pairs for testing MiniGPT scale-up."""
    print(f"Generating {size_mb}MB of synthetic data to {file_path}...")
    
    topics = [
        "quantum physics", "ancient history", "machine learning", "robotics",
        "space exploration", "culinary arts", "music theory", "climate change",
        "renewable energy", "artificial intelligence", "philosophy", "economics",
        "biology", "chemistry", "mathematics", "literature", "poetry", "art history"
    ]
    
    questions = [
        "What is {topic}?", "Can you explain {topic}?", "How does {topic} work?",
        "Tell me a story about {topic}.", "What are the latest advancements in {topic}?",
        "Who are the key figures in {topic}?", "Why is {topic} important?",
        "What are the ethical implications of {topic}?", "How can I learn {topic}?",
        "What is the history of {topic}?", "Give me an example of {topic}.",
        "What are the challenges in {topic}?", "How does {topic} relate to daily life?"
    ]
    
    responses_starts = [
        "That's a fascinating question. {Topic} is a broad field.",
        "To understand {topic}, we must first look at its fundamentals.",
        "{Topic} refers to the study and application of various principles.",
        "In simple terms, {topic} is all about understanding complex systems.",
        "The history of {topic} is rich with innovation and discovery.",
        "Many consider {topic} to be one of the most important disciplines today.",
        "Let me explain {topic} to you. It begins with a basic concept.",
        "When we talk about {topic}, we are generally referring to a specific domain.",
        "An excellent topic. {Topic} has fundamentally changed how we view the world."
    ]
    
    responses_bodies = [
        " Researchers have spent decades analyzing its implications. The core mechanism involves rigorous testing and validation of hypotheses. This leads to breakthroughs that reshape our understanding.",
        " It combines theoretical models with practical applications. For instance, many modern technologies rely heavily on the foundational theories developed in this area.",
        " The primary goal is to optimize efficiency and uncover underlying patterns. By doing so, we can predict future trends and build more robust frameworks.",
        " There is a lot of ongoing debate among experts. Some argue it is the key to future progress, while others urge caution due to potential unintended consequences.",
        " Key advancements include the development of new algorithms, better measurement tools, and a deeper appreciation for interdisciplinary collaboration.",
        " Think of it like a puzzle where every new discovery is a piece. Over time, the picture becomes clearer, revealing incredible complexity and beauty.",
        " Students often find it challenging at first, but with practice, the concepts become intuitive. The key is consistent study and practical application.",
        " Many books and research papers have been dedicated to exploring its depths. The literature is vast, spanning centuries of human thought."
    ]
    
    responses_ends = [
        " I hope this provides a good overview of {topic}.",
        " Would you like to know more about a specific aspect of {topic}?",
        " In conclusion, {topic} continues to be a vibrant and essential area of study.",
        " This is just the tip of the iceberg when it comes to {topic}.",
        " If you have more questions about {topic}, feel free to ask!"
    ]
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    target_bytes = size_mb * 1024 * 1024
    current_bytes = 0
    
    with open(file_path, "w", encoding="utf-8") as f:
        while current_bytes < target_bytes:
            topic = random.choice(topics)
            q = random.choice(questions).format(topic=topic)
            r_start = random.choice(responses_starts).format(topic=topic.capitalize(), Topic=topic.capitalize())
            r_body = random.choice(responses_bodies)
            r_end = random.choice(responses_ends).format(topic=topic)
            
            # Need a longer body to make it more diverse and token-rich
            r_body += random.choice(responses_bodies)
            r_body += random.choice(responses_bodies)
            
            pair = f"User: {q}\nAssistant: {r_start}{r_body}{r_end}\n---\n"
            f.write(pair)
            current_bytes += len(pair.encode('utf-8'))
            
    print("Synthetic dataset generated successfully.")

if __name__ == "__main__":
    generate_synthetic_data("data/synthetic_train.txt", size_mb=10)
    # Also generate a small validation split manually or let the script do it.
