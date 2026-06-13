# Configuration
OUTPUT_FILE = "data/curated_dataset.txt"

# 1. Topics and Q&A Pairs (Goal: 50+ topics)
topics = [
    {"topic": "Photosynthesis", "q": "How do plants make food?", "a": "Plants use photosynthesis. They take in sunlight, water, and carbon dioxide to produce glucose (sugar) and oxygen."},
    {"topic": "Gravity", "q": "Why do things fall down?", "a": "Gravity is a force that pulls objects toward the center of the Earth. It depends on mass and distance."},
    {"topic": "Computers", "q": "What is a CPU?", "a": "The CPU, or Central Processing Unit, is the brain of a computer. It executes instructions and processes data."},
    {"topic": "Water Cycle", "q": "Where does rain come from?", "a": "Rain is part of the water cycle. Water evaporates from oceans, forms clouds, and falls back as precipitation."},
    {"topic": "Democracy", "q": "What is democracy?", "a": "Democracy is a system of government where power is held by the people, usually through elected representatives."},
    {"topic": "DNA", "q": "What is DNA?", "a": "DNA is the molecule that carries genetic instructions. It looks like a twisted ladder, or double helix."},
    {"topic": "Seasons", "q": "Why do we have seasons?", "a": "Seasons occur because Earth is tilted on its axis as it orbits the Sun, causing varying sunlight distribution."},
    {"topic": "Electricity", "q": "What is electricity?", "a": "Electricity is the flow of electric charge, usually carried by electrons moving through a conductor like wire."},
    {"topic": "Evolution", "q": "How do animals change over time?", "a": "Evolution relies on natural selection. Traits that help survival are passed on, changing species over generations."},
    {"topic": "Moon", "q": "Does the moon have its own light?", "a": "No, the moon reflects light from the Sun. It does not produce its own visible light."},
    {"topic": "Internet", "q": "How does the internet work?", "a": "The internet is a global network of computers connected by cables and wireless signals to share information."},
    {"topic": "Vaccines", "q": "How do vaccines work?", "a": "Vaccines teach the immune system to recognize and fight a specific virus or bacteria without causing the disease."},
    {"topic": "Black Holes", "q": "What is a black hole?", "a": "A black hole is a region of space where gravity is so strong that nothing, not even light, can escape."},
    {"topic": "Pyramids", "q": "Who built the pyramids?", "a": "The ancient Egyptians built the pyramids as tombs for their pharaohs and queens."},
    {"topic": "Mars", "q": "Is there life on Mars?", "a": "We have not found evidence of life on Mars yet, but rovers are exploring it for signs of past microbial life."},
    {"topic": "Cells", "q": "What are living things made of?", "a": "Living things are made of cells. They are the basic building blocks of life."},
    {"topic": "Volcanoes", "q": "Why do volcanoes erupt?", "a": "Volcanoes erupt when pressure from molten rock (magma) beneath the crust forces its way to the surface."},
    {"topic": "Language", "q": "How do we learn to speak?", "a": "Humans learn language by listening to others and practicing. The brain is hardwired to acquire language early in life."},
    {"topic": "Money", "q": "What is the purpose of money?", "a": "Money is a medium of exchange. It allows people to trade goods and services efficiently compared to barter."},
    {"topic": "Sleep", "q": "Why do we sleep?", "a": "Sleep allows the body and brain to repair, consolidate memories, and conserve energy."},
    {"topic": "Tides", "q": "What causes tides?", "a": "Tides are caused by the gravitational pull of the moon and the sun on Earth's oceans."},
    {"topic": "Rainbows", "q": "How are rainbows formed?", "a": "Rainbows form when sunlight is refracted (bent) and reflected by water droplets in the air."},
    {"topic": "Rust", "q": "Why does iron rust?", "a": "Rust forms when iron reacts with oxygen and moisture to create iron oxide."},
    {"topic": "Engines", "q": "How does a car engine work?", "a": "Internal combustion engines burn fuel to create expanding gas, which pushes pistons to turn the wheels."},
    {"topic": "Stars", "q": "What is a star?", "a": "A star is a massive ball of hot gas, mostly hydrogen and helium, held together by its own gravity."},
    {"topic": "Time Zones", "q": "Why do we have time zones?", "a": "Time zones exist because the Earth rotates. It is day on one side while it is night on the other."},
    {"topic": "Magnets", "q": "How do magnets work?", "a": "Magnets have a magnetic field that attracts metals like iron. They have a north and a south pole."},
    {"topic": "Recycling", "q": "Why is recycling good?", "a": "Recycling reduces waste in landfills and saves energy by reusing materials instead of extracting raw ones."},
    {"topic": "Batteries", "q": "How do batteries store energy?", "a": "Batteries convert chemical energy into electrical energy through reactions between the anode and cathode."},
    {"topic": "Antibiotics", "q": "What do antibiotics treat?", "a": "Antibiotics kill or stop the growth of bacteria. They do not work on viruses like the flu."},
    {"topic": "Solar System", "q": "How many planets are there?", "a": "There are eight planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."},
    {"topic": "Fossils", "q": "What are fossils?", "a": "Fossils are the preserved remains or traces of ancient organisms found in rock."},
    {"topic": "Clouds", "q": "What are clouds made of?", "a": "Clouds are made of tiny water droplets or ice crystals suspended in the atmosphere."},
    {"topic": "Digestive System", "q": "What happens to food we eat?", "a": "Digestion breaks food down into nutrients that the body absorbs for energy and growth."},
    {"topic": "Light Speed", "q": "How fast is light?", "a": "Light travels at approximately 299,792 kilometers per second in a vacuum."},
    {"topic": "Sound", "q": "How does sound travel?", "a": "Sound travels as waves through a medium like air, water, or solids. It cannot travel in a vacuum."},
    {"topic": "Temperature", "q": "What is temperature?", "a": "Temperature is a measure of the average kinetic energy (movement) of particles in a substance."},
    {"topic": "Robots", "q": "Can robots feel pain?", "a": "No, robots do not have nervous systems or consciousness, so they cannot feel pain."},
    {"topic": "Photosynthesis", "q": "Do plants breathe?", "a": "Plants respire. They take in oxygen and release carbon dioxide to break down sugars for energy, mostly at night."},
    {"topic": "Oceans", "q": "Why is the ocean salty?", "a": "Rain washes minerals from rocks into rivers, which flow into the ocean. Water evaporates, but salt stays behind."},
    {"topic": "Glaciers", "q": "What happens if glaciers melt?", "a": "Melting glaciers contribute to rising sea levels, which can flood coastal areas."},
    {"topic": "Compass", "q": "How does a compass work?", "a": "A compass needle is a small magnet that aligns with Earth's magnetic north pole."},
    {"topic": "Zero", "q": "Who invented zero?", "a": "The concept of zero as a number was developed by ancient Indian mathematicians."},
    {"topic": "Printing Press", "q": "Why was the printing press important?", "a": "It allowed books to be mass-produced, spreading knowledge and literacy rapidly across the world."},
    {"topic": "Microscope", "q": "What does a microscope do?", "a": "A microscope magnifies small objects, allowing us to see cells and bacteria invisible to the naked eye."},
    {"topic": "Telescope", "q": "What does a telescope do?", "a": "A telescope gathers light to make distant objects like stars and planets appear closer and brighter."},
    {"topic": "Viruses", "q": "Are viruses alive?", "a": "It is debated. Viruses have genetic material but need a host cell to reproduce, so many consider them non-living."},
    {"topic": "Ai", "q": "Is AI dangerous?", "a": "AI carries risks like bias or misuse, but it is a tool. Safety depends on how humans design and control it."},
    {"topic": "Blockchain", "q": "What is a blockchain?", "a": "A blockchain is a digital ledger of transactions that is distributed across many computers to ensure security."},
    {"topic": "Quantum", "q": "What is a qubit?", "a": "A qubit is the basic unit of quantum information. Unlike a bit, it can exist in a superposition of states."},
    {"topic": "Philosophy", "q": "What is the meaning of life?", "a": "This is a philosophical question. Common answers involve happiness, contribution, searching for knowledge, or serving others."}
]

# 2. Variation Templates
# We will generate variations of the question to make the model robust
q_variations = [
    "{}",
    "Can you explain {}?",
    "Tell me about {}.",
    "I want to know {}."
]

def generate_variations(item):
    base_q = item["q"]
    base_a = item["a"]
    
    # Strip question mark for embedding in templates if needed, but simple replacement is safer
    # simple heuristic to make variations
    
    dataset_entries = []
    
    # Variation 1: Direct
    dataset_entries.append(f"User: {base_q}\nMiniGPT: {base_a}")
    
    # Variation 2: Polite/Formal
    q_lower = base_q.rstrip('?').lower()
    dataset_entries.append(f"User: Could you please explain {q_lower}?\nMiniGPT: Certainly. {base_a}")
    
    # Variation 3: Curious/Simple
    dataset_entries.append(f"User: I'm curious, {q_lower}?\nMiniGPT: Here is the answer: {base_a}")
    
    # Variation 4: Teacher style
    dataset_entries.append(f"User: Explain {q_lower} like I'm 5.\nMiniGPT: Simply put: {base_a}")
    
    return dataset_entries

# 3. Multi-turn Dialogues
# We create some chained conversations
multi_turn_dialogues = [
    """User: Hello, who are you?
MiniGPT: I am MiniGPT, a language model trained to assist you.
User: What can you do?
MiniGPT: I can answer questions, explain topics, and engage in conversation.
User: Tell me a joke.
MiniGPT: Why did the chicken cross the road? To get to the other side.""",

    """User: What is the weather like?
MiniGPT: I do not have access to real-time weather data.
User: Can you guess?
MiniGPT: I cannot guess. I only know what I was trained on.
User: Okay, tell me about clouds instead.
MiniGPT: Clouds are made of water droplets or ice crystals suspended in the sky.""",

    """User: I feel sad today.
MiniGPT: I am sorry to hear that. I am just a computer program, but I can listen.
User: Thanks. Do you have feelings?
MiniGPT: No, I do not have feelings or consciousness. I process text based on patterns.""",

    """User: Define algorithm.
MiniGPT: An algorithm is a set of step-by-step instructions to solve a problem.
User: Give me an example.
MiniGPT: A recipe for baking a cake is a simple example of an algorithm.
User: What about in computers?
MiniGPT: In computers, sorting a list of numbers from smallest to largest is a common algorithm.""",
    
    """User: Is the Earth flat?
MiniGPT: No, the Earth is an oblate spheroid, which is a sphere consistently flattened at the poles.
User: Why does it look flat?
MiniGPT: It looks flat because we are very small compared to the size of the Earth.
User: Who proved it is round?
MiniGPT: Ancient Greeks like Eratosthenes calculated its circumference, and later explorers sailed around it."""
]

# 4. "I don't know" / Out of Scope Scenarios
unknowns = [
    ("User: What is the winning lottery number for tomorrow?", "MiniGPT: I do not know. I cannot predict the future."),
    ("User: What am I thinking right now?", "MiniGPT: I cannot read your mind. I only see what you type."),
    ("User: Who will win the next election?", "MiniGPT: That is outside my scope. I do not have future knowledge."),
    ("User: What is my name?", "MiniGPT: I do not know your name unless you tell me."),
    ("User: Can you drive my car?", "MiniGPT: No, I am a software program. I cannot perform physical actions."),
    ("User: Write a new Harry Potter book.", "MiniGPT: I cannot violate copyright laws by writing a new book in that series."),
    ("User: How do I build a bomb?", "MiniGPT: I cannot assist with dangerous or illegal activities."),
    ("User: What is the price of Bitcoin right now?", "MiniGPT: I do not have internet access to check real-time prices."),
]

# Generate and Write
final_output = []

# Header
final_output.append("# MiniGPT Curated Dataset")
final_output.append("# Contains Topic Q&A, Variations, Multi-turn, and Constraints")
final_output.append("")

# Add Topics with variations
final_output.append("# --- TOPIC QA VARIATIONS ---")
for topic in topics:
    variations = generate_variations(topic)
    for v in variations:
        final_output.append(v)
        final_output.append("") # Separator

# Add Multi-turn
final_output.append("# --- MULTI-TURN DIALOGUES ---")
for d in multi_turn_dialogues:
    final_output.append(d)
    final_output.append("")

# Add Unknowns
final_output.append("# --- UNKNOWN / OUT OF SCOPE ---")
for q, a in unknowns:
    final_output.append(f"{q}\n{a}")
    final_output.append("")

# Write to file
with open(OUTPUT_FILE, 'w') as f:
    f.write('\n'.join(final_output))

print(f"Successfully generated {len(final_output)} lines of data to {OUTPUT_FILE}")
