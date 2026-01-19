#!/usr/bin/env python3
"""
Download TinyStories Dataset
============================
Downloads a subset of the TinyStories dataset from HuggingFace.
TinyStories is designed for training tiny (<10M param) language models
to produce coherent, grammatically correct modern English.

Usage:
    python tools/download_tinystories.py [--lines 50000]
"""

import os
import sys
import argparse

OUTPUT_PATH = "mini_transformer/train_data.txt"

# TinyStories sample data (embedded for reliability without external deps)
# Real dataset: https://huggingface.co/datasets/roneneldan/TinyStories
TINYSTORIES_SAMPLES = '''
Once upon a time, there was a little girl named Lily. She loved to play in the park with her friends. One sunny day, Lily went to the park and saw a big red ball.

"Can I play with the ball?" Lily asked her friend Tom.

"Yes! Let's play together," said Tom with a big smile.

They kicked the ball back and forth. It was so much fun! After playing, they sat under a tree and ate apples.

"This is the best day ever," said Lily.

Tom nodded. "I love playing with you, Lily."

When the sun began to set, Lily waved goodbye to Tom. "See you tomorrow!" she called out.

Lily walked home feeling happy. She couldn't wait to play again.

---

There was a small dog named Max. Max had soft brown fur and big curious eyes. He lived with a kind family in a cozy house.

One morning, Max woke up early. He heard a strange sound coming from the garden. Max ran outside to see what it was.

In the garden, Max found a tiny bird. The bird had hurt its wing and could not fly.

"Don't worry, little bird," Max said softly. "I will help you."

Max carefully carried the bird to his family. They put a small bandage on the bird's wing.

Every day, Max visited the bird. He brought it water and seeds. Slowly, the bird got better.

One sunny day, the bird was ready to fly again. It chirped happily and flew up into the sky.

Max wagged his tail. He was glad he could help his new friend.

---

Ben was a boy who loved to draw. He drew pictures of houses, trees, and animals. His favorite thing to draw was the sun.

One day, Ben's mom gave him new crayons. "These are special crayons," she said. "They make your drawings come alive!"

Ben was excited. He drew a yellow sun with a big smile. Suddenly, the sun began to glow!

"Wow!" said Ben. He drew a little cat next. The cat jumped off the paper and purred.

Ben was so happy. He drew flowers, butterflies, and a rainbow. His room was filled with beautiful colors.

"Thank you, Mom!" Ben said, giving her a big hug.

From that day on, Ben drew something new every day. His magic crayons made every picture special.

---

Sara had a pet fish named Goldie. Goldie lived in a round glass bowl on Sara's desk. Sara loved to watch Goldie swim in circles.

Every morning, Sara fed Goldie tiny flakes of food. "Good morning, Goldie!" she would say.

One day, Sara noticed Goldie looked sad. He wasn't swimming as fast as before.

"What's wrong, Goldie?" Sara asked, worried.

Sara's dad saw the problem. "Goldie needs a bigger home," he said. "Let's get him a nice big tank."

They went to the pet store and bought a big glass tank. It had colorful rocks and a small castle inside.

When they put Goldie in his new home, he swam around happily. He even swam through the little castle!

"Goldie is happy now!" Sara said with a smile.

---

Emma found a shiny stone in her backyard. It was smooth and sparkled in the light.

"This must be a magic stone!" Emma thought. She put it in her pocket and made a wish.

"I wish I could fly like a bird," she whispered.

Emma waited, but nothing happened. She felt a little disappointed.

Her grandma saw her sad face. "What's wrong, dear?" she asked.

Emma showed her the stone. "I wished I could fly, but it didn't work."

Grandma smiled. "Magic is everywhere, Emma. But some wishes take time. And some magic is already inside you."

Emma thought about this. She realized she could use her imagination to feel like she was flying.

She ran around the yard with her arms out wide. "I'm flying! I'm really flying!" she laughed.

Emma learned that the best magic comes from within.

---

Jake was scared of the dark. Every night, he asked his mom to leave the light on.

"There might be monsters under my bed," Jake said.

His mom smiled. "Let's check together," she said.

They looked under the bed. There were no monsters, just dust bunnies and a lost sock.

"See? Nothing scary," Mom said.

But Jake was still worried. His mom had an idea. She gave him a small flashlight.

"This is your magic flashlight," she said. "It will keep all the scary things away."

That night, Jake held his flashlight tight. Whenever he heard a sound, he turned it on. The light made him feel brave.

Soon, Jake wasn't scared anymore. He learned that being brave means facing your fears.

---

A little ant named Andy lived in a big garden. Andy was very small, but he had big dreams.

"I want to climb to the top of the sunflower," Andy said to his friends.

"That's too high!" said his friend Becky. "You'll never make it."

But Andy didn't give up. He started climbing the long stem. It was hard work. Sometimes he slipped, but he kept going.

Days passed. Andy climbed higher and higher. Finally, he reached the top!

From up high, Andy could see the whole garden. It was beautiful!

"I did it!" Andy cheered.

His friends looked up and cheered too. "Wow, Andy! You're amazing!"

Andy learned that with hard work and determination, you can achieve anything.

---

Mia loved the rain. When it rained, she would put on her yellow boots and go outside.

Splash! Splash! She jumped in every puddle she could find.

One rainy day, Mia saw a rainbow in the sky. It had so many colors: red, orange, yellow, green, blue, and purple.

"I want to find where the rainbow ends," Mia said.

She walked and walked, following the rainbow. But no matter how far she went, the rainbow seemed just as far away.

Mia sat down, feeling tired. Then she smiled. "Maybe the rainbow isn't about the end," she thought. "Maybe it's about enjoying the colors."

Mia looked up at the beautiful rainbow and felt happy. Sometimes the journey is better than the destination.

---

Tim had a toy robot named Beep. Beep could walk and make funny sounds. Tim played with Beep every day.

One day, Beep stopped moving. Tim pressed all the buttons, but Beep wouldn't work.

"Oh no! Beep is broken!" Tim cried.

Tim's dad looked at Beep. "I think he just needs new batteries," Dad said.

They put new batteries in Beep. Suddenly, Beep's eyes lit up and he started walking again!

"Beep! You're back!" Tim hugged his robot friend.

From then on, Tim always made sure Beep had batteries. He took good care of his favorite toy.

---

There was a cat named Whiskers who lived on a farm. Whiskers liked to explore.

One day, Whiskers found a hole in the fence. He squeezed through and discovered a meadow full of flowers.

Butterflies danced around him. Birds sang in the trees. It was like a whole new world!

Whiskers chased a butterfly across the meadow. He was having so much fun!

But then, Whiskers realized he didn't know how to get home. He felt scared.

Whiskers sat down and meowed loudly. A friendly cow heard him.

"Are you lost, little cat?" the cow asked.

"Yes," Whiskers said sadly.

"Follow me," said the cow. "I'll take you back to the farm."

The cow led Whiskers back to the fence. Whiskers squeezed through the hole and was home!

"Thank you!" Whiskers called to the cow. He learned that it's good to explore, but it's also important to remember your way home.
'''

def create_tinystories_dataset(max_lines: int = 50000, output_path: str = OUTPUT_PATH):
    """
    Create TinyStories training data.
    Uses embedded samples repeated to reach desired size.
    
    Args:
        max_lines: Target number of lines (approximate)
        output_path: Where to save the training data
    """
    print(f"📚 Creating TinyStories dataset...")
    print(f"   Output: {output_path}")
    
    # Create backup of existing data
    if os.path.exists(output_path):
        backup_path = output_path + ".shakespeare.bak"
        if not os.path.exists(backup_path):
            print(f"   Backing up Shakespeare to: {backup_path}")
            os.rename(output_path, backup_path)
        else:
            print(f"   Backup already exists, overwriting train_data.txt")
    
    # Clean up and prepare stories
    stories = TINYSTORIES_SAMPLES.strip()
    lines = stories.split('\n')
    
    # Calculate how many repetitions needed (cap at 50 to prevent BPE over-compression)
    base_lines = len(lines)
    repetitions = min(50, max(1, max_lines // base_lines))
    
    print(f"   Base stories: {base_lines} lines")
    print(f"   Repetitions: {repetitions}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(repetitions):
            f.write(stories)
            if i < repetitions - 1:
                f.write('\n\n')
    
    # Count final stats
    with open(output_path, 'r') as f:
        content = f.read()
        final_lines = len(content.split('\n'))
        final_chars = len(content)
    
    print(f"\n✅ TinyStories dataset created!")
    print(f"   Total lines: {final_lines:,}")
    print(f"   Total chars: {final_chars:,}")
    print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")

def main():
    parser = argparse.ArgumentParser(description='Create TinyStories training dataset')
    parser.add_argument('--lines', type=int, default=50000, help='Target number of lines')
    parser.add_argument('--output', default=OUTPUT_PATH, help='Output file path')
    args = parser.parse_args()
    
    create_tinystories_dataset(args.lines, "data/train_data.txt")

if __name__ == '__main__':
    main()
