import collections
import sys

def check_dist(path="mini_transformer/train_data.txt"):
    try:
        with open(path, "r") as f:
            data = f.read()
    except FileNotFoundError:
        print("Data file not found.")
        return

    total = len(data)
    print(f"Total characters: {total}")
    
    counts = collections.Counter(data)
    
    print("\nTop 10 Characters:")
    for char, count in counts.most_common(10):
        readable = char.replace("\n", "\\n").replace(" ", "<SPACE>")
        print(f"'{readable}': {count} ({count/total*100:.2f}%)")

if __name__ == "__main__":
    check_dist()
