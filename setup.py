from setuptools import setup, find_packages

setup(
    name="minigpt",
    version="0.1.0",
    description="A pure NumPy implementation of a MiniGPT model",
    author="Elsoro Technologies",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.20.0",
        "tqdm>=4.60.0",
        "requests>=2.25.0",  # For downloading data
        "regex>=2022.1.18",  # For GPT-4 tokenizer
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
    },
    python_requires=">=3.8",
)
