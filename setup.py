from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as fh:
            return fh.read()
    return "Lightweight AI Agents SDK for building intelligent automation systems"

# Read requirements from requirements.txt
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req_path):
        with open(req_path, "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    
    # Fallback requirements list
    return [
        "httpx==0.28.1",
        "pydantic==2.12.5",
        "requests==2.32.5"
    ]

setup(
    name="moonlight-ai",
    version="0.2.6",
    author="ecstra",
    author_email="themythbustertmb@gmail.com",
    description="Lightweight AI Agents SDK for building intelligent automation systems",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/ecstra/moonlight",
    packages=find_packages(include=["moonlight", "moonlight.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    include_package_data=True,
    package_data={
        "": ["*.json", "*.md"]
    },
    keywords=[
        "ai", "agents", "automation", "sdk", "llm", "minimal",
        "artificial-intelligence", "openai", "deepseek", "groq",
        "openrouter", "google", "together", "hugging-face"
    ],
    zip_safe=False,
)