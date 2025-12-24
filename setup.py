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
        "openai>=1.61.1",
        "playwright>=1.50.0",
        "rich>=13.9.4",
        "python-dotenv>=1.0.1",
        "python-docx>=1.1.2",
        "PyPDF2>=3.0.1",
        "beautifulsoup4>=4.12.0",
        "requests>=2.32.3",
        "numpy>=2.0.0",
        "pillow>=10.0.0",
        "html2text>=2024.2.26",
        "crawl4ai>=0.4.248",
        "qdrant-client>=1.16.0",
        "tiktoken>=0.12.0",
        "transformers>=4.45.0",
        "nltk>=3.9.0",
        "openpyxl>=3.1.0",
        "xlrd>=2.0.0",
    ]

setup(
    name="moonlight",
    version="0.2.0",
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    include_package_data=True,
    package_data={
        "": ["*.json", "*.md"],
        "moonlight.core.processors": ["nltk_data/**/*"],
        "moonlight.core.token": [".token_cache/.gitkeep"],
    },
    keywords=[
        "ai", "agents", "automation", "sdk", "rag", "llm",
        "artificial-intelligence", "openai", "deepseek", "groq",
        "web-search", "document-processing", "qdrant", "embeddings"
    ],
    zip_safe=False,
)