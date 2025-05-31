from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as fh:
            return fh.read()
    return "🌙 Advanced Multi-Agent AI Orchestration Framework"

# Read requirements from requirements.txt
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(req_path):
        with open(req_path, "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    
    # Full requirements list - everything included by default
    return [
        "openai>=1.61.1",
        "cloudpickle>=3.1.1", 
        "playwright>=1.50.0",
        "rich>=13.9.4",
        "sympy>=1.13.3",
        "python-dotenv>=1.0.1",
        "python-docx>=1.1.2",
        "PyPDF2>=3.0.1",
        "urllib3>=2.3.0",
        "beautifulsoup4>=4.13.3",
        "requests>=2.32.3",
        "pandas>=2.2.3",
        "numpy>=2.2.2",
        "pillow>=10.4.0",
        "html2text>=2024.2.26",
        "markdown>=3.7",
        "python-pptx>=1.0.2",
        "crawl4ai>=0.4.248",
        "mcp>=1.9.1",
    ]

setup(
    name="moonlight",
    version="0.1.0",
    author="ecstra",
    author_email="themythbustertmb@gmail.com",
    description="🌙 Advanced Multi-Agent AI Orchestration Framework",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/ecstra/moonlight",
    packages=find_packages(),
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
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    include_package_data=True,
    keywords=[
        "ai", "agents", "workflow", "automation", "multi-agent",
        "artificial-intelligence", "llm", "gpt", "claude", "deepseek",
        "orchestration", "hive", "orchestra", "deepsearch", "mcp", "openai"
    ],
    zip_safe=False,
)