from setuptools import setup, find_packages

setup(
    name="writehere_planner",
    version="0.1.0",
    description="A generic Recursive DAG Agentic framework based on the WriteHERE engine",
    author="Extracted from WriteHERE",
    packages=find_packages(),
    install_requires=[
        "loguru",
        "dill",
        # Add other dependencies like openai, requests, etc., based on what's used in utils/llm if needed
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
