from setuptools import setup, find_packages # type: ignore

setup(
    name="EchoWeave",
    version="0.0.1", 
    packages=find_packages(),
    install_requires=[
        "networkx",
        "langchain_community",
        "matplotlib",
        "numpy",
    ],
    author="Jacob Yoder",
    author_email="jayoder25@gmail.com",
    description="A Python package for building, maintaining, and searching knowledge graphs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Novaii-Yoder/EchoWeave",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.12",
)
