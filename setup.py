from setuptools import setup, find_packages

setup(
    name="apex-track",
    version="0.1.0",
    packages=find_packages(include=["apex", "apex.*"]),
    python_requires=">=3.10",
)
