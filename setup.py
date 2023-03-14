from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name = "QuantGYMM",
    version = "0.2",
    author = "Gianluca Broll",
    author_email = "gianluca.broll@gmail.com",
    description = "Package for dealing with bond pricing and hedging.", 
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/GianlucaBroll95/QuantGYMM",
    classifiers = [
    	"Programming Language :: Python :: 3",
    	"License :: OSI Approved :: MIT License"],
    python_requires=">3.9",
    package_dir={"": "src"},
    packages = find_packages(where="src")
    
)
