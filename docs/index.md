# QuantGYMM Documentation

- [What is QuantGYMM](#what_is)
- [Getting Started](#getting_started)
- [Instruments and Usages](#usages)

## [What is QuantGYMM](#what_is)

QuantGYMM was born within the group project for the course in Fixed Income and Credit Risk, offered in the Master in Finance Insurance and Risk Management at the Collegio Carlo Alberto. The code was therefore devised to tackle the project task, and later become a structured package. The library is mainly based on ```Pandas```, ```Scipy``` and ```NumPy```. ```QuantGYMM``` is still at embrional stage, and any suggestion or contribution is well accepted. 

## [Getting started](#getting_started)

Before installing 'QuantGYMM', be sure that you have a Python version >= 3.9 installed in you computer/local enviroment. If you are using a conda, I suggest to create a virtual enviroment and install a suitable Python version. 

### Suggested Set Up:

1) Create virtual enviroment and activate it:
```
conda create --name [your_env_name_here] python=3.10
conda activate [your_env_name_here]
```
2) Install Jupyter (optional):
```
conda install jupyter
```
3) Install QuantGYMM:
```
pip install QuantGYMM
```
### Alternative:
1) You could clone the repository:
    
```
git clone https://github.com/GianlucaBroll95/QuantGYMM.git
```
and then install running from within the directory:
```
pip install .
```

## [Instruments and Usages](#usages)

`QuantGYMM` comprises six main modules: `instruments`, `models`, `term_structures`, `calendar`, `pricers` and `utils`. 
### Instruments
At the moment, there are two instruments available: `FloatingRateBond` and `VanillaSwap`. 
### Models
This modules gathers two models for fitting a term structure for simulation. At the moment, `MertonSimulator` allows to simulate the term structure by using Merton model and allowing for risk-neutral and real-world calibration.
