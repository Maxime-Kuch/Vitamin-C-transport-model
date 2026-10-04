# Vitamin-C-transport-model
The repository contains the code and description for mathematical model of cellular vitamin C transport across intestinal wall, featured in manuscript of the articla "Fluxes of vitamin C across the intestinal wall - the mathematical model".
# Abstract
....
# Model architecture
The model is divided into two parts: one serving as a validation on cell culture results and the second being the model of intestinal wall, integrating mucus layer, enterocyte and interstitial space. The validation model is restricted only to intra- and extracellular compartments, where extracellular vitamin C concnetration time profile has been implemented from ... work after fitting with exponential function. The intracellular conentration results from ... were later used for model validation. The second model analyzes bidiractional fluxes of vitamin C through the intestinal wall and constitutes of three finite compartments enclosed by infinite intestinal lumen compartment and infinite blood compartment, where chosen concentration time dependence functions are applied. The main model describing differential equations is supplemented with fuctions responsible for analysis and generation of results featured in the manuscript.
# Reproducting the results
In order to reproduce the results presented in the manuscript run the corresponding scripts from the root directory, the results will be automatically saved in the `results/` folder. Prior to running any smulations install the requirements:
```bash
pip install -r requirements.txt
```
**Model validation - figure 1 and figure 2**
