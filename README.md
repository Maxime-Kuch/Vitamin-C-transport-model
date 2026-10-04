# Vitamin-C-transport-model
The repository contains the code and description for mathematical model of cellular vitamin C transport across intestinal wall, featured in manuscript of the articla "Fluxes of vitamin C across the intestinal wall - the mathematical model".
# Abstract
....
# Model architecture
The model is divided into two parts: one serving as a validation on cell culture results and the second being the model of intestinal wall, integrating mucus layer, enterocyte and interstitial space. The validation model is restricted only to intra- and extracellular compartments, where extracellular vitamin C concnetration time profile has been implemented from Modrzejewska *et al.* [1] work after fitting with exponential function. The intracellular conentration results from [1] were later used for model validation. The second model analyzes bidiractional fluxes of vitamin C through the intestinal wall and constitutes of three finite compartments enclosed by infinite intestinal lumen compartment and infinite blood compartment, where chosen concentration time dependence functions are applied. The main model describing differential equations is supplemented with fuctions responsible for analysis and generation of results featured in the manuscript.
# Reproducting the results
In order to reproduce the results presented in the manuscript run the corresponding scripts from the root directory, the results will be automatically saved in the `results/` folder. Prior to running any smulations install the requirements:
```bash
pip install -r requirements.txt
```
**Model validation - figure 1 and figure 2**
```bash
python scripts/validation_figure1_figure2.py
```
**Intestinal wall transport - figure 3 and figure 4**
```bash
python scripts/Intestinal_model_figure3_figure4.py
```
**The mucus impact - figure 5**

Here the simulations might take longer, mainly due to optimal mucus dose dependency plot, the script runs a substantial amount of simulations to produce the result. Thus 
here the number of samples in the plot was reduced for a faster result with lower quality.
```bash
python scripts/Intestinal_model_figure5.py
```
In order to obtain higher quality results sampling in the original function generating the plot can be increased. To do that the values of variables ```samples_conc``` and ```samples_thic``` in the ```main()``` function of `scripts/Intestinal_model_figure5.py` can be increased. In the original figure for the article manuscript ```samples_conc=70``` and ```samples_thic=70``` were used. In these settings the simulation took around 30 min.
# References
1. M. Modrzejewska, et al., Vitamin C enhances substantially formation of 5-hydroxymethyluracil in cellular DNA. Free Radic. Biol. Med. 101, 378–383 (2016).
