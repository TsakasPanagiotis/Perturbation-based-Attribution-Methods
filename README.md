# Resilient Perturbation-based Attribution Methods to Adversarial Attacks


### Description

This repository was developed for the Interpretability and Explainability in Artificial Intelligence course of the MSc in Artificial Intelligence at the University of Amsterdam - UvA for the academic year 2023. It contains the scripts for a novel method which is the combination of two methods along with a comparison between other known methods. A technique of whether these methods can be fooled is also included, according to the details of this [paper](https://arxiv.org/pdf/1911.02508.pdf).

### Instructions

In order to run the scripts you should follow the steps below:
1. Clone this repo:

`git clone https://github.com/TsakasPanagiotis/Perturbation-based-Attribution-Methods.git`

2. Make sure to install the environment using the env.yml or the env.job file
3. Download the datasets from [here](https://drive.google.com/file/d/1sXfYSES4B84yucqUK1pWVbMiOu41ruB3/view?usp=sharing)
4. Run the main.job with the args of your preference.

If you want to train the model, use the main.job file and replace the last line with this:

`srun python -u main.py --experiment stl --mode train`

If you want to produce explanations using different methods, use the main.job file and replace the last line with this:

`srun python -u main.py --experiment stl --mode explain`


