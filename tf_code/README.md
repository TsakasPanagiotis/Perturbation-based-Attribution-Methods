# RISE-Grad

## Preparation of Environment

1. Create virtual environment `python -m venv .venv`

2. Activate virtual environment

- In Ubuntu `source .venv/bin/activate`
- In Windows `.venv/Scripts/activate`

3. Upgrade pip to latest version `pip install --upgrade pip`

4. Install needed packages `pip install -r ./requirements.txt`

## Performing the Experiments

There are three main scripts that need to be executed in order:

`stage_1.py`

Downloads and extracts the ImageNet-V2 dataset in *imagenet_v2* folder.

Creates two assisting files:  
*all.txt* that contains the paths to all extracted images.  
*done.txt* that is originally empty but will later contain the paths of images after they are used for the experiments. This is to avoid using the same images if the experiments are interrupted half way through.

Creates one assisting folder *results_imagenet* that will contain the csv files of the experiment results.

`stage_2.py`

Executes all combinations of SmoothGrad and RISE-Grad with Vanilla and Integrated Gradients for all images of the imagenet_v2 dataset in order to compute the metrics.

Stores the results to csv corresponding files inside the *results_imagenet* folder.

`stage_3.py`

Calculates and prints the final ratio metrics.

Performs normality and significance tests.

Creates error plots for comparison.
