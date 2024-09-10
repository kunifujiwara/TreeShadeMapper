![Logo](https://github.com/kunifujiwara/TreeShadeMapper/blob/main/images/logo.png)

[![PyPi version](https://img.shields.io/pypi/v/tree-shade-mapper.svg)](https://pypi.python.org/pypi/tree-shade-mapper)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fUcqN6aSLGZnzzahIZiy_AkigFn5gY2e?usp=sharing)
[![License: CC BY-SA 4.0](https://licensebuttons.net/l/by-sa/4.0/80x15.png)](https://creativecommons.org/licenses/by-sa/4.0/)

## Overview
Repository for the script used for estimating and mapping the shading effect of trees considering transmittance of tree Canopies from panoramic imagery, developed by the [Urban Analytics Lab (UAL)](https://ual.sg/) at the National University of Singapore (NUS).

You can read more about this project on [its website](https://ual.sg/publication/***/) too.

The journal paper can be found [here](https://doi.org/***). 

The method integrates semantic segmentation and binariation to calculate transmittace of tree canopies and estimates the sky view factor and solar irradiance using the calculated transmittance.

<p align="center">
  <img src="images/method.png" width="400" alt="method">
</p>

The potential use cases include high-resolution mapping of the sky view factor and solar irradiance and walk route evaluation considering sunlight exposure.

<p align="center">
  <img src="images/usecases.png" width="800" alt="usecases">
</p>

## Installation of `tree_shade_mapper`

```bash
$ pip install tree_shade_mapper
```
## Installation of `pytorch` and `torchvision`

Since `tree_shade_mapper` uses `pytorch` and `torchvision`, you may need to install them separately. Please refer to the [official website](https://pytorch.org/get-started/locally/) for installation instructions.

## Usage

### Data preparation
You need to create a dataset integrating microclimate data with street-level and satellite imagery. Please refer to the sample files in the "data" directory. Note that these sample files do not contain observed real data, but rather virtual data created to demonstrate the proper data structure.

![data](images/examle_datapoint.jpg)

#### Config file

The detailed settings, e.g., hyperparameters of the model and paths to dataset files, are specified in a config file. Please refer to 'configs/sample.yaml'.

### Training
```
python train.py --config path/to/config
```
### Test
```
python test.py --config path/to/config --model path/to/model --result path/to/result
```

## License

`tree_shade_mapper` was created by Kunihiko Fujiwara. It is licensed under the terms of the CC BY-SA 4.0.

## Citation

Please cite the [paper](https://doi.org/XXX) if you use `tree_shade_mapper` in a scientific publication:

XXX

```bibtex
@article{2024_XXX,
 author = {XXX, XXX},
 doi = {XXX},
 journal = {XXX},
 pages = {XXX},
 title = {XXX},
 volume = {XXX},
 year = {XXX}
}
```
