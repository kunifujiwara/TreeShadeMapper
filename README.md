![Logo](https://github.com/kunifujiwara/TreeShadeMapper/blob/main/images/logo.jpg)

[![PyPi version](https://pypi.python.org/pypi/tree-shade-mapper.svg)](https://pypi.python.org/pypi/tree-shade-mapper)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fUcqN6aSLGZnzzahIZiy_AkigFn5gY2e?usp=sharing)
[![License: CC BY-SA 4.0](https://licensebuttons.net/l/by-sa/4.0/80x15.png)](https://creativecommons.org/licenses/by-sa/4.0/)

## Overview
Repository for the script used for estimating and mapping the shading effect of trees considering transmittance of tree Canopies from panoramic imagery, developed by the [Urban Analytics Lab (UAL)](https://ual.sg/) at the National University of Singapore (NUS).

You can read more about this project on [its website](https://ual.sg/publication/***/) too.

The journal paper can be found [here](https://doi.org/***). 

The method integrates semantic segmentation and binariation to calculate transmittace of tree canopies and estimates the sky view factor and solar irradiance using the calculated transmittance.

<img src="images/concept.png" width="300" alt="concept">

The potential use cases include high-resolution mapping of the sky view factor and solar irradiance and walk route evaluation considering sunlight exposure.

<img src="images/usecases.png" width="600" alt="usecases">

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

`Microclimate Vision` was created by Kunihiko Fujiwara. It is licensed under the terms of the CC BY-SA 4.0.

## Citation

Please cite the [paper](https://doi.org/10.1016/j.scs.2024.105733) if you use `Microclimate Vision` in a scientific publication:

Fujiwara, K., Khomiakov, M., Yap, W., Ignatius, M., & Biljecki, F. (2024). Microclimate Vision: Multimodal prediction of climatic parameters using street-level and satellite imagery. Sustainable Cities and Society, 105733. doi:[10.1016/j.scs.2024.105733](https://doi.org/10.1016/j.scs.2024.105733)

```bibtex
@article{2024_scs_microclimate_vision,
 author = {Fujiwara, Kunihiko and Khomiakov, Maxim and Yap, Winston and Ignatius, Marcel and Biljecki, Filip},
 doi = {10.1016/j.scs.2024.105733},
 journal = {Sustainable Cities and Society},
 pages = {105733},
 title = {Microclimate Vision: Multimodal prediction of climatic parameters using street-level and satellite imagery},
 volume = {114},
 year = {2024}
}
```
