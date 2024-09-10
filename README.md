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

### Tree shade calculation and mapping
```
from tree_shade_mapper import get_tree_shade

base_dir = '/path/to/your/input/directory'
calc_type = 'map'

# Define start and end time, and the interval
time_start = '2024-01-01 07:00:00'
time_end = '2024-01-01 20:00:00'
interval = '5min'

# Define time zone and location
time_zone = 'Asia/Singapore'
latitude = 1.29751
longitude = 103.77012

# Define varibles for visualization
vmin = 0
vmax = 1200
resolution = 14

get_tree_shade(base_dir, time_start, time_end, interval, time_zone, latitude, longitude, calc_type=calc_type, vmin=vmin, vmax=vmax, resolution = resolution)
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
