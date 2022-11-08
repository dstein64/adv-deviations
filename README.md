# Deviations in Representations Induced by Adversarial Attacks

This repository contains the code for [Deviations in Representations Induced by Adversarial
Attacks](http://arxiv.org/abs/2211.03714).

<div align="center">
  <img src="https://github.com/dstein64/media/blob/main/adv-deviations/plot.svg?raw=true" width="720"/>
</div>

Reported running times are approximate, intended to give a general idea of how long each step will
take. Estimates are based on times encountered while developing on Ubuntu 22.04 with hardware that
includes an AMD Ryzen 9 3950X CPU, 64GB of memory, and an NVIDIA TITAN RTX GPU with 24GB of memory.
The intermediate results utilize about 29 gigabytes of storage.

### Requirements

The code was developed using Python 3.10 on Ubuntu 22.04. Other systems and Python versions may
work, but have not been tested.

Python library dependencies are specified in [requirements.txt](requirements.txt). Versions are
pinned for reproducibility.

### Installation

- Optionally create and activate a virtual environment.

```shell
python3 -m venv env
source env/bin/activate
```

- Install Python dependencies, specified in `requirements.txt`.
  * 4 minutes

```shell
pip3 install -r requirements.txt
```

### Running the Code

By default, output is saved to the `./workspace` directory, which is created automatically.

- Train a ResNet classification model.
  * 1 hour

```shell
python3 src/train_net.py
```

- Evaluate the model, extracting representations from the corresponding data.
  * 1 minute

```shell
python3 src/eval_net.py
```

- At each layer, calculate pairwise distances between representations, for normalization.
  * 2 hours

```shell
python3 src/calc_pairwise_distances.py
```

- Adversarially perturb test images, evaluating and extracting representations from the
  corresponding data.
  * 7 hours

```shell
python3 src/attack.py
```

- Calculate the distances between representations for original images and their adversarially
  perturbed counterparts.
  * 3 minutes

```shell
python3 src/calc_distances.py
```

- Analyze data by first normalizing and then plotting and tabulating.
  * 3 seconds

```shell
python3 src/analyze.py
```

### Citation

```
@misc{steinberg2022deviations,
  doi = {10.48550/ARXIV.2211.03714},
  url = {https://arxiv.org/abs/2211.03714},
  author = {Steinberg, Daniel and Munro, Paul},
  title = {Deviations in Representations Induced by Adversarial Attacks},
  publisher = {arXiv},
  year = {2022},
  eprint={2211.03714},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
