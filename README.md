# Sounds interesting

[This repository](https://github.com/BetelhemNebebe/Sounds_interesting) provides the implementation of the sounds interesting project. The goal of the project is to use video-audio data to generate human-like scan paths by matching these modalities.

This work is based on:
- [ Neural Visual Attention (NeVA)](https://github.com/SchwinnL/NeVA/tree/main): a model that generates human-like scanpaths in a top-down manner.
- [AudioCLIP](https://github.com/AndreyGuzhov/AudioCLIP/tree/master): an extension of the CLIP model that handles audio in addition to text and images.

## Examples


## Downloading Pre-Trained Weights

The pretrained weight should be downloaded and placed under [assets](https://github.com/BetelhemNebebe/Sounds_interesting/tree/master/assets) folder. The path to this file is defined [here](https://github.com/BetelhemNebebe/Sounds_interesting/blob/395affff7ec9e3005f6ad7da29d6e664bbaad5d2/sounds_interesting.py#L30) in the code.

[https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Full-Training.pt](https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Full-Training.pt)

## Set up environment
The required Python version is >= 3.7.

The yml file for the conda environment can be found [here](https://github.com/BetelhemNebebe/Sounds_interesting/blob/master/environment.yml). This environment has all the dependencies needed to run the code. It should be downloaded, created, and activated before running the code.

**Creating the environment:**

`conda env create -f environment.yml`

**Activating the environment:**

`conda activate sounds_interesting`

## Paths that should be defined

- [Path to video data](https://github.com/BetelhemNebebe/Sounds_interesting/blob/d8b2f00e0bed13841522b5f1d311b1b0c429a2e7/sounds_interesting.py#L191)
- [Path for extracted frames](https://github.com/BetelhemNebebe/Sounds_interesting/blob/d8b2f00e0bed13841522b5f1d311b1b0c429a2e7/sounds_interesting.py#L88)
- [Path for extracted audio](https://github.com/BetelhemNebebe/Sounds_interesting/blob/d8b2f00e0bed13841522b5f1d311b1b0c429a2e7/sounds_interesting.py#L128)
- [Path for foveated frames output](https://github.com/BetelhemNebebe/Sounds_interesting/blob/d8b2f00e0bed13841522b5f1d311b1b0c429a2e7/NeVA.py#L109)
- [Path for final output video with scan path](https://github.com/BetelhemNebebe/Sounds_interesting/blob/d8b2f00e0bed13841522b5f1d311b1b0c429a2e7/sounds_interesting.py#L284)

**Note:** It is only necessary to specify the path to the input video data, as the paths for the extracted frames, audios, and outputs can be automatically created as defined in the code.

## Running Code

`python sounds_interesting.py`
