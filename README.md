# Playing Atari games with Double Deep Q Learning

This is an implementation of the Double Deep Q Learning algorithm, introduced by researchers from Google DeepMind in 2015. This is an off-policy Reinforcement Learning model, capable of dealing with the overestimation issue of the classic Deep Q Learning model.

## Installation

The implementation utilizes OpenAI's Gym library to run the Atari emulation, which can only run properly on a Linux machine. Don't try to run this on Windows :(

Packages this project depends on are listed in the file `requirements.txt`, and can be install system-wide or in a virtual environment with Pip:

```bash
pip3 install -r requirements.txt
```

Also, [GPU support for Tensorflow](https://www.tensorflow.org/install/gpu) should also be installed, so that the model can easily run in real time.

## How to use

Run `python3 main.py` with the following parameters:

-   `-g`, `--game`: Choose from available games. Default is "Breakout"
-   `-m`, `--mode`: Choose from available modes: training, testing. Default is "training".
-   `-tsl`, `--step_per_run_limit`: Choose how many total steps (frames visible by agent) should be performed. Default is 10000.
-   `-trl`, `--total_run_limit`: Choose after how many runs we should stop. Default is None (no limit).
-   `-r`, `--render`: Choose if the game should be rendered. Default is False.
-   `-s`, `--sign_only`: Choose whether we should clip rewards to its sign. Default is True.
