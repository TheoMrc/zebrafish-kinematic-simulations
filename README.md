# Zebrafish Procrustes Analysis

This repository contains Python code for the segmentation of zebrafish larvae shapes as a pre-processing to the reconstruction of a 3D model of the larvae bending over time with translations and rotations subtracted, during an escape response triggered by an electrical stimuli.

## Project Structure

The main code is located in the `video_preprocessing/experiment.py` file. This file contains two classes: `Video` and `Frame`, which are used to process and analyze video frames.
Currently the processing is executed from the main.py in the project's root folder.

## Setup

To set up the project, follow these steps:

1. Clone the repository to your local machine.
2. Create a clean venv with python 3.10.x
3. Install the required Python packages using pip:
    ```
    pip install -r requirements.txt
    ```
4. Run the `main.py` script after adapting the path to source and target folders.

## Example of processing

