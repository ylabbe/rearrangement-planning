Visually Guided Rearrangement Planning
=============

Code for reproducing "Monte-Carlo Tree Search for Visually Guided Rearrangement Planning".

[[arXiv](https://arxiv.org/abs/1904.10348)] [[Project Page](https://ylabbe.github.io/rearrangement-planning/index.html)] [[Video](http://www.youtube.com/watch?v=fS5tTa_Tl1Y)] :


[![Rearrangement Planning](http://img.youtube.com/vi/fS5tTa_Tl1Y/0.jpg)](http://www.youtube.com/watch?v=fS5tTa_Tl1Y)

So far we provide the training and evaluation code along a ResNet-18 pretrained model for extracting object positions from RGB images with a UR5 robot. 

We used the predicted object positions for performing visually guided rearrangement planning but it can be used for various other tasks such as building a pyramid with multiple cubes:

![pyramid](https://drive.google.com/uc?export=view&id=1xuTQ5Os09RQ8P-onxngCqHFfsDUfbzXk)

Synthetic dataset generation, rearrangement planning simulation environment and MCTS planning code will be added later.


## Installation

The code requires PyTorch 1.0.1 and python 3.6+. After installing PyTorch, run:

```
pip install -r requirements.txt
```

### Download data
You may use the code from this repository to predict objects positions relative to the robot on new images using a pretrained model, reproduce the results from the paper on the real evaluation dataset or train a new model on the synthetic training dataset. 
We provide the following data:
* A pretrained model.
* The real evaluation dataset with 1200 images, taken under two different viewpoints (we used the RGB images from a Kinect 1 and Kinect 2 cameras).
* The training dataset that contains 2 million synthetic images (50G).

You can download all (or parts of) the data with:
```
python download.py --model --eval --train
```
Downloading and unpacking the training data may take a while depending on your connection and machine.

### Tests
You can check that you installed the dependencies and downloaded the data correctly by running the tests for the parts of the code you are planning to use:
```
pytest -v rearrangement/test -m 'model or eval or train' --disable-pytest-warnings
```

## Using the pretrained model


![predictions](https://drive.google.com/uc?export=view&id=12NAVqMy29CKny6f9xugr-OZXcwjrxNWk)


We provide a pretrained model for the UR5 robot with the [3-Finger Adaptive Robot Gripper](https://robotiq.com/products/3-finger-adaptive-robot-gripper).
If you have the same configuration, you can use this model to predict the following from one RGB camera pointed at the robot (**without any additionnal calibration**):

 * The 2D coordinates of individual objects in the workspace (a 40cm x 40cm area in front of the robot).
 * The semantic segmentation for the robot and gripper and instance segmentation for the objects.
 * A depth map. (We did not use it in our experiments).

This information can be plugged into standard planning tools to perform various tasks.

For example, you can use the predictions on one image to extract the position of multiple cubes and build a pyramid:


![pyramid](https://drive.google.com/uc?export=view&id=1xuTQ5Os09RQ8P-onxngCqHFfsDUfbzXk)

We provide the notebook [`rearrangement/notebooks/Visualization`](https://github.com/ylabbe/rearrangement-planning/blob/master/rearrangement/notebooks/Visualization.ipynb) that makes predictions and visualize them on frames from the evaluation dataset.
You can use it as a starting point for using the model on your own data. 

You can also see in this notebook how to use AlexNet features for matching the patches from a source and a target image.

## Reproducing the results from the paper


![eval_ds](https://drive.google.com/uc?export=view&id=156_BWOXCfZstbwxthwNCD7jaa33YEFKd)


You can reproduce the numbers from the papers which are reported on the real evaluation dataset by running:
```
python -m rearrangement.eval_on_real --run_id state-prediction-71538 --log_dir data/models
```
This assumes that you have downloaded the data with the flag ``--eval``.

The figures in relation with the vision section and experiments can be reproduced using the notebook [`rearrangement/notebooks/Paper Figures`](https://github.com/ylabbe/rearrangement-planning/blob/master/rearrangement/notebooks/Paper%20Figures.ipynb). 

## Training new models


![train_ds](https://drive.google.com/uc?export=view&id=1WQEhA5DsOHb7iKMp5SjPICdmV4yM9awA)


You can train the equivalent of the model that we provide using the following command:
```
python -m rearrangement.main --ds_root data/datasets/synthetic-shapes-1to6 --save data/models/my-model
```
This assumes that you have downloaded the data with the flag ``--train``.

You can visualize the training and validation losses in the notebook [`rearrangement/notebooks/Training logs`](https://github.com/ylabbe/rearrangement-planning/blob/master/rearrangement/notebooks/Training%20logs.ipynb).


## Citation

If you find the code or the model useful please cite this work:

```
@misc{1904.10348,
Author = {Sergey Zagoruyko and Yann Labbé and Igor Kalevatykh and Ivan Laptev and Justin Carpentier and Mathieu Aubry and Josef Sivic},
Title = {Monte-Carlo Tree Search for Efficient Visually Guided Rearrangement Planning},
Year = {2019},
Eprint = {arXiv:1904.10348},
}
```
