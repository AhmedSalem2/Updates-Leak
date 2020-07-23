# Updates-Leak
This repository contains code for our paper: "Updates-Leak: Data Set Inference and Reconstruction Attacks in Online Learning".

We focus on our multi-sample attacks and present them on the MNIST dataset.

# Code usage: 
The first step needed is to generate the training and testing datasets by calling the dataGeneration.py script.

The labelPrediction.py script implements and evaluates our Multi-sample Label Distribution Estimation Attack (A_<sub>LDE</sub>).

The MSR_attack.py script implments our Multi-sample Reconstruction Attack (A_<sub>MSR</sub>).

The MSR_evaluation.py script evaluates our Multi-sample Reconstruction Attack (A_<sub>MSR</sub>). It both calculates the Mean Square Error (MSE), and plot the images. However, plotting works better if the denormalization in the "to_img" function is commented out.


# Citation
If you use this code, please cite the following paper: 
# <a href="https://arxiv.org/abs/1904.01067">Updates-Leak</a>
```
@inproceedings{SBBFZ20,
author = {Ahmed Salem and Apratim Bhattacharya and Michael Backes and Mario Fritz and Yang Zhang},
title = {{Updates-Leak: Data Set Inference and Reconstruction Attacks in Online Learning}},
booktitle = {{USENIX Security Symposium (USENIX Security)}},
publisher = {USENIX},
year = {2020}
}

```
