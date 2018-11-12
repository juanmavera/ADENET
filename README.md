# ADENET.py

This code have been used in the paper "Towards End-to-End Acoustic Localization using Deep Learning: from Audio Signal to Source Position Coordinates" in order to obtain the results.

## Requisites
```
python 2.7
keras
tensorflow-gpu
numpy
wave
struct
scipy
matplotlib
```

## Run the software
The given code is set to train with "synthetic" data, generated from [Albayzin Corpus](http://catalogue.elra.info/en-us/repository/browse/ELRA-S0089/), and fine tuning and testing on [IDIAP room](http://www.idiap.ch/dataset/av16-3/) recordings.

There is a main class named ADENET which configures everything related to the proposed architecture. There are also two functions included in this class, which facilitate the training and fine tuning processes. Two other functions are also included to facilitate the testing process, both on "synthetic" data and on data recorded in the real room. 

For more information about any procedure, read this [paper](http://www.mdpi.com/1424-8220/18/10/3418)

### ADENET architecture

<img style= "center" src="https://github.com/juanmavera/ADENET/blob/master/images/Adenet_architecture.png" />

## Authors
* **Juan Manuel Vera-Diaz**
* **Daniel Pizarro**
* **Javier Macias-Guarasa**
