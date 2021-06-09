# Quantifying Peers Contribution in FL and P2PL
Machine Learning and Optimization Laboratory - EPFL - 2021

Project entirely developed by Frédéric **Berdoz**, Master student in Computational Science at EPFL.

Supervised by Prof. Martin **Jaggi** and Dr. Mary-Anne **Hartley** from EPFL.

## Context
The objective of this project is to design contribution measure that can be applied to both the federated learning setting  (FL) and the peer-to-peer learning setting (P2PL).


## Content

* ```Income_FL.ipynb```: Python notebook in which the project was developed for the FL setting.

* ```Income_P2PL.ipynb```: Python notebook in which the project was developed for the P2PL setting.

* ```report.pdf```: The report that was written on the basis on this code.

* ```Figures.ipynb```: Python notebook in which the Figures of ```report.pdf```were made.

* ```helpers.py```: Python script containing all the functions that are needed in ```Income_FL.ipynb``` and ```Income_P2PL.ipynb```.

* ```models.py```: Python script containing the model class that is used in ```Income_FL.ipynb``` and ```Income_P2PL.ipynb```.

* ```visualization.py```: Python script containing the different visualization functions that are used in ```Income_FL.ipynb``` and ```Income_P2PL.ipynb```. These functions are here to help analysing the learning phase.

* ```data``` folder: Folder were the data is stored (empty on git).

* ```figures``` folder: Folder were the figures for ```report.pdf``` are stored.

* ```saves``` folder: Folder were the outputs (variable, plots, etc.) of ```Income_FL.ipynb``` and ```Income_P2PL.ipynb``` are stored. It contains several subfolder that are named after the corresponding experiment.



## Prerequisite

This code was developed using ```python 3.7.7``` along with the libraries:
* ```numpy 1.18.1``` (Contribution Measures manipulation)
* ```pytorch 1.4.0``` (Machine Learning)
* ```pandas 1.0.3``` (Data loading and preprocessing)
* ```torchvision 0.5.0``` (Dataset class)
* ```matplotlib 3.1.3``` (Visualization)
* ```scikit-learn 0.22.1``` (Label encoding)
* ```dill 0.3.2``` (Saving and loading workspaces)






