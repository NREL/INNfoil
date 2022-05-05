## INNfoil - Invertible Neural Networks for Airfoil Design

This repository contains code for the model described in 

Glaws, A., King, R. N., Vijayakumar, G., & Ananthan, S. (2022). Invertible Neural Networks for Airfoil Design. AIAA Journal, 1-13.

___

The problem of airfoil inverse design, in which an engineer specifies desired performance characteristics and seeks a shape that satisfies those requirements, is fundamental to aerospace engineering. These design workflows traditionally rely on adjoint-based CFD methods that are computationally expensive and have only been demonstrated on steady-state flows. Surrogate-based approaches can accelerate this process by learning cheap forward mappings between airfoil shapes and outputs of interest. However, these workflows must still be wrapped in some optimization- or Bayesian-based inverse design process. In this work, we propose to leverage emerging invertible neural network (INN) tools to enable to rapid inverse design of airfoil shapes. INNs are deep learning models that are architected to have a well-defined inverse mapping that shares model parameters between the forward and inverse passes. When trained appropriately, the resulting INN surrogate model is capable of forward prediction of aerodynamic and structural quantities for a given airfoil shape and inverse recovery of airfoil shapes with specified aerodynamic and structural characteristics.

The invertible neural network (INN) model is built using Python and TensorFlow. The code is accompanied by a YML file `INNfoil_env.yml` that can be used to set up an appropriate conda environment for running the code. The `main.py` file contains an example script for loading data, training the model, and running the inversion process. The `INNfoil.py` file contains the INN model with functionality to run the model in forward and inverse directions. The `model` directory contains all piece necessary to load a pretrained version of the INN.


#### Acknowledgments
This work was authored by the National Renewable Energy Laboratory, operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. Funding provided by [applicable Department of Energy office and program office, e.g., U.S. Department of Energy Office of Energy Efficiency and Renewable Energy Solar Energy Technologies Office (spell out full office names; do not use initialisms/acronyms)]. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.
