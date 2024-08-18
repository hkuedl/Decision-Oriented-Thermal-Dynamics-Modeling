# Decision-Oriented-Thermal-Dynamics-Modeling

_This work proposes a decision-oriented modeling method for building thermal dynamics. The model parameters are updated through an end-to-end gradient-based training strategy wherein the downstream optimization is used as the loss function._

Codes for Paper "Decision-Oriented Modeling of Thermal Dynamics within Buildings".

Authors: Xueyuan Cui, Jean-François Toubeau, François Vallée, and Yi Wang.

## Requirements
Python version: 3.8.17

The must-have packages can be installed by running
```
pip install requirements.txt
```

## Experiments
### Data
All the data for experiments can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1bLnuXXj0fGOjxzkPeCBybAkFgsqIVPYU?usp=drive_link).

### Reproduction
To reproduce the experiments of the proposed method, please run
```
cd Codes/
python Archive_Model_ODE_OPT_Grad.py
```
To reproduce the experiments of generating initial models and comparisons, please run
```
cd Codes/
python Archive_Model_ODE_then_OPT.py
```
Note: There is NO multi-GPU/parallelling training in our codes. 

The models and logs are saved in ```Results/Archive_NNfile/```, and the case results are saved in ```Results/Archive_Results/```.

Please refer to ```readme.md``` in each fold for more details.

## Citation


## Acknowledgments
Package ```Codes/torchdiffeq1/``` is modified based on the open code of [Neural ODE](https://github.com/rtqichen/torchdiffeq). The rapid development of this work would not have been possible without this open-source package. 
