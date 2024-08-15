# Decision-Oriented-Thermal-Dynamics-Modeling

Codes for Paper "Decision-Oriented Modeling of Thermal Dynamics within Buildings".

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
python Archive_Model_ODE_then_OPT
```
Note: There is NO multi-GPU/parallelling training in our codes. 

The models and logs are saved in fold ```Results/Archive_NNfile/```, and the case results are saved in ```Results/Archive_Results/```.

Please refer to ```readme.md``` in each fold for more details.

## Citation


## Acknowledgments
Package ```Codes/torchdiffeq1/``` is modified based on the open code of [Neural ODE](https://github.com/rtqichen/torchdiffeq). The rapid development of this work would not have been possible without this open-souce package. 
