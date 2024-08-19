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
All the data for experiments can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1bLnuXXj0fGOjxzkPeCBybAkFgsqIVPYU?usp=sharing).

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
```
@ARTICLE{10638763,
  author={Cui, Xueyuan and Toubeau, Jean-François and Vallée, François and Wang, Yi},
  journal={IEEE Transactions on Smart Grid}, 
  title={Decision-Oriented Modeling of Thermal Dynamics Within Buildings}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Buildings;Mathematical models;Optimization;Training;Task analysis;Costs;Accuracy;Building energy management;thermal dynamics;thermostatically controlled loads;neural dynamic equations},
  doi={10.1109/TSG.2024.3445574}}
```

## Acknowledgments
Package ```Codes/torchdiffeq1/``` is modified based on the open code of [Neural ODE](https://github.com/rtqichen/torchdiffeq). The rapid development of this work would not have been possible without this open-source package. 
