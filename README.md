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
To reproduce the experiments in the paper, please run
```
cd experiments/
bash run_experiments.sh
```
Note: There is NO multi-GPU/parallelling training in our codes. 

The results(models) and logs will be saved into new folders i.e. ```results/``` and ```logs/``` under ```experiments/```.

Then, go into ```experiments/02-post_analysis.ipynb``` for post analysis, including the post-coordination for benchmarks and the comparison of the total consistency error. Please replace the variable ```pth``` in the notebook.

If you also want to have the same figures in the paper, please refer to ```experiments/03-plot.ipynb```.

### Example
An example of seamless multi-resolution forecasts on wind power from Hierarchical Neural Laplace(HNL) and benchmarks.

![image info](./figs/display.png)

## Citation


## Acknowledgments
Package ```torchlapalace/``` is modified based on the open code of [Neural Laplace](https://github.com/samholt/NeuralLaplace). The rapid development of this work would not have been possible without this open-souce package. 
