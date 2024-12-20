# PIDPlus Training
Training of our model for IFT6757 "Duckietown" class project PIDPlus. While this page will help you run the training of our models, this will also allow you to run the gym-duckietown simulator and our script to generate the dataset we used in training.

The training of our models are done in Jupyter Notebooks. You can find one for the [CNN](learning/distance/model.ipynb) and another for the [RNN](learning/RNN_distance/RNN.ipynb)

## How to use it

### 1 Basic requirements
For this project, you will need these requirements
- Linux File System 
- Docker 
- Duckietown Shell
- Conda (Or miniconda)
  - For installing miniconda on linux, you can use these commands
    ```
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash ~/Miniconda3-latest-Linux-x86_64.sh
    ```
    Do the install steps, select a default install location and initialization options. After that, you can re-open a terminal window or use this command to refresh the current terminal.
    ```
    source ~/.bashrc
    ```
### 2. Create Virtual Environment
You will then create your virtual environment with conda for being able to run everything. This will install most of the dependencies and is necessary to run Gym-Duckietown in general. Run this command at the root of this repository. This might take a while.
```
conda env create -f environment.yaml
```

You can then simply activate your newly created environment using 
```
conda activate gym-duckietown
```

### 3. Install extra dependencies
There are some extra dependencies needed for running everything. You will first need to do 
```
pip install -e .
```
After that there is some versions that need to be specified, so simply run this command : 
``` 
pip install numpy==1.22 pyglet==1.5.15 Pillow
```
Finally, we use conda to install the last of the dependencies, running
```
conda install -c conda-forge libstdcxx-ng
```
### 4. Importing dataset
Before training a model, you will need a dataset ! We've made the one we created available [here](https://1drv.ms/u/s!AmxJyID0MPIzlZ1PKpnpfFRp1i37Pw?e=6nEsEP). Simply download it, and place it in the repository `learning/distance/road_images/trail1` for the CNN, and `learning/RNN_distance/training_images/trail2` for the RNN.

### 5. Run the training
After installing the necessary dependencies, and importing the dataset, you can simply run the cells one by one to train a new model.
You can find the respective notebooks here : [CNN](learning/distance/model.ipynb) and [RNN](learning/RNN_distance/RNN.ipynb)
