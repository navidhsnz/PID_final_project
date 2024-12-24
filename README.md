This is our final project for IFT6757 "Duckietown" class at UdeM.
Please refer to the project report for a complete explanation of the project. This repository only contains a part of the project.
The project report is available [here](Final_Project_Report_Navid_and_Simon.pdf)
# PIDPlus Training and Simulation
This repository contains everything you need to train the models and run the simulation. All the code that concerns model implementainos and their training are in `learning`. The simulation is `run_simulation.py` which runs the simulation and also is used for creating datsets. For more detailed explanations on reproducing the results of our experiment including training the models and running the simulator, please continue reading.
![](media/CNN1_1.png)
![](media/RNN3_2.png)
## Videos
Here is our real world results: [here](https://www.youtube.com/watch?v=c7_-a6bwr1I&list=PL9zoqgzvQ0ABEr8rfXTN7xGBawXYTd_H6)

Here is a video of the simulation for model CNN1: [here](https://www.youtube.com/watch?v=rJtUea_6V4s&list=PL9zoqgzvQ0ABEr8rfXTN7xGBawXYTd_H6&index=5)

Here is a video of the simulation for model CNN2: [here](https://www.youtube.com/watch?v=UVOseRL8ieA&list=PL9zoqgzvQ0ABEr8rfXTN7xGBawXYTd_H6&index=5&pp=iAQB)

Here is a video of the simulation for model RNN1: [here](https://www.youtube.com/watch?v=lfG0cgGci4E&list=PL9zoqgzvQ0ABEr8rfXTN7xGBawXYTd_H6&index=6&pp=iAQB)

Here is a video of the simulation for model RNN2: [here](https://www.youtube.com/watch?v=e02h_0ez7BQ&list=PL9zoqgzvQ0ABEr8rfXTN7xGBawXYTd_H6&index=7&pp=iAQB)

Here is a video of the simulation for model RNN3: [here](https://www.youtube.com/watch?v=tXs06yWc4Pc&list=PL9zoqgzvQ0ABEr8rfXTN7xGBawXYTd_H6&index=2&pp=iAQB)

Here is a video of the simulation for model RNN4: [here](https://www.youtube.com/watch?v=PqAEIR1alOY&list=PL9zoqgzvQ0ABEr8rfXTN7xGBawXYTd_H6&index=3&pp=iAQB)


## Datasets and weights
Here are our datasets we created and some weights for each of our models : 

Dataset : [here](https://1drv.ms/u/s!AmxJyID0MPIzlZ1eQO8Wp9isMPlmOg?e=r5OoxG)

Weights : [here](https://1drv.ms/u/s!AmxJyID0MPIzlZ1Wola6EJrP-KDuyA)
## How to train a model

### 1 Basic requirements for the simulator
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

### 3. Installations and resolving extra dependencies
There are some extra dependencies needed for running everything. You will first need to install gym-dcukietown with:
```
pip install -e .
```
After that there is some versions that need to be specified, so simply run this command : 
``` 
pip install numpy==1.22 pyglet==1.5.15 Pillow torchsummary tqdm
```
Finally, we use conda to install the last of the dependencies, running
```
conda install -c conda-forge libstdcxx-ng
```
It is expected too see some conflicting version of dependencies while running these. But after running these commands you should be set to run the simulator !

### 4. Importing dataset
Before training a model, you will need a dataset ! We've made the one we created available [here](https://1drv.ms/u/s!AmxJyID0MPIzlZ1eQO8Wp9isMPlmOg?e=r5OoxG). Simply download it and extract it. You should place the folder named `datasets` in `learning/`. We have three datasets (`trail1`, `trail2`, and `trail3`). In our experiments we used `trail2`. `trail1` is a smaller version of `trail2`, and `trail3` contains data of clock-wise rotation of the robot, which is not used in our project. The second and third ones are for the RNN since they have `actions` folder, but they still work without any trouble for the CNN training. After adding the datasets, the content of the `learning/` folder should include three folders, namely `CNN`, `RNN`, and `datasets`.

### 5. How to train the models?
After installing the necessary dependencies, and importing the dataset, you can simply run the training by going into the `learning/CNN` or `learning/RNN` folder and running

```
python MODEL_NAME.py
```
MODEL_NAME here is either `CNN1`, `CNN2`, `RNN1`, `RNN2`, `RNN3`, `RNN4` which are all of our different model we tested.

These files contain the code needed to train the models. During training, they will produce some snapshots of the weights of the model as backup. After training is done, they produce a final weight file. Create a folder named `models` inside `learning/CNN/` and another one with the same name inside `learning/RNN`. Put the final weights of the model inside of this folder. ie. the respective weights of the models for CNN and RNN should be placed in `learning/CNN/models` and `learning/RNN/models`. After this, you can run the `Evaluate_CNN{1,2}.ipynb` or `Evaluate_RNN{1,2,3,4}.ipynb` to evaluate the models predictions on some sample images from the dataset. Note that for these notebooks, you will need both the datsets as explained in step 4 above, as well as the weights. After running the training code, two text files will also be produced. These include the training and validation loss of the models during thraining. The notebook `loss_graphs.ipynb` draws the corresponding loss graphs using these text files.

### 6. Import weights
Training the models takes a while. If you want to import the weigths of the models we have already trained, follow the following steps:
Download this [folder](https://1drv.ms/u/s!AmxJyID0MPIzlZ1Wola6EJrP-KDuyA) and extract it. This folder contains two subfolders, namely `CNN` and `RNN`. In each one, there are the weights of the trained models. Copy these weights and place them in either `learning/CNN/models` for the CNN models or `learning/RNN/models` for the RNN models.


## Running a model in simulation

For testing a model in simulation, make sure you have trained one or imported the weights following the previous instructions.
The weights are expected to be either in `learning/CNN/models` or in `learning/RNN/models`. The simulation file is called `run_simulation.py`.
Also, you will need all the dependencies.
You can run a model in simulation with the command 
```
python run_simulation.py --model_type MODEL_TYPE
```
MODEL_TYPE here is `CNN1`, `CNN2`, `RNN1`, `RNN2`, `RNN3` or `RNN4`

Once the simulation is loaded up, you can control the robot with arrow keys. If you want to enable the PID contorl so that the robot uses the predictions to follow the lane, press `SPACE`. Pressing it again will disengage the PID.

In the simulation, you have access to multiple controls from the keyboard. Here is a complete list: 
- `SPACE` for engaging/disengaging the PID
- `o` for increasing kp
- `p` for decreasing kp
- `k` for increasing kd
- `l` for decreasind kd
- `n` for increasing speed
- `b` for decreasing speed
- `s` for image capturing (This is for creating datssets. It records images and their labels every few miliseconds and creats a datset.)
- `ESCAPE` for closing
- `SHIFT`  to increase the speed with arrow keys (1.5x)
