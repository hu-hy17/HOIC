**Note: This repository is under building!!!**

## Introduction

This is the official implementation of paper:

**Hand-Object Interaction Controller (HOIC): Deep Reinforcement Learning for Reconstructing Interactions with Physics**



## Installation

1. Clone repository and create **Conda** (https://www.anaconda.com/) environment

   ```shell
   git clone https://github.com/hu-hy17/HOIC.git
   conda create -n HOIC python=3.8
   conda activate HOIC
   ```

2. Install physics simulator **Mujoco210** (https://mujoco.org/)

   - For Windows:

     - Download from https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-windows-x86_64.zip
     - Extract `mujoco210` folder to the root folder of this project.

   - For Linux:

     - Download from https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz

     - Create `.mujoco` folder under `/home` and extract `mujoco210` folder into `.mujoco` 

     - Edit`~/.bashrc` and add the following environmental variables:

       ```Shell
       export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/{username}/.mujoco/mujoco210/bin
       export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
       ```

     - You may need to install GL libraries if you meet this error: `fatal error: GL/gl.h: No such file or directory`

       ```Shell
       sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
       ```

3. Install **Pytorch** (https://pytorch.org/)

   ```Shell
   conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
   ```

4. Install other dependencies

   ```Shell
   pip install -r requirements.txt
   ```



## Testing



## Training