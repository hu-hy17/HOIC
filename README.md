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

1. Download training and testing dataset from https://drive.google.com/file/d/1XyljeKuDSE0WSV5NPXF02GRkDFUjeMeQ/view?usp=sharing . Extract `SingleDepth` folder under `sample_data` folder.

2. Download pre-trained weights from https://drive.google.com/file/d/1mNWecKFdkqeuz_3bUqWbWSlFVfH6tcDI/view?usp=sharing. Extract `motion_im` folder under `results` folder.

3. After the first two steps, your project structure looks like this:

   ```
   HOIC
   ├── assets
   ├── config
   ├── InferenceServer
   ├── mujoco210
   │   ├── bin
   │   ├── include
   │   ├── model
   │   └── sample
   ├── results
   │   └── motion_im
   │       ├── banana_future5_light_add_geom
   │       ├── bottle_future5_light_add_geom
   │       └── box_future5_light_add_geom
   ├── sample_data
   │   └── SingleDepth
   │       ├── Banana
   │       ├── Bottle
   │       └── Box
   ├── scripts
   └── uhc
   ```

3. Run the following command under **the project root** folder:

   ```Shell
   python eval_handmimic.py --cfg box_future5_light_add_geom --epoch 4000
   ```

    where `box_future5_light_add_geom` is the config for `Box` object, it can be replaced with `bottle_future5_light_add_geom` and `banana_future5_light_add_geom`
   
   

## Training

1. Download training and testing dataset from . Extract `SingleDepth` folder under `sample_data` folder. (The same as testing)

2. Run the following command under **the project root** folder:

   ```Shell
   python train_hand_mimic.py --cfg box_future5_light_add_geom --num_threads 32 --gpu_index 0
   ```

   where the option `--num_threads` and option `--gpu_index` can be set depending on your hardware, but currently this project only support one GPU for training.

   By default, the checkpoint will be save under 

   **Note:** Currently this project only support **single thread training on** **Windows** platform, which is very slow (There is something wrong with `multiprocessing` library on Windows). For higher speed, please use **Linux** platform for training.



## Citation

```latex
@article{hu2024hand,
  title={Hand-Object Interaction Controller (HOIC): Deep Reinforcement Learning for Reconstructing Interactions with Physics},
  author={Hu, Haoyu and Yi, Xinyu and Cao, Zhe and Yong, Jun-Hai and Xu, Feng},
  journal={arXiv preprint arXiv:2405.02676},
  year={2024}
}
```



## Acknowledgement

- This repository is built upon [UHC: Universal Humanoid Controller](https://github.com/ZhengyiLuo/UHC)

