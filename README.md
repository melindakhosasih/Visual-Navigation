# Visual Navigation

## Installation

1. Create conda env

        conda env create -f environment.yaml
        conda activate vinav

2. Install habitat sim

        conda install habitat-sim==0.2.5 withbullet -c conda-forge -c aihabitat

3. Install habitat lab

        cd ..
        mkdir habitat
        cd habitat
        git clone --branch stable https://github.com/facebookresearch/habitat-lab.git habitat-lab-v0.2.5
        cd habitat-lab-v0.2.5
        git checkout tags/v0.2.5
        pip install -e habitat-lab
        cd ../../Visual-Navigation/

4. Install PyTorch

        conda install pytorch=2.0.1 torchvision=0.15.2 torchaudio=2.0.2 cudatoolkit=11.8 cuda-nvcc -c pytorch -c nvidia/label/cuda-11.8.0


5. Install Opencv

        pip install opencv-python-headless

## Train
- Train ddpg

        python train.py --title test --algo ddpg

- Train sac

        python train.py --title test --algo sac

## Demo

    python env.py