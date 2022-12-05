# Detector-Free Weakly Supervised Group Activity Recognition

### [Dongkeun Kim](https://dk-kim.github.io/), [Jinsung Lee](https://cvlab.postech.ac.kr/lab/members.php), [Minsu Cho](https://cvlab.postech.ac.kr/~mcho/), [Suha Kwak](https://suhakwak.github.io/)

### [Project Page](http://cvlab.postech.ac.kr/research/DFWSGAR/) | [Paper](https://arxiv.org/abs/2204.02139)

## Overview
This work introduces a detector-free approach for weakly supervised group activity recognition. 


## Citation
If you find our code or paper useful, please consider citing our paper:

    @InProceedings{Kim_2022_CVPR,
    author    = {Kim, Dongkeun and Lee, Jinsung and Cho, Minsu and Kwak, Suha},
    title     = {Detector-Free Weakly Supervised Group Activity Recognition},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {20083-20093}
    }

## Requirements

- Ubuntu 16.04
- Python 3.8.5
- CUDA 11.0
- PyTorch 1.7.1

## Conda environment installation
    conda env create --file environment.yml

    conda activate gar
    
## Install additional package
    sh scripts/setup.sh
    
## Download dataset
- Volleyball dataset <br/>
Download Volleyball dataset from:   <br/> 
https://drive.google.com/file/d/1DaUE3ODT_H5mBFi8JzOVBNzVldxfbPbX/view?usp=sharing      
Dataset should be located following the file structure described below. <br/>

- NBA dataset <br/>
The dataset is available upon request to the authors of 
  "Social Adaptive Module for Weakly-supervised Group Activity Recognition (ECCV 2020)". 
  

## Download trained weights
    sh scripts/download_checkpoints.sh

## Run test scripts

- Volleyball dataset (Merged 6 class classification)  

        sh scripts/test_volleyball_merged.sh

- Volleyball dataset (Original 8 class classification)   

        sh scripts/test_volleyball.sh

- NBA dataset  

        sh scripts/test_nba.sh


## Run train scripts

- Volleyball dataset (Merged 6 class classification)
    
        sh scripts/train_volleyball_merged.sh

- Volleyball dataset (Original 8 class classification)
    
        sh scripts/train_volleyball.sh

- NBA dataset
    
        sh scripts/train_nba.sh



## File structure

│── Dataset/ <br/>
│   │── volleyball/ <br/>
│   │    └── videos/ <br/>
│   │── NBA_dataset/ <br/>
│   │    └── videos/ <br/>
│   │    └── train_video_ids <br/>
│   │    └── test_video_ids <br/>
│── checkpoints/ <br/>
│── scripts/ <br/>
│── dataloader/ <br/>
│── models/ <br/>
│── util/ <br/>
train.py <br/>
test.py <br/>
README.md <br/> 
environment.yml <br/>


## Acknowledgement
This work was supported by the NRF grant and the IITP grant funded by Ministry of Science and ICT, Korea (NRF-2021R1A2C3012728, NRF-2018R1A5A1060031, IITP-2020-0-00842, IITP-2021-0-00537, No. 2019-0-01906 Artificial Intelligence Graduate School Program-POSTECH). 

