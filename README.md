# HR-SiamRPNplusplus


This is an PyTorch implementation of [SiamRPN++ (CVPR2019)](https://arxiv.org/pdf/1812.11703.pdf), using HRnet as the backbone. We **train** the model on ILSVRC2015_VID dataset with **multi-GPUs**, and use **LMDB** data format to speed up the data loading.

## Details of HR-SiamRPN++
We use HRnet as the backbone and mutiple FPNs to fuse feature maps with different solutions at different depths of the model. Fused features are then fed to SiamRPN for tracking.

## Requirements
Ubuntu 14.04

Python 3.7

PyTorch 1.01


## Training Instructions

```
# 1. Download training data. In this project, we provide the downloading and preprocessing scripts for ILSVRC2015_VID dataset. Please download ILSVRC2015_VID dataset (86GB). The cripts for other tracking datasets are coming soon.
cd data
wget -c http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz
tar -xvf ILSVRC2015_VID.tar.gz
rm ILSVRC2015_VID.tar.gz
cd ..

# 2. Preprocess data.
chmod u+x ./preprocessing/create_dataset.sh
./preprocessing/create_dataset.sh

# 3. Pack the preprocessed data into LMDB format to accelerate data loading.
chmod u+x ./preprocessing/create_lmdb.sh
./preprocessing/create_lmdb.sh

# 4. Start the training.
chmod u+x ./train.sh
./train_hrnet.sh
```
