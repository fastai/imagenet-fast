#!/bin/bash

# Update Server
sudo dpkg --configure -a
sudo apt-get upgrade

# Long running task
tmux

# Download imagnet from Kaggle
source activate fastai
pip install kaggle
chmod 600 /home/ubuntu/.kaggle/kaggle.json
kaggle competitions download -c imagenet-object-localization-challenge
cd ~/.kaggle/competitions/imagenet-object-localization-challenge
tar -xvzf imagenet_object_localization.tar.gz -C ~/


# Install Pillow-simd for speed
pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

# Format validation set
cd ~/ILSVRC/Data/CLS-LOC/val
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
rm valprep.sh

# Create resizing directory
mkdir ~/ILSVRC/Data/CLS-LOC/resized_output

# Run Jupyter notebook - Resize_images.ipynb
# Resize images for desired sizes
jupyter notebook  --ip=0.0.0.0 --port=8888 --no-browser

# Tar individual sizes
cd ~/ILSVRC/Data/CLS-LOC/resized_output
tar -zcvf imagenet_80.tar.gz 80
tar -zcvf imagenet_160.tar.gz 160
tar -zcvf imagenet_320.tar.gz 320
tar -zcvf imagenet_320.tar.gz 375

# Move them to EFS
mv ~/.kaggle/competitions/imagenet-object-localization-challenge/imagenet_object_localization.tar.gz ~/efs_mount_point
mv ~/ILSVRC/Data/CLS-LOC/resized_output/imagenet_80.tar.gz ~/efs_mount_point
mv ~/ILSVRC/Data/CLS-LOC/resized_output/imagenet_160.tar.gz ~/efs_mount_point
mv ~/ILSVRC/Data/CLS-LOC/resized_output/imagenet_320.tar.gz ~/efs_mount_point
mv ~/ILSVRC/Data/CLS-LOC/resized_output/imagenet_375.tar.gz ~/efs_mount_point