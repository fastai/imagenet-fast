#!/bin/bash

# Download imagnet from Kaggle
source activate fastai
pip install kaggle
chmod 600 /home/ubuntu/.kaggle/kaggle.json
kaggle competitions download -c imagenet-object-localization-challenge
tar -xvzf ~/.kaggle/competitions/imagenet-object-localization-challenge/imagenet_object_localization.tar.gz -C ~/

# Save tar to EFS
sudo mv ~/.kaggle/competitions/imagenet-object-localization-challenge/imagenet_object_localization.tar.gz ~/efs_mount_point