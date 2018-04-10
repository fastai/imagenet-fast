#!/bin/bash

# Install Pillow-simd for speed
yes | pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

imagenet_dir=~/ILSVRC/Data/CLS-LOC

if [ ! -d $imagenet_dir/val/n01440764 ]; then
    echo "Formatting validation set"
    # Following format here: https://github.com/soumith/imagenet-multiGPU.torch
    cd $imagenet_dir/val
    wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
    cd ~/
else
    echo "Validation set already formatted"
fi

# Create resizing directory
output_dir=$imagenet_dir/resized_output
mkdir $output_dir

# Run Jupyter notebook - resize_images.ipynb
# Resize images for desired sizes
# jupyter notebook  --ip=0.0.0.0 --port=8888 --no-browser

conda install fire -c conda-forge
# Tar individual sizes
cd ~/

yes | sudo apt-get install pigz

# Resize, tar, move to efs
python resize_images.py 80 $imagenet_dir "resized_output"
tar -I pigz -cf $output_dir/imagenet_80.tar.gz $output_dir/80
sudo mv $output_dir/imagenet_80.tar.gz ~/efs_mount

python resize_images.py 160 $imagenet_dir "resized_output"
tar -I pigz -cf $output_dir/imagenet_160.tar.gz $output_dir/160
sudo mv $output_dir/imagenet_160.tar.gz ~/efs_mount

python resize_images.py 320 $imagenet_dir "resized_output"
tar -I pigz -cf $output_dir/imagenet_320.tar.gz $output_dir/320
sudo mv $output_dir/imagenet_320.tar.gz ~/efs_mount

python resize_images.py 375 $imagenet_dir "resized_output"
tar -I pigz -cf $output_dir/imagenet_375.tar.gz $output_dir/375
sudo sudomv $output_dir/imagenet_375.tar.gz ~/efs_mount
