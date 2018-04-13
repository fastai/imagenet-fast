# Cifar10 training in PyTorch
This implements training of popular model architectures, such as ResNet, AlexNet, and VGG on the ImageNet dataset.

This version has been modified from using the included DataParallel and DistributedDataParallel modules included in pytorch to a custom DistributedDataParallel included in distributed.py.
For description of how this works please see the distributed example included in this repo.

To run multi-gpu on a single node use the command
```python -m multiproc main.py ...```
adding any normal arguments.

## Requirements

- Install PyTorch from source, master branch of ([pytorch on github](https://www.github.com/pytorch/pytorch)
- `pip install -r requirements.txt`
- Download the ImageNet dataset and move validation images to labeled subfolders
    - To do this, you can use the following script: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

## iPython Notebook

Please look at: `cifar10-super-convergence-tuned.ipynb` to run in notebook. It contains similar code to `train_cifar10.py`

## Training

To train a model, run `train_cifar10.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
python train_cifar10.py -a preact_resnet30 [cifar-folder with train and val folders. Otherwise we download automatically]
```


## Usage

```
usage: train_cifar10.py [-h] [--save-dir SAVE_DIR] [--arch ARCH] [-j N]
                        [--epochs N] [--cycle-len N] [-b N] [--lr LR]
                        [--momentum M] [--weight-decay W] [--pretrained]
                        [--fp16] [--use-tta USE_TTA] [--sz SZ]
                        [--use-clr USE_CLR] [--loss-scale LOSS_SCALE] [--prof]
                        [--dist-url DIST_URL] [--dist-backend DIST_BACKEND]
                        [--world-size WORLD_SIZE] [--rank RANK]
                        DIR

PyTorch Cifar10 Training

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --save-dir SAVE_DIR   Directory to save logs and models.
  --arch ARCH, -a ARCH  model architecture: preact_resnet18 | preact_resnet34
                        | preact_resnet50 | resnet56 | resnext29_16_64 |
                        resnext29_8_64 | dpn107 | dpn131 | dpn68 | dpn92 |
                        dpn98 | inceptionresnetv2 | inceptionresnetv2_conc |
                        inceptionv4 | load | load_block17 | load_block35 |
                        load_block8 | load_conv2d | load_conv2d_nobn |
                        load_linear | load_mixed_4a_7a | load_mixed_5 |
                        load_mixed_5b | load_mixed_6 | load_mixed_6a |
                        load_mixed_7 | load_mixed_7a | nasnetalarge |
                        pre_resnet101 | pre_resnet152 | pre_resnet18 |
                        pre_resnet34 | pre_resnet50 | reduce | resnet101 |
                        resnet152 | resnet18 | resnet34 | resnet50 |
                        resnext101 | resnext152 | resnext18 | resnext34 |
                        resnext50 | resnext_101_32x4d | resnext_101_64x4d |
                        resnext_50_32x4d | se_resnet_101 | se_resnet_152 |
                        se_resnet_18 | se_resnet_34 | se_resnet_50 |
                        se_resnet_50_conc | se_resnext_101 | se_resnext_152 |
                        se_resnext_50 | test | test_block17 | test_block35 |
                        test_block8 | test_conv2d | test_conv2d_nobn |
                        test_mixed_4a_7a | test_mixed_5b | test_mixed_6a |
                        test_mixed_7a | wrn_50_2f (default: resnet56)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --cycle-len N         Length of cycle to run
  -b N, --batch-size N  mini-batch size (default: 256)
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)
  --pretrained          use pre-trained model
  --fp16                Run model fp16 mode.
  --use-tta USE_TTA     Validate model with TTA at the end of traiing.
  --sz SZ               Size of transformed image.
  --use-clr USE_CLR     div,pct,max_mom,min_mom. Pass in a string delimited by
                        commas. Ex: "20,2,0.95,0.85"
  --loss-scale LOSS_SCALE
                        Loss scaling, positive power of 2 values can improve
                        fp16 convergence.
  --prof                Only run a few iters for profiling.
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --world-size WORLD_SIZE
                        Number of GPUs to use. Can either be manually set or
                        automatically set by using 'python -m multiproc'.
  --rank RANK           Used for multi-process training. Can either be
                        manually set or automatically set by using 'python -m
                        multiproc'.
```
