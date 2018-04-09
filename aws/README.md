# Run Instances on AWS

### Downloading ImageNet
1. Open and run download_imagenet.ipynb `jupyter notebook`

Note: You can observe the script output by SSH'ing into the ec2 instance. `get_ssh_command(instance)` should output the command . 
Inside the instance, run: `tmux a -t sess`

### Setting up AWS Instances
AWS wrapper functions located in aws/aws_setup.py

To setup a new VPC with EFS and EBS volumes: python_scripts/end2end_create_all.ipynb
To create new spot instance and mount EFS: python_scripts/end2end_demo.ipynb

### Running Fastai on Imagenet
1. `python run_script.py -p myproject -i p2.xlarge -ami ami-b67711ce --use-fastai -sargs "-sargs '-a resnet50 -j 7 --epochs 100 -b 128 --loss-scale 128 --fp16 --world-size 8' -multi"`
2. This should launch an instance. SSH into the box, run tmux -a. You should start to see output

#### Pipeline:
Passing `--use-fastai` as an argument uploads the script `train_imagenet.sh` . 
It then runs the script with the supplied arguments `-sargs '-a resnet50 -j 7 --epochs 1 -b 128 --loss-scale 128 --fp16 --world-size 8`, `-multi` and `-p myproject`

`upload_scripts/train_imagenet.sh` updates the instance and runs `imagenet-fast/imagenet_nv/fastai_imagenet.py` with the exact same arguments that you passed in above . 
Check the header for `fastai_imagenet.py` to figure out what to pass in for `-sargs`

### Running Imagenet with nvidia
1. Use the same `run_script.py` as above, but pass in `--use-nvidia`
2. This will upload and run the script `upload_scripts/train_nv.sh` with supplied arguments
3. `train_nv.sh` will in turn run `imagenet_nv/main.py`


### Running `fastai_imagnet.py`
Example usage: `python fastai_imagenet.py ~/data/imagenet -a resnet50 -j 7 --epochs 1 -b 128 --loss-scale 128 --fp16 --save-dir ~/data/test_saving`