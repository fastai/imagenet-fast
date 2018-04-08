# Run Instances on AWS

### Downloading ImageNet
1. Open and run download_imagenet.ipynb `jupyter notebook`

Note: You can observe the script output by SSH'ing into the ec2 instance. `get_ssh_command(instance)` should output the command . 
Inside the instance, run: `tmux a -t sess`

### Setting up AWS Instances
AWS wrapper functions located in aws/aws_setup.py

To setup a new VPC with EFS and EBS volumes: python_scripts/end2end_create_all.ipynb
To create new spot instance and mount EFS: python_scripts/end2end_demo.ipynb

### Running Imagenet
1. `python run_script.py -p myproject -i p3.2xlarge -ami ami-b67711ce --use-fastai -sargs '-a resnet50 -j 7 --epochs 100 -b 128 --loss-scale 128 --fp16 --world-size 8'`
2. This should launch an instance. SSH into the box, run tmux -a. You should start to see output