# Run Instances on AWS

### Downloading imagenet
1. Open and run download_imagenet.ipynb `jupyter notebook`

Note: You can observe the script output by SSH'ing into the ec2 instance. `get_ssh_command(instance)` should output the command . 
Inside the instance, run: `tmux a -t sess`

### Setting up AWS Instances
AWS wrapper functions located in aws/aws_setup.py

To setup a new VPC with EFS and EBS volumes: python_scripts/end2end_create_all.ipynb

To create new spot instance and mount EFS: python_scripts/end2end_demo.ipynb

### Running cifar10
1. Run `run_cifar10.ipynb`
2. This will upload `run_cifar10.sh` and `cifar10.py` to a new ec2 instance
3. Edit `cifar10` to change training parameters