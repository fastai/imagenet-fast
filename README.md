# End to end imagenet training

1. Spin up an AWS instance
2. Train imagnet
3. Save weights and loss

### AWS
AWS code located in python_scripts/aws_setup.py

To setup a new VPC with EFS and EBS volumes: python_scripts/end2end_create_all.ipynb

To create new spot instance and mount EFS: python_scripts/end2end_demo.ipynb