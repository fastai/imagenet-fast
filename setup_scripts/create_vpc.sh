#!/bin/bash

# settings
export name="fast-ai"

export cidr="0.0.0.0/0"
export vpcTag=$name
export sgTag="SG for fast.ai machine"
export subnetTag="$name-subnet"

hash aws 2>/dev/null
if [ $? -ne 0 ]; then
    echo >&2 "'aws' command line tool required, but not installed.  Aborting."
    exit 1
fi

if [ -z "$(aws configure get aws_access_key_id)" ]; then
    echo "AWS credentials not configured.  Aborting"
    exit 1
fi

export vpcId=$(aws ec2 create-vpc --cidr-block 10.0.0.0/28 --query 'Vpc.VpcId' --output text)
aws ec2 create-tags --resources $vpcId --tags --tags Key=Name,Value=$name
aws ec2 modify-vpc-attribute --vpc-id $vpcId --enable-dns-support "{\"Value\":true}"
aws ec2 modify-vpc-attribute --vpc-id $vpcId --enable-dns-hostnames "{\"Value\":true}"

export internetGatewayId=$(aws ec2 create-internet-gateway --query 'InternetGateway.InternetGatewayId' --output text)
aws ec2 create-tags --resources $internetGatewayId --tags --tags Key=Name,Value=$name-gateway
aws ec2 attach-internet-gateway --internet-gateway-id $internetGatewayId --vpc-id $vpcId

export subnetId=$(aws ec2 create-subnet --vpc-id $vpcId --cidr-block 10.0.0.0/28 --query 'Subnet.SubnetId' --output text)
aws ec2 create-tags --resources $subnetId --tags --tags Key=Name,Value=$name-subnet

export routeTableId=$(aws ec2 create-route-table --vpc-id $vpcId --query 'RouteTable.RouteTableId' --output text)
aws ec2 create-tags --resources $routeTableId --tags --tags Key=Name,Value=$name-route-table
export routeTableAssoc=$(aws ec2 associate-route-table --route-table-id $routeTableId --subnet-id $subnetId --output text)
aws ec2 create-route --route-table-id $routeTableId --destination-cidr-block 0.0.0.0/0 --gateway-id $internetGatewayId

export securityGroupId=$(aws ec2 create-security-group --group-name $name-security-group --description $sgTag --vpc-id $vpcId --query 'GroupId' --output text)
# ssh
aws ec2 authorize-security-group-ingress --group-id $securityGroupId --protocol tcp --port 22 --cidr $cidr
# jupyter notebook
aws ec2 authorize-security-group-ingress --group-id $securityGroupId --protocol tcp --port 8888-8898 --cidr $cidr

if [ ! -d ~/.ssh ]
then
	mkdir ~/.ssh
fi

if [ ! -f ~/.ssh/aws-key-$name.pem ]
then
	aws ec2 create-key-pair --key-name aws-key-$name --query 'KeyMaterial' --output text > ~/.ssh/aws-key-$name.pem
	chmod 400 ~/.ssh/aws-key-$name.pem
fi