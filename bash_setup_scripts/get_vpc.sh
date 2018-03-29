#!/bin/bash
# settings
export name="fast-ai"
export vpcTag=$name
export sgTag="$name-security-group"
export subnetTag="$name-subnet"

export vpcId="$(aws ec2 describe-vpcs --filters Name=tag-value,Values=$vpcTag --query 'Vpcs[0].VpcId' --output text)"
export securityGroupId="$(aws ec2 describe-security-groups --filter "Name=vpc-id,Values=$vpcId,Name=group-name,Values=$sgTag" --query 'SecurityGroups[0].GroupId' --output text)"
export subnetId="$(aws ec2 describe-subnets --filter Name=tag-value,Values=$subnetTag --query 'Subnets[0].SubnetId' --output text)"