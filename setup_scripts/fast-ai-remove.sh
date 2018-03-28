#!/bin/bash
aws ec2 disassociate-address --association-id eipassoc-0452e0f8
aws ec2 release-address --allocation-id eipalloc-aab11596
aws ec2 terminate-instances --instance-ids i-030b3b97e04f2bf00
aws ec2 wait instance-terminated --instance-ids i-030b3b97e04f2bf00
aws ec2 delete-security-group --group-id sg-f60fca88
aws ec2 disassociate-route-table --association-id rtbassoc-9119a2eb
aws ec2 delete-route-table --route-table-id rtb-1c146964
aws ec2 detach-internet-gateway --internet-gateway-id igw-c36cf6a5 --vpc-id vpc-6e6b2a17
aws ec2 delete-internet-gateway --internet-gateway-id igw-c36cf6a5
aws ec2 delete-subnet --subnet-id subnet-f056ff89
aws ec2 delete-vpc --vpc-id vpc-6e6b2a17
echo If you want to delete the key-pair, please do it manually.
