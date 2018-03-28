import argparse
import boto3
import os, sys, time
from pathlib import Path

session = boto3.Session()
ec2 = session.resource('ec2')

def get_vpc(name):
    vpcs = list(ec2.vpcs.filter(Filters=[{'Name': 'tag-value', 'Values': [name]}]))
    return vpcs[0] if vpcs else None

def get_vpc_ids(name):
    vpc = get_vpc(name)
    if vpc is None: return None
    sg = list(vpc.security_groups.all())[0]
    subnet = list(vpc.subnets.all())[0]
    return vpc.id, sg.id, subnet.id

def create_ec2_keypair(name):
    ssh_dir = Path.home()/'.ssh'
    ssh_dir.mkdir(exist_ok=True)
    keypair_name = f'aws-key-{name}'
    filename = ssh_dir/f'{keypair_name}.pem'
    if filename.exists():
        print('Keypair exists')
        return
    keypair = ec2.create_key_pair(KeyName=keypair_name)
    keypair_out = keypair.key_material
    outfile = open(filename,'w')
    outfile.write(keypair_out)
    os.chmod(filename, 0o400)
    print('Created keypair')

def create_vpc(name):
    cidr_block='10.0.0.0/28'
    vpc = ec2.create_vpc(CidrBlock=cidr_block)
    vpc.modify_attribute(EnableDnsSupport={'Value':True})
    vpc.modify_attribute(EnableDnsHostnames={'Value':True})
    vpc.create_tags(Tags=[{'Key':'Name','Value':name}])
    
    ig = ec2.create_internet_gateway()
    ig.attach_to_vpc(VpcId=vpc.id)
    ig.create_tags(Tags=[{'Key':'Name','Value':f'{name}-gateway'}])
    
    subnet = vpc.create_subnet(CidrBlock=cidr_block)
    subnet.create_tags(Tags=[{'Key':'Name','Value':f'{name}-subnet'}])
    # TODO: enable public ip?
    # subnet.meta.client.modify_subnet_attribute(SubnetId=subnet.id, MapPublicIpOnLaunch={"Value": True})

    rt = vpc.create_route_table()
    rt.create_tags(Tags=[{'Key':'Name','Value':f'{name}-route-table'}])
    rt.associate_with_subnet(SubnetId=subnet.id)
    rt.create_route(DestinationCidrBlock='0.0.0.0/0', GatewayId=ig.id)
    
    
    cidr = '0.0.0.0/0'
    sg = vpc.create_security_group(GroupName=f'{name}-security-group-test', Description='SG for {name} machine')
    # ssh
    sg.authorize_ingress(IpProtocol='tcp', FromPort=22, ToPort=22, CidrIp=cidr)
    # jupyter notebook
    sg.authorize_ingress(IpProtocol='tcp', FromPort=8888, ToPort=8898, CidrIp=cidr)
    
    return vpc

region2ami = {
    'us-west-2': 'ami-8c4288f4',
    'eu-west-1': 'ami-b93c9ec0',
    'us-east-1': 'ami-c6ac1cbc'
}

def allocate_vpc_addr(instance_id):
    ec2c = session.client('ec2')
    alloc_addr = ec2c.allocate_address(Domain='vpc')
    addr_id = alloc_addr['AllocationId']
    ec2c.associate_address(InstanceId=instance_id, AllocationId=addr_id)
    return addr_id

def create_instance(name, instance_type='t2.nano'):
    ami = region2ami[session.region_name]
    vpc_id, sg_id, subnet_id = get_vpc_ids(name)
    network_interfaces=[{
    'DeviceIndex': 0,
    'SubnetId': subnet_id,
    'Groups': [sg_id],
    'AssociatePublicIpAddress': True            
    }]
    block_device_mappings = [{ 
        'DeviceName': '/dev/sda1', 
        'Ebs': { 
            'VolumeSize': 128, 
            'VolumeType': 'gp2' 
        } 
    }]
    instance = ec2.create_instances(ImageId=ami, InstanceType=instance_type, 
                     MinCount=1, MaxCount=1,
                     KeyName=f'aws-key-{name}',
                     BlockDeviceMappings=block_device_mappings,
                     NetworkInterfaces=network_interfaces
                    )[0]
    instance.create_tags(Tags=[{'Key':'Name','Value':f'{name}-gpu-machine'}])
    
    print('Instance created...')
    instance.wait_until_running()
    
    print('Creating public IP address...')
    addr_id = allocate_vpc_addr(instance.id)
    
    print('Rebooting...')
    instance.reboot()
    instance.wait_until_running()
    return instance