import argparse
import boto3
import paramiko
import os, sys, time
from pathlib import Path

session = boto3.Session()
ec2 = session.resource('ec2')
ec2c = session.client('ec2')

def get_vpc(name):
    vpcs = list(ec2.vpcs.filter(Filters=[{'Name': 'tag-value', 'Values': [name]}]))
    return vpcs[0] if vpcs else None

def get_instance(name):
    instances = list(ec2.instances.filter(Filters=[{'Name': 'tag-value', 'Values': [name]}, {'Name': 'instance-state-name', 'Values': ['running', 'stopped']}]))
    return instances[0] if instances else None

def get_vpc_info(vpc, availability_zone=None):
    try:
        vpc_tag_name = list(filter(lambda i: i['Key'] == 'Name', vpc.tags))[0]['Value']
        sg = list(vpc.security_groups.filter(Filters=[{'Name': 'group-name', 'Values': [f'{vpc_tag_name}-security-group']}]))[0]
        subnet_filters = [{'Name': 'tag-value', 'Values': [f'{vpc_tag_name}-subnet']}]
        #import pdb; pdb.set_trace()
        if availability_zone: subnet_filters.append({'Name': 'tag-value', 'Values': [availability_zone]})
        subnet = list(vpc.subnets.filter(Filters=subnet_filters))[0]
    except Exception as e: print('Could not get VPC info: ', e)
    return sg.id, subnet.id

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

def get_ssh_command(instance):
    return f'ssh -i ~/.ssh/{instance.key_name}.pem ubuntu@{instance.public_ip_address}'

def create_vpc(name):
    cidr_block='10.1.0.0/24'
    vpc = ec2.create_vpc(CidrBlock=cidr_block)
    vpc.modify_attribute(EnableDnsSupport={'Value':True})
    vpc.modify_attribute(EnableDnsHostnames={'Value':True})
    vpc.create_tags(Tags=[{'Key':'Name','Value':name}])
    
    ig = ec2.create_internet_gateway()
    ig.attach_to_vpc(VpcId=vpc.id)
    ig.create_tags(Tags=[{'Key':'Name','Value':f'{name}-gateway'}])
    
    rt = vpc.create_route_table()
    rt.create_tags(Tags=[{'Key':'Name','Value':f'{name}-route-table'}])
    rt.create_route(DestinationCidrBlock='0.0.0.0/0', GatewayId=ig.id)

    # Note: (AS) I have no Idea what is going on here with subnets. 
    zones = [av['ZoneName'] for av in ec2c.describe_availability_zones()['AvailabilityZones']]
    addr_zone = list(zip(reversed(range(0, 256, 64)), zones))
    for addr,zone in reversed(addr_zone):
        cidr_block = f'10.1.0.{addr}/26'
        subnet = vpc.create_subnet(CidrBlock=cidr_block, AvailabilityZone=zone)
        subnet.create_tags(Tags=[{'Key':'Name','Value':f'fast-ai-subnet'}])
        subnet.create_tags(Tags=[{'Key':'Region','Value':zone}])
        # TODO: enable public ip?
        # subnet.meta.client.modify_subnet_attribute(SubnetId=subnet.id, MapPublicIpOnLaunch={"Value": True})

        rt.associate_with_subnet(SubnetId=subnet.id)
    
    
    cidr = '0.0.0.0/0'
    sg = vpc.create_security_group(GroupName=f'{name}-security-group', Description='SG for {name} machine')
    # ssh
    sg.authorize_ingress(IpProtocol='tcp', FromPort=22, ToPort=22, CidrIp=cidr)
    # jupyter notebook
    sg.authorize_ingress(IpProtocol='tcp', FromPort=8888, ToPort=8898, CidrIp=cidr)
    # allow efs
    IpPermissions=[{
        'FromPort': 2049,
        'ToPort': 2049,
        'IpProtocol': 'tcp',
        'UserIdGroupPairs': [{ 'GroupId': sg.id }],
    }]
    sg.authorize_ingress(IpPermissions=IpPermissions)
    
    return vpc

def get_ami(region=None, imagenet=False):
    if imagenet: return 'ami-b67711ce' # Private AMI with imagenet loaded
    if region is None: region = session.region_name
    region2ami = {
        'us-west-2': 'ami-8c4288f4',
        'eu-west-1': 'ami-b93c9ec0',
        'us-east-1': 'ami-c6ac1cbc'
    }
    return region2ami[region]

def allocate_vpc_addr(instance_id):
    alloc_addr = ec2c.allocate_address(Domain='vpc')
    ec2c.associate_address(InstanceId=instance_id, AllocationId=alloc_addr['AllocationId'])
    return alloc_addr

def create_instance(name, launch_specs, allocate_public_ip=False):
    instance = ec2.create_instances(ImageId=launch_specs['ImageId'], InstanceType=launch_specs['InstanceType'], 
                     MinCount=1, MaxCount=1,
                     KeyName=launch_specs['KeyName'],
                     InstanceInitiatedShutdownBehavior='terminate',
                     BlockDeviceMappings=launch_specs['BlockDeviceMappings'],
                     NetworkInterfaces=launch_specs['NetworkInterfaces']
                    )[0]
    instance.create_tags(Tags=[{'Key':'Name','Value':f'{name}'}])
    
    print('Instance created...')
    instance.wait_until_running()
    volume = list(instance.volumes.all())[0]
    volume.create_tags(Tags=[{'Key':'Name','Value':f'{name}'}])

    if allocate_public_ip:
        print('Creating public IP address...')
        addr_id = allocate_vpc_addr(instance.id)['AllocationId']
    
    print('Rebooting...')
    instance.reboot()
    instance.wait_until_running()
    return instance


def wait_on_fulfillment(req):
    while req['State'] != 'active':
        print('Waiting on spot fullfillment...')
        time.sleep(5)
        reqs = ec2c.describe_spot_instance_requests(Filters=[{'Name': 'spot-instance-request-id', 'Values': [req['SpotInstanceRequestId']]}])
        req = reqs['SpotInstanceRequests'][0]
        req_status = req['Status']
        if req_status['Code'] not in ['pending-evaluation', 'pending-fulfillment', 'fulfilled']:
            print('Spot instance request failed:', req_status['Message'])
            print('Cancelling request. Please try again or use on demand.')
            ec2c.cancel_spot_instance_requests(SpotInstanceRequestIds=[req['SpotInstanceRequestId']])
            print(req)
            return None
    instance_id = req['InstanceId']
    print('Fulfillment completed. InstanceId:', instance_id)
    return instance_id
    
def get_spot_prices():
    hist = ec2c.describe_spot_price_history()['SpotPriceHistory']
    return {h['InstanceType']:h['SpotPrice'] for h in hist}

class LaunchSpecs:
    def __init__(self, vpc, instance_type='t2.micro', volume_size=300, delete_ebs=True, ami=None, availability_zone=None):
        self.ami = ami if ami else get_ami()
        self.sg_id, self.subnet_id = get_vpc_info(vpc, availability_zone=availability_zone)
        self.instance_type = instance_type
        self.device = '/dev/sda1'
        self.volume_size = volume_size
        self.volume_type = 'gp2'
        self.io = 5000
        self.delete_ebs = delete_ebs
        self.vpc_tagname = list(filter(lambda i: i['Key'] == 'Name', vpc.tags))[0]['Value']
        self.keypair_name = f'aws-key-{self.vpc_tagname}'

    def build_ebs(self):
        ebs = {
            # Volume size must be greater than snapshot size of 80
            'VolumeSize': self.volume_size, 
            'DeleteOnTermination': self.delete_ebs,
            'VolumeType': self.volume_type
        }
        if self.volume_type == 'io1': ebs['Iops'] = self.io
        return ebs

    def build(self):        
        launch_specification = {
            'ImageId': self.ami, 
            'InstanceType': self.instance_type, 
            'KeyName': self.keypair_name,
            'NetworkInterfaces': [{
                'DeviceIndex': 0,
                'SubnetId': self.subnet_id,
                'Groups': [self.sg_id],
                'AssociatePublicIpAddress': True            
            }],
            'BlockDeviceMappings': [{
                'DeviceName': '/dev/sda1', 
                'Ebs': self.build_ebs()
            }]
        }
        return launch_specification
    
def create_spot_instance(name, launch_specs, spot_price=None):
    if spot_price is None:
        spot_requests = ec2c.request_spot_instances(LaunchSpecification=launch_specs)    
    else:
        spot_requests = ec2c.request_spot_instances(SpotPrice=spot_price, LaunchSpecification=launch_specs)
    spot_request = spot_requests['SpotInstanceRequests'][0]
    instance_id = wait_on_fulfillment(spot_request)
    if not instance_id:
        return

    print('Rebooting...')
    instance = list(ec2.instances.filter(Filters=[{'Name': 'instance-id', 'Values': [instance_id]}]))[0]
    instance.reboot()
    instance.wait_until_running()
    instance.create_tags(Tags=[{'Key':'Name','Value':f'{name}'}])
    volume = list(instance.volumes.all())[0]
    volume.create_tags(Tags=[{'Key':'Name','Value':f'{name}'}])
    print(f'Completed. SSH: ', get_ssh_command(instance))
    return instance

from time import sleep
def create_efs(name, vpc, performance_mode='generalPurpose'):
    sg_id, subnet_id = get_vpc_info(vpc)
    efsc = session.client('efs')
    efs_response = efsc.create_file_system(CreationToken=f'{name}', PerformanceMode=performance_mode)
    efs_id = efs_response['FileSystemId']
    efsc.create_tags(FileSystemId=efs_id, Tags=[{'Key': 'Name', 'Value': f'{name}'}])
    sleep(5)
    mount_target = efsc.create_mount_target(FileSystemId=efs_id,
                                              SubnetId=subnet_id,
                                              SecurityGroups=[sg_id])
    return efs_response

def get_efs_address(name):
    efsc = session.client('efs')
    file_systems = efsc.describe_file_systems()['FileSystems']
    target = list(filter(lambda x: x['Name'] == name, file_systems))
    if target:
        fs_id = target[0]['FileSystemId']
        region = session.region_name
        return f'{fs_id}.efs.{region}.amazonaws.com'
    print(f'Could not find address with name: {name}. Here are existing addresses:', file_systems)

def attach_efs(efs_name, client):
    efs_addr = get_efs_address(efs_name)
    efs_mount_cmd = f'sudo mount -t nfs -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 {efs_addr}:/ ~/efs_mount'
    run_command(client, efs_mount_cmd)

def attach_volume(instance, volume_tag, device='/dev/xvdf'):
    volumes = list(ec2.volumes.filter(Filters=[{'Name': 'tag-value', 'Values': [volume_tag]}]))
    if not volumes: print('Could not find volume for tag:', volume_tag); return
    volume = volumes[0]
    if volume.state == 'in-use': print('Volume already attached to a different instance.'); return
    instance.attach_volume(Device=device, VolumeId=volumes[0].id)
    instance.wait_until_running()
    return volume

def mount_volume(client, device='/dev/xvdf', mount_dir='ebs_mount', reformat=False):
    if reformat:
        run_command(client, f'sudo mkfs -t ext4 {device}')
    run_command(client, f'sudo mkdir {mount_dir}')
    run_command(client, f'sudo mount {device}1 {mount_dir}') # no reformatting

def create_volume(name, az, size=120, volume_type='gp2'):
    tag_specs = [{
        'Tags': [{
            'Key': 'Name',
            'Value': f'{name}'
        }]
    }]
    volume = ec2.create_volume(Size=size, VolumeType=volume_type, TagSpecifications=tag_specs,
                              AvailabilityZone=az)
    return volume
    

# SSH

def connect_to_instance(instance, keypath=f'{Path.home()}/.ssh/aws-key-fast-ai.pem', username='ubuntu', timeout=10):
    print('Connecting to SSH...')
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    retries = 20
    while retries > 0:
        try:
            client.connect(instance.public_ip_address, username=username, key_filename=keypath, timeout=timeout)
            print('Connected!')
            break
        except Exception as e:
            print(f'Exception: {e} Retrying...')
            retries = retries - 1
            time.sleep(10)
    return client

def run_command(client, cmd, inputs=[], print_output=False):
    stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)
    for inp in inputs:
        # example = 'mypassword\n'
        stdin.write(inp)
    stdout_str = stdout.read().decode('utf8')
    stderr_str = stderr.read().decode('utf8')
    
    if print_output:
        print("run_command returned: \n" + stdout_str)
    return stdout_str, stderr_str

def upload_file(client, localpath, remotepath):
    #     file = f'{Path.home()}/Projects/ML/fastai/fastai_imagenet/testfile.txt'
    ftp_client=client.open_sftp()
    ftp_client.put(localpath, remotepath)
    ftp_client.close()

    return run_command(client, f'chmod +x {remotepath}')


# TMUX
class TmuxSession:
    def __init__(self, client, name):
        self.client = client
        self.name = name
        out, _ = run_command(client, f'tmux new-session -s {name} -n 0 -d')
        if out and 'duplicate session' in out:
            self.attach()
        self.windows = [f'{self.name}:0']
        
    def attach(self):
        return run_command(self.client, f'tmux a -t {self.name}')

    def run_command(self, cmd, window_id=0):
        num_windows = len(self.windows)
        if window_id >= num_windows:
            print('Window does not exist. Creating new one at index:', num_windows)
            window_id = self.new_window()
        window_name = self.windows[window_id]
        return run_command(self.client, f'tmux send-keys -t {window_name} "{cmd}" Enter')
        
    def close(self):
        return run_command(self.client, f'tmux kill-session -t {self.name}')

    def new_window(self):
        window_id = len(self.windows)
        self.windows.append(f'{self.name}:{window_id}')
        run_command(self.client, f'tmux new-window -t {self.name} -n {window_id}')
        print('Created new window. Id:', window_id)
        return window_id

    def get_tmux_command(self, window_id=0):
        if window_id >= len(self.windows): print('Could not find window')
        return f'tmux a -t {self.name}'
