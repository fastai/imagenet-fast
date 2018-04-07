from aws_setup import *

parser = argparse.ArgumentParser(description='Fast.ai ImageNet Training')

parser.add_argument('-p', '--project-name', required=True, type=str,
                    help='Name this experiment/project. It is also the instance name')
parser.add_argument('-t', '--instance-type', required=True, type=str,
                    help='Instance type')
parser.add_argument('-vpc', '--vpc-name', default='fast-ai', type=str,
                    help='AWS VPC to create instance on (default: fast-ai)')
parser.add_argument('-vs', '--volume-size', default=300, type=int,
                    help='Size of ebs volume to create')
parser.add_argument('-d', '--delete-ebs', default=True, type=bool,
                    help='Delete ebs volume instance termination (default: True)')
parser.add_argument('-efs', '--efs-name', default='fast-ai-efs', type=str,
                    help='Name of efs volume to attach (default: fast-ai-efs)')
parser.add_argument('-r', '--run-script', type=str,
                    help='Run Script')

args = parser.parse_args()

def launch_instance(instance_name, launch_specs, itype):
    instance = None
    if itype == 'demand':
        print('Starting on demand instance')
        instance = create_instance(instance_name, launch_specs)
    elif itype == 'spot':
        spot_prices = get_spot_prices()
        print('Creating Spot. Prices:', {k:v for (k,v) in spot_prices.items() if args.instance_type[:2] in k})
        instance = create_spot_instance(instance_name, launch_specs)
    elif itype == 'cancel':
        print('Cancelling request...')
        return
    if not instance:
        itype = input("Instance creation error. Try again? spot/demand/cancel")
        return launch_instance(instance_name, launch_specs, itype)
    return instance
    

def main():
    instance_name = f'{args.vpc_name}-{args.project_name}'
    instance = get_instance(instance_name)
    if instance: 
        print(f'Instance with name already found with name: {instance_name}. Connecting to this instead')
    else:
        vpc = get_vpc(args.vpc_name);
        launch_specs = LaunchSpecs(vpc, instance_type=args.instance_type, volume_size=args.volume_size, delete_ebs=args.delete_ebs).build()
        instance = launch_instance(instance_name, launch_specs, 'spot')
        
    if not instance: print('Instance failed.'); return;

    client = connect_to_instance(instance)
    print(f'Completed. SSH: ', get_ssh_command(instance))

    if not args.run_script:
        return
        
    script_loc = Path(args.run_script)
    script_loc = Path(script_loc.expanduser())

    tsess = TmuxSession(client, 'imagenet')

    upload_file(client, script_loc, script_loc.name)
    tsess.run_command(f'bash {script_loc.name}')

    tmux_cmd = tsess.get_tmux_command()
    print(f'Running script. \nTmux: {tmux_cmd}')

main()


