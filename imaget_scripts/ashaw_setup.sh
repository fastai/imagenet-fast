#!/bin/bash
. $(dirname "$0")/get_ami.sh
. $(dirname "$0")/get_vpc.sh

if [[ -n "$vpcId" ]] && [[ "$vpcId" != "None" ]]; then
    echo "VPC found. Using $vpcId"
    echo "Security Group: $securityGroupId"
    echo "SubnetId: $subnetId"
else
    echo "Could not find existing VPC. Creating new one"
    # . $(dirname "$0")/create_vpc.sh
fi

. $(dirname "$0")/start_spot_no_swap.sh

