#!/bin/bash
if [[ $# -lt 2 ]] ; then
  echo 'Please provide instance type and bid amount'
  exit 0
fi

# settings
export envName="main-env"
export name="spot-instance"
#export ami="ami-6f587e1c"
export ami="ami-785db401"

. $envName-vars.sh
. utils/create-ssh-key-pair.sh

export networkInterfaceId=`aws ec2 create-network-interface --subnet-id $subnetId --groups $securityGroupId --query 'NetworkInterface.NetworkInterfaceId' --output text`
export allocAddr=`aws ec2 allocate-address --domain vpc --query 'AllocationId' --output text`
export assocId=`aws ec2 associate-address --network-interface-id $networkInterfaceId --allocation-id $allocAddr --query 'AssociationId' --output text`
export instancePublicIp=`aws ec2 describe-addresses --query 'Addresses[?AssociationId==\`'$assocId'\`][PublicIp]' --output text`

export spotInstanceRequestId=`aws ec2 request-spot-instances --spot-price "$2" --launch-specification '{"ImageId": "'$ami'", "InstanceType": "'$1'", "KeyName": "'aws-key-$name'", "NetworkInterfaces": [{"DeviceIndex": 0, "NetworkInterfaceId": "'$networkInterfaceId'"}], "BlockDeviceMappings": [{"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 20, "VolumeType": "gp2", "DeleteOnTermination": true}}]}' --query 'SpotInstanceRequests[0].[SpotInstanceRequestId]' --output text`

export removeFileName=$name-remove.sh
echo "#!/bin/bash" > $removeFileName
echo instanceId=\$\(aws ec2 describe-spot-instance-requests --query "'SpotInstanceRequests[?SpotInstanceRequestId==\`$spotInstanceRequestId\`].[InstanceId]'" --output text\) >> $removeFileName
echo aws ec2 disassociate-address --association-id $assocId >> $removeFileName
echo aws ec2 release-address --allocation-id $allocAddr >> $removeFileName

echo aws ec2 terminate-instances --instance-ids \$instanceId >> $name-remove.sh
echo aws ec2 wait instance-terminated --instance-ids \$instanceId >> $name-remove.sh

echo aws ec2 delete-network-interface --network-interface-id $networkInterfaceId >> $removeFileName
echo aws ec2 cancel-spot-instance-requests --spot-instance-request-ids $spotInstanceRequestId >> $removeFileName
echo rm -f $removeFileName >> $removeFileName
echo rm -f ~/aws_scripts/$name* >> $name-remove.sh
chmod +x $removeFileName
#echo aws ec2 delete-key-pair --key-name aws-key-$name >> $name-remove.sh

# Create maintenance scripts
. utils/create-login-script.sh
chmod +x ~/aws_scripts/$name*