#!/bin/bash
#
# Configure a t2.xlarge instance

# get the correct ami
export region=$(aws configure get region)
if [[ $region = "us-west-2" ]]; then
  export ami="ami-8c4288f4" # Oregon
elif [[ $region = "eu-west-1" ]]; then
  export ami="ami-b93c9ec0" # Ireland
elif [[ $region = "us-east-1" ]]; then
  export ami="ami-c6ac1cbc" # Virginia
else
  echo "Only us-west-2 (Oregon), eu-west-1 (Ireland), and us-east-1 (Virginia) are currently supported"
  exit 1
fi

export instanceType="t2.xlarge"

. $(dirname "$0")/setup_instance.sh
