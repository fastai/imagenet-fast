#!/bin/bash
tmux new -s my_session
cd cifar10
bash run_cifar10.sh
tmux detach