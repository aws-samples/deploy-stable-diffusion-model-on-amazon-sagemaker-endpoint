#!/bin/bash

sudo service docker stop
sudo cp daemon.json /etc/docker/daemon.json
sudo cp -rp /var/lib/docker /home/ec2-user/SageMaker/docker
sudo service docker start