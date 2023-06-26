#!/bin/bash

# Start a docker, mapping dev code bases into its volume
nvidia-docker run \
                -v /mnt/efs/gsf-data:/data \
                -v /home/ubuntu/graphstorm:/develop/graphstorm \
                --network=host \
                --shm-size=700g \
                -d graphstorm:0608

# connect the container and run experiments inside it.
docker exec -it test /bin/bash

# set up e2e test
cd /develop/graphstorm/
bash tests/end2end-tests/setup.sh

