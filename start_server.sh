#!/bin/bash


#sudo mkdir -p /usr/lib/x86_64-linux-gnu/nvidia_manual
#cd /usr/lib/x86_64-linux-gnu/nvidia_manual/
#sudo rm *
#sudo find /usr/lib/ -name "libnvidia*" -exec ln -s {} ./ \;
#sudo find /usr/lib/ -name "libcuda*" -exec ln -s {} ./ \;
#cd -

#LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia_manual/:$LD_LIBRARY_PATH


./build-vec_indexer-VecIndexerKit-Debug/src/vec_indexer_server -c etc/server.config -l etc/log_config.conf
