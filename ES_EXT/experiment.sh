#!/bin/bash

#chmod +x run_python_script.sh
#orinagx ALL=(ALL) NOPASSWD: /home/orinagx/development/intelligentVehicle/ES_EXT/experiment.sh
export PYTHONPATH=~/development/intelligentVehicle
export INITIAL_TRAINING=40
export EXPERIMENT_DURATION=600 # 600 = 10 min
export SHOW_IMG="False" # 600 = 10 min

mkdir mkdir ~/development/intelligentVehicle/ES_EXT/models/backup
mkdir mkdir ~/development/intelligentVehicle/ES_EXT/results/
mkdir mkdir ~/development/intelligentVehicle/ES_EXT/results/pv
mkdir mkdir ~/development/intelligentVehicle/ES_EXT/results/slo_f

python3 ~/development/intelligentVehicle/ES_EXT/models/model_trainer.py

if [ "$DEVICE_NAME" = "NX" ] || [ "$DEVICE_NAME" = "AGX" ]; then
    sudo nvpmodel -m 0
    export POWER_MODE="MAX"
fi
	
#SERVICE_NAME="CV" python3 ~/development/intelligentVehicle/ES_EXT/agent.py
#SERVICE_NAME="QR" python3 ~/development/intelligentVehicle/ES_EXT/agent.py
#SERVICE_NAME="LI" python3 ~/development/intelligentVehicle/ES_EXT/agent.py

python3 ~/development/intelligentVehicle/ES_EXT/models/model_trainer.py

if [ "$DEVICE_NAME" = "NX" ] || [ "$DEVICE_NAME" = "AGX" ]; then
    sudo nvpmodel -m 3
    export POWER_MODE="LIM"
fi

SERVICE_NAME="CV" python3 ~/development/intelligentVehicle/ES_EXT/agent.py
SERVICE_NAME="QR" python3 ~/development/intelligentVehicle/ES_EXT/agent.py
SERVICE_NAME="LI" python3 ~/development/intelligentVehicle/ES_EXT/agent.py
