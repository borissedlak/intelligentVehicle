#!/bin/bash

#chmod +x run_python_script.sh
export PYTHONPATH=~/development/intelligentVehicle
export INITIAL_TRAINING=5
export EXPERIMENT_DURATION=3000 # 600 = 10 min

python3 ~/development/intelligentVehicle/ES_EXT/models/model_trainer.py

if [ "$DEVICE_NAME" = "NX" ] || [ "$DEVICE_NAME" = "AGX" ]; then
    sudo nvpmodel -m 0
fi

SERVICE_NAME="CV" python3 ~/development/intelligentVehicle/ES_EXT/agent.py
#SERVICE_NAME="QR" python3 ~/development/intelligentVehicle/ES_EXT/agent.py
#SERVICE_NAME="LI" python3 ~/development/intelligentVehicle/ES_EXT/agent.py

if [ "$DEVICE_NAME" = "NX" ] || [ "$DEVICE_NAME" = "AGX" ]; then
    sudo nvpmodel -m 3
fi

#SERVICE_NAME="CV" python3 ~/development/intelligentVehicle/ES_EXT/agent.py
#SERVICE_NAME="QR" python3 ~/development/intelligentVehicle/ES_EXT/agent.py
#SERVICE_NAME="LI" python3 ~/development/intelligentVehicle/ES_EXT/agent.py