1.) Clone project
--------------------
    
    mkdir development
    git clone http://github.com/borissedlak/intelligentVehicle

2.) Set Env Variables
--------------------
    sudo apt install nano
    sudo nano /etc/environment
    PYTHONPATH=/home/jetson/development/intelligentVehicle
    DEVICE_NAME=NX//AGX
    source /etc/environment

X.) Install Python3.10
--------------------
    XX sudo add-apt-repository ppa:deadsnakes/ppa
    XX sudo apt install python3.10
    XX sudo apt install python3.10-distutils
    XX curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

3.) Install CUDA (if missing)
--------------------
    sudo apt-get install nvidia-cuda
    sudo apt-get install nvidia-jetpack
    sudo apt-get install libopenblas-dev
    nvcc --version

4.) Install package dependencies
--------------------
    pip install -r requirements.txt
    pip install ./misc/onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl --force-reinstall
    pip install ./torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl --force-reinstall
    
    pip install -r requirements_3_10.txt
    pip install ./misc/onnxruntime_gpu-1.17.0-cp310-cp310-linux_aarch64.whl --force-reinstall
    pip install ./torch-2.3.0-cp310-cp310-linux_aarch64.whl --force-reinstall

5.) Before running
--------------------
    cd orchestration/models
    python3 ./model_trainer.py
    cp dataset to /services/LI

6.) Running
--------------------

    python3 /HttpServer.py

Start services through HTTP, e.g., POST to 

    localhost:8080/start_service?service_description={'id': {{ServiceID}}, "type": '1', 'slo_vars': ['in_time','energy_saved'], 'constraints': {'pixel': '480', 'fps': '10'}}

