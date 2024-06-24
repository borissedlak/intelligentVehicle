import ast
import logging
import os
from io import StringIO

import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from pgmpy.readwrite import XMLBIFReader

import utils
from orchestration import model_trainer
from orchestration.HttpClient import HttpClient
from orchestration.ServiceStarter import start_service

app = Flask(__name__)

MODEL_DIRECTORY = "./"

logger = logging.getLogger("vehicle")
logging.getLogger('pgmpy').setLevel(logging.ERROR)  # This worked, but the ones below not...
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.getLogger('vehicle').setLevel(logging.DEBUG)
# logging.filterwarnings("ignore", category=Warning, module='pgmpy')

HTTP_SERVER = utils.get_ENV_PARAM('HTTP_SERVER', "127.0.0.1")
DEVICE_NAME = utils.get_ENV_PARAM('DEVICE_NAME', "Unknown")

http_client = HttpClient(DEFAULT_HOST=HTTP_SERVER)
thread_lib = []
current_platoon = ['host.docker.internal:8000']


# MEMBER ROUTES ######################################

@app.route("/start_service", methods=['POST'])
def start():
    global thread_lib, current_platoon
    service_d = ast.literal_eval(request.args.get('service_description'))
    isolated = not len(thread_lib) > 0

    if not isolated:
        for (thread, wrapper) in thread_lib:
            wrapper.update_isolation(False)

    thread_ref = start_service(service_d, current_platoon, isolated)
    thread_lib.append(thread_ref)

    return utils.log_and_return(logger, logging.INFO, "M| Started service successfully")


@app.route("/stop_all_services", methods=['POST'])
def stop_all():
    global thread_lib
    if len(thread_lib) <= 0:
        return utils.log_and_return(logger, logging.INFO, f"M| No service threads running locally")

    logger.info(f"M| Going to stop {len(thread_lib)} threads")

    for bg_thread, task_object in thread_lib:
        task_object.terminate()
    # service_d = ast.literal_eval(request.args.get('service_description'))
    # start_service(service_d)
    thread_lib = []

    return utils.log_and_return(logger, logging.INFO, "M| Stopped all threads")


@app.route('/model/upload', methods=['POST'])
def override_model():
    global thread_lib
    for f_key in request.files.keys():
        file = request.files[f_key]
        file.save(file.filename)
        # logger.info(f"M| Receiving model file '{file.filename}'")

        for (thread, wrapper) in thread_lib:
            if file.filename == utils.create_model_name(wrapper.s_description['name'], DEVICE_NAME):
                model = XMLBIFReader(file.filename).get_model()
                wrapper.update_model(model)
                logger.info(f"M| Update model for {thread} > {wrapper}")

    return utils.log_and_return(logger, logging.INFO, "M| All files received successfully")


@app.route('/update_platoon_members', methods=['POST'])
def update_platoon_members():
    global current_platoon
    member_ips = request.args.get('platoon_members')
    current_platoon = member_ips.split(",")

    for (thread, wrapper) in thread_lib:
        wrapper.update_platoon(current_platoon)

    return utils.log_and_return(logger, logging.INFO, f"M| Update local list of platoon members to {current_platoon}")


# LEADER ROUTES ######################################

@app.route('/model_list', methods=['GET'])
def list_files():
    files = os.listdir(MODEL_DIRECTORY)
    filtered_files = [f for f in files if f.endswith('model.xml')]

    return jsonify(filtered_files)


@app.route('/model/<model_name>', methods=['GET'])
def provide_model(model_name):
    return send_from_directory(MODEL_DIRECTORY, model_name)


@app.route('/model/update/<model_name>', methods=['POST'])
def update_model_immediately(model_name):
    logger.info(f"L| Start updating '{model_name}'")
    csv_string = request.data.decode('utf-8')
    df = pd.read_csv(StringIO(csv_string))

    # TODO: This should run in a new thread in the bg
    # TODO: What if another process requests to retrain while retrain is still running?
    model_trainer.update_models_new_samples(model_name, df)
    for client_ip in current_platoon:
        http_client.push_files_to_member([model_name], target_route=client_ip)

    return utils.log_and_return(logger, logging.INFO, "L| Updated model successfully")


@app.route('/retrain_models', methods=['POST'])
def retrain_models():
    logger.info("L| Starting model training")

    # TODO: This should run in a new thread in the bg
    model_trainer.retrieve_full_data()
    n = model_trainer.prepare_models()
    http_client.push_files_to_member()  # TODO: Must filter which files

    return utils.log_and_return(logger, logging.INFO, f"Trained {n} models")


def run_server():
    app.run(host='0.0.0.0', port=8080)


services = [{"name": 'CV', 'slo_vars': ["in_time"], 'constraints': {'pixel': '480', 'fps': '5'}}]  # ,
# {"name": 'CV', 'slo_vars': ["in_time"], 'constraints': {'pixel': '480', 'fps': '5'}}]

for service_description in services:
    logger.info(f"Starting {service_description['name']} by default")
    is_isolated = len(services) == 1
    thread_reference = start_service(service_description, current_platoon, isolated=is_isolated)
    thread_lib.append(thread_reference)

# http_client.push_files_to_member([utils.create_model_name("CV", "Orin")], target_route="192.168.31.183")

run_server()
