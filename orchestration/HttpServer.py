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
logging.getLogger('vehicle').setLevel(logging.INFO)
# logging.filterwarnings("ignore", category=Warning, module='pgmpy')

HTTP_SERVER = utils.get_ENV_PARAM('HTTP_SERVER', "127.0.0.1")
DEVICE_NAME = utils.get_ENV_PARAM('DEVICE_NAME', "Unknown")

http_client = HttpClient(DEFAULT_HOST=HTTP_SERVER)
thread_lib = []


# MEMBER ROUTES ######################################

@app.route("/start_service", methods=['POST'])
def start():
    global thread_lib
    service_d = ast.literal_eval(request.args.get('service_description'))
    thread_description = start_service(service_d)
    thread_lib.append(thread_description)

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
    for client_ip in utils.discover_platoon_devices():
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


services = []  # [{"name": 'CV', 'slo_var': ["in_time"], 'constraints': {'pixel': '480', 'fps': '5'}}]#,
# {"name": 'CV', 'slo_var': ["in_time"], 'constraints': {'pixel': '480', 'fps': '5'}}]

for service_description in services:
    logger.info(f"Starting {service_description['name']} by default")
    thread_reference = start_service(service_description)
    thread_lib.append(thread_reference)

run_server()
