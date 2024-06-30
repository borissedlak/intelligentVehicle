import ast
import logging
import os
import threading
from io import StringIO

import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from pgmpy.readwrite import XMLBIFReader

import utils
from orchestration.HttpClient import HttpClient
from orchestration.ServiceWrapper import start_service
from orchestration.models import model_trainer

app = Flask(__name__)

MODEL_DIRECTORY = "./"

logger = logging.getLogger("vehicle")
logging.getLogger('pgmpy').setLevel(logging.ERROR)
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.getLogger('vehicle').setLevel(logging.INFO)

HTTP_SERVER = utils.get_ENV_PARAM('HTTP_SERVER', "127.0.0.1")
DEVICE_NAME = utils.get_ENV_PARAM('DEVICE_NAME', "Unknown")

http_client = HttpClient(DEFAULT_HOST=HTTP_SERVER)
thread_lib = []
current_platoon = ['localhost']
service_host_map = {}


# MEMBER ROUTES ######################################

@utils.print_execution_time
@app.route("/start_service", methods=['POST'])
def start():
    global thread_lib, current_platoon, service_host_map
    service_d = ast.literal_eval(request.args.get('service_description'))
    isolated = not len(thread_lib) > 0

    if not isolated:
        for wrapper in thread_lib:
            wrapper.update_isolation(False)

    thread_ref = start_service(service_d, current_platoon, isolated)
    thread_lib.append(thread_ref)
    localhost = utils.get_local_ip()
    s_id_type = f"{service_d['type']}-{service_d['id']}"
    service_host_map[s_id_type] = {'desc': service_d, 'host': localhost}
    update_wrapper_service_assignments()

    threading.Thread(target=update_other_members, args=(service_d, localhost)).start()

    return "M| Started service successfully"


def update_other_members(service_d, localhost):
    global current_platoon
    other_members = utils.get_all_other_members(current_platoon)

    if len(other_members) == 0:
        logger.debug("M| No other platoon members to inform about start")
    else:
        for vehicle_address in other_members:
            http_client.update_service_assignment(str(service_d), localhost, vehicle_address)
        logger.debug(f"M| Informed {len(other_members)} platoon members about service start")


# Write: This resets their SLO history to avoid making decisions while the new assignment is not reflected yet
def update_wrapper_service_assignments():
    global service_host_map, thread_lib

    for thread in thread_lib:
        thread.update_service_assignment(service_host_map)
        thread.reset_slo_history()


@app.route("/stop_all_services", methods=['POST'])
def stop_all():
    global thread_lib
    if len(thread_lib) <= 0:
        return utils.log_and_return(logger, logging.INFO, f"M| No service threads running locally that can be stopped")

    logger.info(f"M| Going to stop {len(thread_lib)} threads")

    for wrapper in thread_lib:
        wrapper.terminate()
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

        for wrapper in thread_lib:
            if file.filename == utils.create_model_name(wrapper.s_desc['type'], DEVICE_NAME):
                model = XMLBIFReader("models/" + file.filename).get_model()
                wrapper.update_model(model)
                logger.info(f"M| Update model for {wrapper.s_desc['type']}-{wrapper.s_desc['id']}")

    return utils.log_and_return(logger, logging.INFO, "M| All models received successfully")


@app.route('/update_platoon_members', methods=['POST'])
def update_platoon_members():
    global current_platoon
    member_ips = request.args.get('platoon_members')
    current_platoon = member_ips.split(",")

    for wrapper in thread_lib:
        wrapper.update_platoon(current_platoon)

    return utils.log_and_return(logger, logging.INFO, f"M| Update local list of platoon members to {current_platoon}")


@app.route('/update_service_assignment', methods=['POST'])
def update_service_assignment():
    global service_host_map
    s_desc = ast.literal_eval(request.args.get('service_description'))
    s_host = request.args.get('service_host')
    s_id_type = f"{s_desc['type']}-{s_desc['id']}"
    service_host_map[s_id_type] = {'desc': s_desc, 'host': s_host}

    update_wrapper_service_assignments()
    return utils.log_and_return(logger, logging.DEBUG, f"M| Updated service assignment for {s_id_type}")


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
    # logger.info(f"L| Start updating '{model_name}'")
    csv_string = request.data.decode('utf-8')
    df = pd.read_csv(StringIO(csv_string))

    # TOD: This should run in a new thread in the bg
    # TOD: What if another process requests to retrain while retrain is still running?
    model_trainer.update_models_new_samples(model_name, df)
    for client_ip in current_platoon:
        http_client.push_files_to_member([model_name], target_route=client_ip)

    return utils.log_and_return(logger, logging.INFO, "L| Updated model successfully")


# @app.route('/retrain_models', methods=['POST'])
# def retrain_models():
#     logger.info("L| Starting model training")
#
#     # TOD: This should run in a new thread in the bg
#     model_trainer.retrieve_full_data()
#     n = model_trainer.prepare_models()
#     http_client.push_files_to_member()  # TOD: Must filter which files
#
#     return utils.log_and_return(logger, logging.INFO, f"Trained {n} models")


def run_server():
    app.run(host='0.0.0.0', port=8080)


services = []  # [{"id": 1, "type": 'CV', 'slo_vars': ["in_time"], 'constraints': {'pixel': '480', 'fps': '5'}}]  # ,
# {"id": 2, "name": 'CV', 'slo_vars': ["in_time"], 'constraints': {'pixel': '480', 'fps': '5'}}]

for service_description in services:
    logger.info(f"Starting {service_description['type']} by default")
    is_isolated = len(services) == 1
    thread_reference = start_service(service_description, current_platoon, isolated=is_isolated)
    thread_lib.append(thread_reference)

run_server()
