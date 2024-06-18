import os
import threading
import time
import traceback

from pgmpy.inference import VariableElimination
from pgmpy.readwrite import XMLBIFReader

import global_utils
from ACI import ACI
from services.CV.VideoDetector import VideoDetector

# from ACI import ACI
# from services.VideoDetector import VideoDetector

HTTP_SERVER = os.environ.get('HTTP_SERVER')
if HTTP_SERVER:
    print(f'Found ENV value for HTTP_SERVER: {HTTP_SERVER}')
else:
    HTTP_SERVER = "127.0.0.1"
    print(f"Didn't find ENV value for HTTP_SERVER, default to: {HTTP_SERVER}")

DEVICE_NAME = os.environ.get('DEVICE_NAME')
if DEVICE_NAME:
    print(f'Found ENV value for DEVICE_NAME: {DEVICE_NAME}')
else:
    DEVICE_NAME = "Unknown"
    print(f"Didn't find ENV value for DEVICE_NAME, default to: {DEVICE_NAME}")

CLEAN_RESTART = os.environ.get('CLEAN_RESTART')
if CLEAN_RESTART:
    print(f'Found ENV value for CLEAN_RESTART: {CLEAN_RESTART}')
else:
    CLEAN_RESTART = False
    print(f"Didn't find ENV value for CLEAN_RESTART, default to: {CLEAN_RESTART}")

DISABLE_ACI = os.environ.get('DISABLE_ACI')
if DISABLE_ACI:
    print(f'Found ENV value for DISABLE_ACI: {DISABLE_ACI}')
else:
    DISABLE_ACI = False
    print(f"Didn't find ENV value for DISABLE_ACI, default to: {DISABLE_ACI}")

SEND_SYSTEM_STATS = os.environ.get('SEND_SYSTEM_STATS')
if SEND_SYSTEM_STATS:
    print(f'Found ENV value for SEND_SYSTEM_STATS: {SEND_SYSTEM_STATS}')
else:
    SEND_SYSTEM_STATS = False
    print(f"Didn't find ENV value for SEND_SYSTEM_STATS, default to: {SEND_SYSTEM_STATS}")

SHOW_IMG = os.environ.get('SHOW_IMG')
if SHOW_IMG:
    print(f'Found ENV value for SHOW_IMG: {SHOW_IMG}')
else:
    SHOW_IMG = False
    print(f"Didn't find ENV value for SHOW_IMG, default to: {SHOW_IMG}")

SERVICE_TYPE = os.environ.get('SERVICE_TYPE')
if SERVICE_TYPE:
    print(f'Found ENV value for SHOW_IMG: {SERVICE_TYPE}')
else:
    SERVICE_TYPE = VideoDetector
    print(f"Didn't find ENV value for SERVICE_TYPE, default to: {SERVICE_TYPE}")

# detector = VideoProcessor(device_name=DEVICE_NAME, privacy_chain=chain, display_stats=False, simulate_fps=True)

model_name = None  # if CLEAN_RESTART else f"model_{DEVICE_NAME}.xml"
aci = ACI(distance_slo=100, network_slo=(420 * 30 * 10), load_model=model_name, device_name=DEVICE_NAME, show_img=SHOW_IMG)

service_params = (180, 22)

new_data = False
override_next_config = None

inferred_config_hist = []


# util_fgcs.clear_performance_history('../data/Performance_History.csv')


# http_client = HttpClient(HOST=HTTP_SERVER)
# http_client.override_stream_config(1)


# Function for the background loop
def processing_loop():
    # loaded_class = load_class("services.CV.VideoDetector", "VideoDetector")
    #
    # if loaded_class:
    #     # Instantiate the class
    #     instance = loaded_class()
    #     print(f"Loaded class: {instance.__class__.__name__}")
    # else:
    #     print("Class could not be loaded")

    vd = SERVICE_TYPE()
    global service_params, new_data
    while True:
        vd.process_one_iteration(params=service_params)
        # if SEND_SYSTEM_STATS:
        #     http_client.send_system_stats(int(psutil.cpu_percent()), DEVICE_NAME, DISABLE_ACI, detector.gpu_available)
        new_data = True


background_thread = threading.Thread(target=processing_loop)
background_thread.daemon = True  # Set the thread as a daemon, so it exits when the main program exits
background_thread.start()


class AIFBackgroundThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True  # Set the thread as a daemon, so it will exit when the main program exits

    def run(self):
        global service_params
        while True:
            try:
                pixel_list = [480, 720, 1080]
                fps_list = [10, 15, 20, 25, 30]

                print("Loading pretained model")
                model = XMLBIFReader("./orchestration/CV_Laptop_model.xml").get_model()
                global_utils.export_BN_to_graph(model, vis_ls=['circo'], save=True, name="raw_model", show=True)

                var_el = VariableElimination(model)

                mode_list = model.get_cpds("config").__getattribute__("state_names")["config"]
                bitrate_list = model.get_cpds("bitrate").__getattribute__("state_names")["bitrate"]
                config_line = []

                for br in bitrate_list:
                    for mode in mode_list:
                        config_line.append((br, distance, time, transformed, mode,
                                            samples[samples['bitrate'] == br]['pixel'].iloc[0],
                                            samples[samples['bitrate'] == br]['fps'].iloc[0],
                                            cons))

                config_line = sorted(config_line, key=lambda x: x[7])
                for (br, distance, time, transformed, mode, pixel, fps, cons) in config_line:
                    print(pixel, fps, mode, distance, time, transformed, cons)

                # new_params = aci.iterate()
                service_params = new_params
                time.sleep(0.2)
                #
                # if new_data:
                #     new_data = False
                #     # d_threads = http_client.get_latest_stream_config()[0]
                #     # past_pixel, past_fps, past_pv, past_ra = real
                #     # http_client.send_app_stats(past_pixel, past_fps, past_pv, past_ra, d_threads, DEVICE_NAME,
                #     #                            detector.gpu_available, surprise)
                #     # inferred_config_hist.append((new_pixel, new_fps))
                #     # if override_next_config:
                #     #     c_pixel, c_fps = override_next_config
                #     #     override_next_config = None
                #     # else:
                # else:
            except Exception as e:
                error_traceback = traceback.format_exc()
                print("Error Traceback:")
                print(error_traceback)
                global_utils.print_in_red(f"ACI Background thread encountered an exception:{e}")


if not DISABLE_ACI:
    background_thread = AIFBackgroundThread()
    background_thread.start()

# Main loop to read commands from the CLI
while True:
    user_input = input()

    # Check if the user entered a command
    if user_input:
        # threads = http_client.get_latest_stream_config()[0]
        # if user_input == "+":
        #     http_client.override_stream_config(threads + 1)
        # elif user_input == "-":
        #     http_client.override_stream_config(1 if threads == 1 else (threads - 1))
        # elif user_input == "i":
        #     aci.initialize_bn()
        if user_input == "e":
            pass
            # aci.export_model(DEVICE_NAME)
        # elif user_input == "q":
        #     aci.export_model()
        #     sys.exit()
