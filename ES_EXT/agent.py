import logging
import os
import threading
import time
import traceback

# import ModelParser
# import Models
import util_fgcs
import utils
from ACI import ACI
from http_client import HttpClient
from monitor.DeviceMetricReporter import CyclicArray
from services.CV.YoloDetector import YoloDetector

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

model_name = None if CLEAN_RESTART else utils.create_model_name("CV", DEVICE_NAME)
aci = ACI(description={'type': 'CV', 'slo_vars': ['in_time', 'rate_60']}, load_model='models/' + model_name, show_img=SHOW_IMG)

c_pixel = ACI.pixel_list[1]
c_fps = ACI.fps_list[2]

logger = logging.getLogger("vehicle")
override_next_config = None
metrics_buffer = CyclicArray(500)

inferred_config_hist = []
util_fgcs.clear_performance_history('../data/Performance_History.csv')

http_client = HttpClient(HOST=HTTP_SERVER)

yd = YoloDetector("localhost", show_results=False)


# Function for the background loop
def processing_loop():
    global c_pixel, c_fps
    while True:
        reality_metrics = yd.process_one_iteration({'pixel': c_pixel, 'fps': c_fps})
        metrics_buffer.append(reality_metrics)
        if SEND_SYSTEM_STATS:
            pass


background_thread = threading.Thread(target=processing_loop, daemon=True)
background_thread.start()


class ACIBackgroundThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True  # Set the thread as a daemon, so it will exit when the main program exits

    def run(self):
        global c_pixel, c_fps, override_next_config, inferred_config_hist
        while True:
            time.sleep(1.0)
            try:
                if metrics_buffer.is_empty():
                    continue
                else:
                    input_metrics = metrics_buffer.get()
                    metrics_buffer.clear()
                    (new_pixel, new_fps, pv, real, surprise) = aci.iterate(input_metrics)
                    # past_pixel, past_fps, past_pv = real

                    inferred_config_hist.append((new_pixel, new_fps))
                    if override_next_config:
                        c_pixel, c_fps = override_next_config
                        override_next_config = None
                    else:
                        if (c_pixel, c_fps) != (new_pixel, new_fps):
                            print(f"Changing configuration to {(new_pixel, new_fps)}")

                        c_pixel, c_fps = new_pixel, new_fps
            except Exception as e:
                error_traceback = traceback.format_exc()
                print("Error Traceback:")
                print(error_traceback)
                logger.error(f"ACI Background thread encountered an exception:{e}")


if not DISABLE_ACI:
    background_thread = ACIBackgroundThread()
    background_thread.start()

# Main loop to read commands from the CLI
while True:
    user_input = input()

    # Check if the user entered a command
    if user_input:
        threads = http_client.get_latest_stream_config()[0]
        if user_input == "+":
            http_client.override_stream_config(threads + 1)
        elif user_input == "-":
            http_client.override_stream_config(1 if threads == 1 else (threads - 1))
        elif user_input == "i":
            aci.bnl(aci.entire_training_data)
        elif user_input == "e":
            aci.export_model()
        # elif user_input == "q":
        #     aci.export_model()
        #     sys.exit()
