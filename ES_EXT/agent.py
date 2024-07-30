import logging
import threading
import time
import traceback

# import ModelParser
# import Models
import utils
from ACI import ACI
from monitor.DeviceMetricReporter import CyclicArray
from services.CV.YoloDetector import YoloDetector
from services.LI.LidarProcessor import LidarProcessor
from services.QR.QrDetector import QrDetector

DEVICE_NAME = utils.get_ENV_PARAM("DEVICE_NAME", "Unknown")
CLEAN_RESTART = utils.get_ENV_PARAM("CLEAN_RESTART", False)
DISABLE_ACI = utils.get_ENV_PARAM("DISABLE_ACI", False)
SEND_SYSTEM_STATS = utils.get_ENV_PARAM("SEND_SYSTEM_STATS", False)
SHOW_IMG = utils.get_ENV_PARAM("SHOW_IMG", True)
SERVICE_NAME = utils.get_ENV_PARAM("SERVICE_NAME", "CV")

if SERVICE_NAME == "CV":
    service = YoloDetector("localhost", show_results=False)
    desc = {'type': SERVICE_NAME, 'slo_vars': ['in_time', 'rate_60', 'energy_saved']}
elif SERVICE_NAME == "QR":
    service = QrDetector("localhost", show_results=False)
    desc = {'type': SERVICE_NAME, 'slo_vars': ['in_time', 'energy_saved']}
elif SERVICE_NAME == "LI":
    service = LidarProcessor("localhost", show_results=False)
    desc = {'type': SERVICE_NAME, 'slo_vars': ['in_time', 'energy_saved']}
else:
    raise RuntimeError("Why?")

model_name = None if CLEAN_RESTART else utils.create_model_name(SERVICE_NAME, DEVICE_NAME)
aci = ACI(desc, load_model='models/' + model_name, show_img=SHOW_IMG)

c_pixel = ACI.pixel_list[1]
c_fps = ACI.fps_list[2]

logger = logging.getLogger("vehicle")
override_next_config = None
metrics_buffer = CyclicArray(5000)

inferred_config_hist = []


# http_client = HttpClient(HOST=HTTP_SERVER)


def processing_loop():
    global c_pixel, c_fps
    print(f"Starting service '{SERVICE_NAME}' with SLO vars {aci.s_desc['slo_vars']}")
    while True:
        reality_metrics = service.process_one_iteration({'pixel': c_pixel, 'fps': c_fps})
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
            time.sleep(10.0 if len(inferred_config_hist) <= 4 else 2.0)
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
        # threads = http_client.get_latest_stream_config()[0]
        # if user_input == "+":
        #     http_client.override_stream_config(threads + 1)
        # elif user_input == "-":
        #     http_client.override_stream_config(1 if threads == 1 else (threads - 1))
        if user_input == "i":
            aci.bnl(aci.backup_data)
        elif user_input == "e":
            aci.export_model()
        # elif user_input == "q":
        #     aci.export_model()
        #     sys.exit()
