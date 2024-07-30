import logging
import sys
import threading
import time
import traceback

import utils
from ACI import ACI
from ACI_LI import ACI_LI
from ES_EXT import util_fgcs
from monitor.DeviceMetricReporter import CyclicArray
from services.CV.YoloDetector import YoloDetector
from services.LI.LidarProcessor import LidarProcessor
from services.QR.QrDetector import QrDetector

DEVICE_NAME = utils.get_ENV_PARAM("DEVICE_NAME", "Unknown")
CLEAN_RESTART = utils.get_ENV_PARAM("CLEAN_RESTART", False)
DISABLE_ACI = utils.get_ENV_PARAM("DISABLE_ACI", False)
SEND_SYSTEM_STATS = utils.get_ENV_PARAM("SEND_SYSTEM_STATS", False)
SHOW_IMG = utils.get_ENV_PARAM("SHOW_IMG", True)
SERVICE_NAME = utils.get_ENV_PARAM("SERVICE_NAME", "LI")
INITIAL_TRAINING = float(utils.get_ENV_PARAM("INITIAL_TRAINING", 5))
EXPERIMENT_DURATION = float(utils.get_ENV_PARAM("EXPERIMENT_DURATION", 30))

c_pixel = ACI.pixel_list[1]
c_fps = ACI.fps_list[2]

model_name = None if CLEAN_RESTART else utils.create_model_name(SERVICE_NAME, DEVICE_NAME)
if SERVICE_NAME == "CV":
    service = YoloDetector("localhost", show_results=False)
    desc = {'type': SERVICE_NAME, 'slo_vars': ['in_time', 'rate_60', 'energy_saved']}
    aci = ACI(desc, load_model='ES_EXT/models/' + model_name, show_img=SHOW_IMG)
elif SERVICE_NAME == "QR":
    service = QrDetector("localhost", show_results=False)
    desc = {'type': SERVICE_NAME, 'slo_vars': ['in_time', 'energy_saved']}
    aci = ACI(desc, load_model='ES_EXT/models/' + model_name, show_img=SHOW_IMG)
elif SERVICE_NAME == "LI":
    service = LidarProcessor("localhost", show_results=False)
    desc = {'type': SERVICE_NAME, 'slo_vars': ['in_time', 'energy_saved']}
    aci = ACI_LI(desc, load_model='ES_EXT/models/' + model_name, show_img=SHOW_IMG)
    c_pixel = ACI.mode_list[1]
else:
    raise RuntimeError("Why?")

logger = logging.getLogger("vehicle")
metrics_buffer = CyclicArray(5000)

inferred_config_hist = []
slo_f_hist = []


def processing_loop():
    global c_pixel, c_fps
    while True:
        if SERVICE_NAME == "LI":
            params = {'mode': c_pixel, 'fps': c_fps}
        else:
            params = {'pixel': c_pixel, 'fps': c_fps}
        reality_metrics = service.process_one_iteration(params)
        metrics_buffer.append(reality_metrics)


processing_thread = threading.Thread(target=processing_loop, daemon=True)
processing_thread.start()


class ACIBackgroundThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True  # Set the thread as a daemon, so it will exit when the main program exits

    def run(self):
        global c_pixel, c_fps, inferred_config_hist
        while True:
            time.sleep(INITIAL_TRAINING if len(inferred_config_hist) <= 4 else 2.0)
            try:
                if metrics_buffer.is_empty():
                    continue
                else:
                    input_metrics = metrics_buffer.get()
                    metrics_buffer.clear()
                    (new_pixel, new_fps, pv, real, surprise) = aci.iterate(input_metrics)
                    inferred_config_hist.append((new_pixel, new_fps))
                    slo_f_hist.append([real, surprise])

                    if (c_pixel, c_fps) != (new_pixel, new_fps):
                        print(f"Changing configuration to {(new_pixel, new_fps)}")
                    c_pixel, c_fps = new_pixel, new_fps
            except Exception as e:
                error_traceback = traceback.format_exc()
                print("Error Traceback:")
                print(error_traceback)
                logger.error(f"ACI Background thread encountered an exception:{e}")


if not DISABLE_ACI:
    aci_thread = ACIBackgroundThread()
    aci_thread.start()

print(f"Starting service '{SERVICE_NAME}' with SLO vars {aci.s_desc['slo_vars']} at {DEVICE_NAME}")
time.sleep(EXPERIMENT_DURATION)
aci.export_model()
util_fgcs.log_performance(SERVICE_NAME, DEVICE_NAME, slo_f_hist)
print("Finished Experiment")
sys.exit()
