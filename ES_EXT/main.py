import itertools
import json
import sys

import paho.mqtt.client as mqtt

import Models
from VideoDetector import *


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker")
        # Subscribe to the desired topic(s) here
        client.subscribe('tele/delock/SENSOR')
    else:
        print("Failed to connect, return code: ", rc)


def on_message(client, userdata, msg):
    payload = msg.payload.decode()
    o = json.loads(payload)
    cons = o['ENERGY']['Power']

    print(f"Current Consumption: {cons}W")
    util.write_to_blank_file(cons)


def on_disconnect(client, userdata, rc):
    if rc != 0:
        print("Unexpected disconnection")


def setup():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    broker_address = "192.168.1.153"
    port = 1883  # Default MQTT port
    client.connect(broker_address, port=port)

    client.loop_start()


# process = multiprocessing.Process(target=setup())
# # process.start()
# process.run()
# time.sleep(0.1)

# privacy_model = ModelParser.parseModel("video:{'tag':'recording'}-->Gender_Trigger:{'prob':0.85}-->Blur_Area_Simple:{'blocks':5}")
privacy_model_1 = ModelParser.parseModel(Models.model_1)
# privacy_model_1_1 = ModelParser.parseModel(Models.model_1_1)
# privacy_model_1_50 = ModelParser.parseModel(Models.model_1_50)
# privacy_model_2 = ModelParser.parseModel(Models.model_2)
# privacy_model_3 = ModelParser.parseModel(Models.model_3)
# privacy_model_4 = ModelParser.parseModel(Models.model_4)

chain_1 = privacy_model_1.getChainForSource("video", "webcam")
# chain_1_1 = privacy_model_1_1.getChainForSource("video", "webcam")
# chain_1_50 = privacy_model_1_50.getChainForSource("video", "webcam")
# chain_2 = privacy_model_2.getChainForSource("video", "webcam")
# chain_3 = privacy_model_3.getChainForSource("video", "webcam")
# chain_4 = privacy_model_4.getChainForSource("video", "webcam")

detector_1 = VideoDetector(privacy_chain=chain_1, display_stats=True, write_stats=True, simulate_fps=True)
# detector_1_1 = VideoDetector(privacy_chain=chain_1_1, display_stats=True, write_stats=True)
# detector_1_50 = VideoDetector(privacy_chain=chain_1_50, display_stats=True, write_stats=True)
# detector_2 = VideoDetector(privacy_chain=chain_2, display_stats=True, write_stats=True)
# detector_3 = VideoDetector(privacy_chain=chain_3, display_stats=True, write_stats=True)
# detector_4 = VideoDetector(privacy_chain=chain_4, display_stats=True, write_stats=True)

# detector_1.processImage("../producer/demo_files/images/head.jpg", show_result=True)
# detector_1_1.processVideo(video_path="../producer/demo_files/videos/", video_name="video_1", model_name="model_1_1", show_result=True)
# detector_1_50.processVideo(video_path="../producer/demo_files/videos/", video_name="video_1", model_name="model_1_50", show_result=True)

# ("120p", 16), ("120p", 16), ("240p", 16), ("360p", 16), ("480p", 16), ("720p", 16),
#                                     ("360p", 30), ("360p", 45), ("360p", 60), ("480p", 30), ("480p", 45), ("480p", 60)

detector_1.processVideo(video_path="../video_data/",
                        video_info=list(
                            itertools.product(["120p", "180p", "240p", "300p", "360p", "420p"], [12, 16, 20, 26, 30],
                                              [1, 2, 3, 4, 5, 6])),
                        model_name="model_1", show_result=False, repeat=5)
# detector_1.processVideo(video_path="../video_data/",
#                         video_info=[("720p", 30)],
#                         model_name="model_1", show_result=False)
# detector_2.processVideo(video_path="../producer/demo_files/videos/", video_name="video_1", model_name="model_1", show_result=False)
# detector_1.processVideo(video_path="../producer/demo_files/videos/", video_name="video_loop_1", model_name="model_1", show_result=False)
# detector_2.processVideo(video_path="../producer/demo_files/videos/", video_name="video_1", model_name="model_2", show_result=True)
# detector_3.processVideo(video_path="../producer/demo_files/videos/", video_name="video_1", model_name="model_3", show_result=True)
# detector_4.processVideo(video_path="../producer/demo_files/videos/", video_name="video_1", model_name="model_4", show_result=True)

# detector_1.processVideo(video_path="../producer/demo_files/videos/", video_name="video_2", model_name="model_1", show_result=True)
# detector_2.processVideo(video_path="../producer/demo_files/videos/", video_name="video_2", model_name="model_2", show_result=True)
# detector_3.processVideo(video_path="../producer/demo_files/videos/", video_name="video_2", model_name="model_3", show_result=True)
# detector_4.processVideo(video_path="../producer/demo_files/videos/", video_name="video_2", model_name="model_4", show_result=True)

sys.exit()
