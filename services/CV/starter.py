from VideoDetector import VideoDetector

vd = VideoDetector(model_path="models/yolov8n.onnx", video_path="data/pamela_reif_cut.mp4")

while True:
    vd.process_one_iteration(params=VideoDetector.VideoDetectorParameters(180, 22))
    # detector.processVideo(video_path=video_path,
    #                       video_info=(c_pixel, c_fps, http_client.get_latest_stream_config()),
    #                       show_result=False)
    # if SEND_SYSTEM_STATS:
    #     http_client.send_system_stats(int(psutil.cpu_percent()), DEVICE_NAME, DISABLE_ACI, detector.gpu_available)
    new_data = True
