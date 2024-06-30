import os
import time

import cv2
import numpy as np
import torch

import services.LI.config.kitti_config as cnf
import utils
from monitor.DeviceMetricReporter import DeviceMetricReporter
from monitor.ServiceMetricReporter import ServiceMetricReporter
from services.LI.data_process.demo_dataset import Demo_KittiDataset
from services.LI.data_process.kitti_data_utils import Calibration
from services.LI.data_process.transformation import lidar_to_camera_box
from services.LI.models.model_utils import create_model
from services.LI.util.demo_utils import download_and_unzip, do_detect
from services.LI.util.demo_utils import parse_demo_configs
from services.LI.util.evaluation_utils import draw_predictions, convert_det_to_real_values
from services.LI.util.visualization_utils import show_rgb_image_with_boxes, merge_rgb_to_bev
from services.VehicleService import VehicleService

DEVICE_NAME = utils.get_ENV_PARAM("DEVICE_NAME", "Unknown")


class LidarProcessor(VehicleService):

    def __init__(self, show_results=False):
        super().__init__()
        self.simulate_fps = True

        self.device_metric_reporter = DeviceMetricReporter()
        self.service_metric_reporter = ServiceMetricReporter("LI")

        self.show_result = show_results

        self.configs = parse_demo_configs()
        self.model, self.demo_dataset = None, None

        self.ensure_demo_data()
        self.sample_index = 0

    def process_one_iteration(self, params):
        # TODO: Must pass parameter from outside
        source_pixel, source_fps, proc_mode = int(params['pixel']), int(params['fps']), 'double'  # params['mode']

        available_time_frame = (1000 / source_fps)
        start_time = time.time()

        with torch.no_grad():
            if proc_mode == "double":
                combined_img = self.process_two_side(self.sample_index)
            else:
                combined_img = self.process_front(self.sample_index)

            self.sample_index = self.sample_index + 1 if self.sample_index < len(self.demo_dataset) - 1 else 0

            if self.show_result:
                cv2.imshow("Detected Objects", combined_img)

        processing_time = (time.time() - start_time) * 1000.0

        service_blanket = self.service_metric_reporter.create_metrics(processing_time, source_fps, source_pixel)
        device_blanket = self.device_metric_reporter.create_metrics()
        merged_metrics = utils.merge_single_dicts(service_blanket["metrics"], device_blanket["metrics"])

        if self.simulate_fps:
            if processing_time < available_time_frame:
                time.sleep((available_time_frame - processing_time) / 1000)

        return merged_metrics

    def ensure_demo_data(self):
        server_url = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data'
        download_url = '{}/{}/{}.zip'.format(server_url, self.configs.foldername[:-5], self.configs.foldername)
        download_and_unzip(self.configs.dataset_dir, download_url)

        self.model = create_model(self.configs)
        # print('\n\n' + '-*=' * 30 + '\n\n')
        assert os.path.isfile(self.configs.pretrained_path), "No file at {}".format(self.configs.pretrained_path)
        self.model.load_state_dict(torch.load(self.configs.pretrained_path, map_location='cpu'))
        # print('Loaded weights from {}\n'.format(self.configs.pretrained_path))

        self.configs.device = torch.device('cpu' if self.configs.no_cuda else 'cuda:{}'.format(self.configs.gpu_idx))
        self.model = self.model.to(device=self.configs.device)
        self.model.eval()

        self.demo_dataset = Demo_KittiDataset(self.configs)

    def process_front(self, sample_idx):
        metadata, bev_map, img_rgb = self.demo_dataset.load_bevmap_front(sample_idx)
        detections, bev_map, fps = do_detect(self.configs, self.model, bev_map, is_front=True)
        bev_map = (bev_map.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        bev_map = cv2.resize(bev_map, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
        bev_map = draw_predictions(bev_map, detections, self.configs.num_classes)
        bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        calib = Calibration(self.configs.calib_path)
        kitti_dets = convert_det_to_real_values(detections)

        if len(kitti_dets) > 0:
            kitti_dets[:, 1:] = lidar_to_camera_box(kitti_dets[:, 1:], calib.V2C, calib.R0, calib.P2)
            img_bgr = show_rgb_image_with_boxes(img_bgr, kitti_dets, calib)

        out_img = merge_rgb_to_bev(img_bgr, bev_map, output_width=self.configs.output_width)
        return out_img

    def process_two_side(self, sample_idx):
        metadatas, front_bevmap, back_bevmap, img_rgb = self.demo_dataset.load_bevmap_front_vs_back(sample_idx)
        front_detections, front_bevmap, fps = do_detect(self.configs, self.model, front_bevmap, is_front=True)
        back_detections, back_bevmap, _ = do_detect(self.configs, self.model, back_bevmap, is_front=False)

        # Draw prediction in the image
        front_bevmap = (front_bevmap.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        front_bevmap = cv2.resize(front_bevmap, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
        front_bevmap = draw_predictions(front_bevmap, front_detections, self.configs.num_classes)
        # Rotate the front_bevmap
        front_bevmap = cv2.rotate(front_bevmap, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Draw prediction in the image
        back_bevmap = (back_bevmap.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        back_bevmap = cv2.resize(back_bevmap, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
        back_bevmap = draw_predictions(back_bevmap, back_detections, self.configs.num_classes)
        # Rotate the back_bevmap
        back_bevmap = cv2.rotate(back_bevmap, cv2.ROTATE_90_CLOCKWISE)

        # merge front and back bevmap
        full_bev = np.concatenate((back_bevmap, front_bevmap), axis=1)

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        calib = Calibration(self.configs.calib_path)
        kitti_dets = convert_det_to_real_values(front_detections)
        if len(kitti_dets) > 0:
            kitti_dets[:, 1:] = lidar_to_camera_box(kitti_dets[:, 1:], calib.V2C, calib.R0, calib.P2)
            img_bgr = show_rgb_image_with_boxes(img_bgr, kitti_dets, calib)
        img_bgr = cv2.resize(img_bgr, (cnf.BEV_WIDTH * 2, 375))

        out_img = np.concatenate((img_bgr, full_bev), axis=0)
        return out_img
