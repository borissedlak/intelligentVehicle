import itertools

from LidarProcessor import LidarProcessor

ld = LidarProcessor("localhost")

for (source_pixel, source_fps) in itertools.product([480, 720, 1080], [5, 10, 15, 20]):

    print(f"Now processing: {source_pixel} p, {source_fps} FPS")
    for i in range(150):
        metrics = ld.process_one_iteration(params={'pixel': source_pixel, 'fps': source_fps, 'mode': 'double'})
        print(metrics)
