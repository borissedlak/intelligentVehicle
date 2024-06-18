import itertools

from VideoDetector import VideoDetector

vd = VideoDetector()

for (source_pixel, source_fps) in itertools.product([480, 720, 1080], [5, 10, 15, 20]):

    print(f"Now processing: {source_pixel} p, {source_fps} FPS")
    for i in range(150):
        vd.process_one_iteration(params=(source_pixel, source_fps))
