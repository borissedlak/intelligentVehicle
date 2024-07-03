import itertools

from QrDetector import QrDetector

qd = QrDetector("localhost", show_results=False)

for (source_pixel, source_fps) in itertools.product([1080, 720, 1080], [100, 10, 15, 20]):

    print(f"Now processing: {source_pixel} p, {source_fps} FPS")
    for i in range(9999):
        metrics = qd.process_one_iteration(params={'pixel': source_pixel, 'fps': source_fps})
        print(metrics)
