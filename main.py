import numpy as np

from vsr import VSRPipeline


def main():
    pipeline = VSRPipeline(k=3)
    stream = pipeline.stream()

    lr_frames = [
        np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(10)
    ]
    batch_size = 2

    hr_frames = []
    for start in range(0, len(lr_frames), batch_size):
        batch = lr_frames[start : start + batch_size]
        hr_frames.extend(stream.push(batch))

    hr_frames.extend(stream.flush())

    print(f"{len(lr_frames)} LR frames in -> {len(hr_frames)} HR frames out")

    for i in range(len(lr_frames)):
        lr = lr_frames[i]
        hr = hr_frames[i]
        print(
            f"frame {i}: lr_frame: {lr.shape}, ({type(lr)}), hr_frame: {hr.shape}, ({type(hr)}))"
        )


if __name__ == "__main__":
    main()
