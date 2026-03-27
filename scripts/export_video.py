import h5py
from glob import glob
import os
import cv2

def write_video(frames, video_path, fps=20):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for frame in frames:
        video_writer.write(frame[::-1, ::-1, ::-1])  # Convert RGB to BGR for OpenCV

    video_writer.release()
    print(f"Video saved to {video_path}")
    

paths = glob("./assets/*.hdf5")
for i, path in enumerate(paths):
    f = h5py.File(path, 'r')
    for demo_i in f['data'].keys():
        print(demo_i)
        frames = f['data'][demo_i]['obs']['agentview_rgb'][()]
        success = f['data'][demo_i]['success'][()]  
        fold = f"./assets/{i}"
        os.makedirs(fold, exist_ok=True)
        write_video(frames, f"{fold}/{demo_i}_{success}.mp4", fps=20)
f.close()
