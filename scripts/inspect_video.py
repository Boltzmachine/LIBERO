import h5py

import cv2

def write_video(frames, video_path, fps=20):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for frame in frames:
        video_writer.write(frame[::-1, ::-1, ::-1])  # Convert RGB to BGR for OpenCV

    video_writer.release()
    print(f"Video saved to {video_path}")
    
    
f = h5py.File("/home/qiuweikang/project/LIBERO/libero/libero/../datasets/libero_stove/KITCHEN_SCENE_heat_the_tomato_sauce_on_the_stove_for_1_second_and_then_heat_the_alphabet_soup_demo.hdf5", 'r')
frames = f['data']['demo_0']['obs']['agentview_rgb'][()]
write_video(frames, "test_video2.mp4", fps=20)
f.close()