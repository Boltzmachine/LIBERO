import h5py
import numpy as np

f = h5py.File("/home/qiuweikang/project/LIBERO/libero/libero/../datasets/moving1/KITCHEN_SCENE1_move_the_tomato_sauce_to_the_milk's_original_position_demo.hdf5", 'r')
lengths = []
for demo_key in f['data'].keys():
    length = f['data'][demo_key]['states'].shape[0]
    lengths.append(length)
print("mean length:", np.mean(lengths))
print("max length:", np.max(lengths))
print("min length:", np.min(lengths))
f.close()