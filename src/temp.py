import numpy as np

file = "meta_data/modelnet40/test_pc.npy"
pc = np.load(file, allow_pickle=True)
print("pc: ", pc[0])
print("pc: ", pc.shape)

idx = 0
xyz = pc[idx]["xyz"]
rgb = pc[idx]["rgb"]

rgb = rgb.astype(np.float32)
print(rgb.dtype)
print("data xyz variance: ", np.var(xyz, axis=0))
print("data rgb variance: ", np.var(rgb, axis=0))

# print("pc[0]: ", pc[0]["xyz"].shape)
# print("pc[0]: ", pc[0]["rgb"].shape)

# print("pc[123]: ", pc[123]["xyz"].shape)
# print("pc[123]: ", pc[123]["rgb"].shape)