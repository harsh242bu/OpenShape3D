import numpy as np
import json
import sys
import os
import data
# import torch.multiprocessing as mp
# from utils.misc import load_config, dump_config  
# from param import parse_args


lvis = json.load(open("meta_data/split/lvis.json", "r"))
train_all = json.load(open("meta_data/split/train_all.json", "r"))

cat_name = "vodka"
cat_x = []
for obj in lvis:
    if obj["category"] == cat_name:
        cat_x.append(obj)


for obj in cat_x:
    d_path = obj["data_path"][1:]
    obj_data = np.load(d_path, allow_pickle=True).item()

    # print("keys: ", obj_data.keys())
    print("text: ", obj_data["text"])
    print("blip_caption: ", obj_data["blip_caption"])
    print("msft_caption: ", obj_data["msft_caption"])
    # print("retrieval_text: ", obj_data["retrieval_text"])
    print("")


# def main(rank, world_size, cli_args, extras):
#     all_text = []
#     config = load_config(cli_args.config, cli_args = vars(cli_args), extra_args = extras)

#     train_loader = data.make(config, 'train', rank, world_size)
#     target = "/projectnb/ivc-ml/harshk/3d_perception/OpenShape3D/mnt/data/objaverse-processed/merged_for_training_final/Objaverse/000-076/15e1a5ff460d4f1ab090409239a47f4d.npy"
    
#     train_data = np.load(target, allow_pickle=True).item()
#     print("keys: ", train_data.keys())
#     print("text: ", train_data["text"])
#     print("blip_caption: ", train_data["blip_caption"])
#     print("msft_caption: ", train_data["msft_caption"])
#     print("retrieval_text: ", train_data["retrieval_text"])


#     for idx, train_data in enumerate(train_loader):
#         if train_data["data_path"] == target:
#             print("train_data: ", train_data.keys())
#             print("train_data: ", train_data["text"])
#             print("train_data: ", train_data["blip_caption"])
#             print("train_data: ", train_data["msft_caption"])
#             print("train_data: ", train_data["retrieval_text"])
#             print("train_data: ", train_data["xyz"])
#             print("train_data: ", train_data["rgb"])
            

    # for idx, train_data in enumerate(train_loader):
    #     # print("idx: ", idx)
    #     if "text" in train_data.keys():
    #         all_text.append(train_data["text"])
    #     elif "texts" in train_data.keys():
    #         all_text.extend(train_data["texts"])
    #     else:
    #         print("train_data_keys: ", train_data.keys())
    #         print("train_data: ", train_data["data_path"], "\n")
    #     if idx > 50:
    #         break
    
    # print("all_text: ", len(all_text))
    # json.dump(all_text, open("all_text.json", "w"))


# cli_args, extras = parse_args(sys.argv[1:])
# world_size = cli_args.ngpu
# main(0, world_size, cli_args, extras)

# mp.spawn(
#         main,
#         args=(world_size, cli_args, extras),
#         nprocs=world_size
#     )


# file = "meta_data/modelnet40/test_pc.npy"
# pc = np.load(file, allow_pickle=True)
# print("pc: ", pc[0])
# print("pc: ", pc.shape)

# idx = 0
# xyz = pc[idx]["xyz"]
# rgb = pc[idx]["rgb"]

# rgb = rgb.astype(np.float32)
# print(rgb.dtype)
# print("data xyz variance: ", np.var(xyz, axis=0))
# print("data rgb variance: ", np.var(rgb, axis=0))

# print("pc[0]: ", pc[0]["xyz"].shape)
# print("pc[0]: ", pc[0]["rgb"].shape)

# print("pc[123]: ", pc[123]["xyz"].shape)
# print("pc[123]: ", pc[123]["rgb"].shape)

# omni_dict = np.load("/projectnb/ivc-ml/harshk/3d_perception/CurveNet/meta_data/omni_dict.npy", allow_pickle=True).item()
# print("omni_dict: ", omni_dict)

# OMNI_HDF5_DIR = "/projectnb/ivc-ml/harshk/3d_perception/OmniObject3D/data/OpenXD-OmniObject3D-New/raw/point_clouds/hdf5_files/1024"
# OMNI_PLY_DIR = "/projectnb/ivc-ml/harshk/3d_perception/OmniObject3D/data/OpenXD-OmniObject3D-New/raw/point_clouds/ply_files"

# num_points = 1024
# point_dir = os.path.join(OMNI_PLY_DIR, str(num_points))

# obj_list = os.listdir(point_dir)

# print("obj_list: ", obj_list)
# print("obj_list: ", len(obj_list))


# data_path = "mnt/data/objaverse-processed/merged_for_training_final/Objaverse/000-057/e395fa630c5741519792617b1b2db67c.npy"
# data_path = "mnt/data/objaverse-processed/merged_for_training_final/Objaverse/000-050/0b91a41d45ff4833966932f8d7d98e4f.npy"
# data_path = "mnt/data/objaverse-processed/merged_for_training_final/ShapeNet/02801938/dc4a523b039e39bda843bb865a04c01a.npy"
# data = np.load(data_path, allow_pickle=True).item()
# # print("data: ", data["has_text_idx"])
# print("data: ", data.keys())
# ['dataset', 'group', 'id', 'text', 'text_feat', 'blip_caption', 'blip_caption_feat', 'msft_caption', 'msft_caption_feat', 'retrieval_text', 'retrieval_text_feat', 'xyz', 'rgb', 'image_feat']

# print("group: ", data["group"])
# print("id: ", data["id"])
# print("text: ", data["text"])
# print("blip_caption: ", data["blip_caption"])
# print("msft_caption: ", data["msft_caption"])
# print("retrieval_text: ", data["retrieval_text"])

# # print("text_feat: ", data["text_feat"])
# print("text_feat: ", data["text_feat"][0]["original"].shape)
