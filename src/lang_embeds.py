import torch
import numpy as np
from numpy.linalg import norm
from sklearn.cluster import KMeans, DBSCAN
import torch.nn.functional as F


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clip_feat_path = "meta_data/lvis_cat_name_pt_feat.npy"
clip_cat_feat = np.load(clip_feat_path, allow_pickle=True)
objaverse_dict = torch.load("src/eval_data/objaverse_dict.pt")
ov_category2idx = objaverse_dict["category2idx"]
ov_idx2category = objaverse_dict["idx2category"]
ov_category2idx = objaverse_dict["category2idx"] 

# print("clip_cat_feat: ", clip_cat_feat.shape) # (1156, 1280)
# clip_cat_feat = F.normalize(clip_cat_feat, dim=-1)
clip_cat_feat = clip_cat_feat / norm(clip_cat_feat, axis=1, keepdims=True)

eps = 0.61855
db_scan = DBSCAN(eps=eps, min_samples=2).fit(clip_cat_feat)
dbscan_labels = db_scan.labels_

obj_idx = ov_category2idx["sherbert"]
val = dbscan_labels[obj_idx]
idx = np.argwhere(dbscan_labels == val)

obj_list = []
for i in idx:
    obj_list.append(ov_idx2category[i.item()])

print("obj_list: ", len(obj_list))
print("obj_list: ", obj_list)

# print("dbscan: ", db_scan.labels_)
print("eps: ", eps)
print("dbscan: ", np.unique(db_scan.labels_))

# kmeans = KMeans(n_clusters=84, random_state=2023).fit(clip_cat_feat)
# print("labes: ", kmeans.labels_.shape)
# km_labels = kmeans.labels_
# obj_idx = ov_category2idx["boiled_egg"]
# val = km_labels[obj_idx]
# idx = np.argwhere(km_labels == val)

# obj_list = []
# for i in idx:
#     obj_list.append(ov_idx2category[i.item()])

# print("obj_list: ", obj_list)

# kmeans.predict([[0, 0], [12, 3]])
# print("cluster_centers: ", kmeans.cluster_centers_.shape)
# kmeans.cluster_centers_


# clip_cat_feat = torch.from_numpy(clip_cat_feat).to(device)
# clip_text_feat = text_proj(clip_text_feat)

# objaverse_dict = torch.load("src/eval_data/objaverse_dict.pt")
# ov_xyz = torch.load("src/eval_data/ov_xyz.pt")
# ov_idx2category = objaverse_dict["idx2category"]
# ov_category2idx = objaverse_dict["category2idx"]
# ov_labels_all = objaverse_dict["labels_all"]
# print("ov_labels_all: ", ov_labels_all.shape)

# ov_confusion = torch.load("src/eval_data/ov_confusion.pt")
# print("ov_confusion: ", ov_confusion.shape)