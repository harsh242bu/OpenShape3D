import torch
import json
import numpy as np

split = json.load(open("meta_data/split/lvis.json", "r"))
print("split: ", len(split))

for i in range(len(split)):
    print("i: ", i)
    data = np.load(split[i]['data_path'], allow_pickle=True).item()

# data = np.load(split, allow_pickle=True)

# print("data: ", data)
# print("shape: ", data.shape)
# data = np.load(split[index]['data_path'], allow_pickle=True).item()

# def test_scanobjectnn(self):
#     self.model.eval()
#     if self.config.training.use_text_proj:
#         self.text_proj.eval()
#     clip_text_feat = torch.from_numpy(self.scanobjectnn_loader.dataset.clip_cat_feat).to(self.config.device)
#     if self.config.training.use_text_proj:
#         clip_text_feat = self.text_proj(clip_text_feat)
#     per_cat_correct = torch.zeros(15).to(self.config.device)
#     per_cat_count = torch.zeros(15).to(self.config.device)
#     category2idx = self.scanobjectnn_loader.dataset.category2idx
#     idx2category = {v: k for k, v in category2idx.items()}
    
#     logits_all = []
#     labels_all = []
#     with torch.no_grad():
#         for data in self.scanobjectnn_loader:
#             if not self.config.model.get("use_dense", False):
#                 pred_feat = self.model(data['xyz'], data['features'], \
#                                         device = self.config.device, \
#                                         quantization_size = self.config.model.voxel_size)
#             else:
#                 pred_feat = self.model(data['xyz_dense'], data['features_dense'])
#             logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
#             labels = data['category'].to(self.config.device)
#             logits_all.append(logits.detach())
#             labels_all.append(labels)
#             # calculate per class accuracy
#             for i in range(15):
#                 idx = (labels == i)
#                 if idx.sum() > 0:
#                     per_cat_correct[i] += (logits[idx].argmax(dim=1) == labels[idx]).float().sum()
#                     per_cat_count[i] += idx.sum()

#     topk_acc, correct = self.accuracy(torch.cat(logits_all), torch.cat(labels_all), topk=(1,3,5,))

#     overall_acc = per_cat_correct.sum() / per_cat_count.sum()
#     per_cat_acc = per_cat_correct / per_cat_count

#     logging.info('Test ScanObjectNN: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
#     logging.info('Test ScanObjectNN: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()))
#     wandb.log({"test_scanobjectnn/epoch": self.epoch,
#                 "test_scanobjectnn/step": self.step,
#                 "test_scanobjectnn/overall_acc": overall_acc,
#                 "test_scanobjectnn/class_acc": per_cat_acc.mean(),
#                 "test_scanobjectnn/top3_acc": topk_acc[1],
#                 "test_scanobjectnn/top5_acc": topk_acc[2],})
