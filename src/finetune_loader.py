import json
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset

from utils.data import normalize_pc


train_lvis_filtered = "/projectnb/ivc-ml/harshk/3d_perception/OpenShape3D/finetune/lvis_filter_train.json"
test_lvis_filtered = "/projectnb/ivc-ml/harshk/3d_perception/OpenShape3D/finetune/lvis_filter_test.json"

class FinetuneLoader(Dataset):
    def __init__(self, config, split="train"):
        self.y_up = config.dataset.y_up
        self.normalize = config.dataset.normalize
        self.use_color = config.dataset.use_color
        self.num_points = config.objaverse_lvis.num_points
        self.text_source = config.dataset.text_source
        self.use_text_filtering = config.dataset.use_text_filtering
        self.use_prompt_engineering = config.dataset.use_prompt_engineering

        self.objaverse_lvis_split = json.load(open(config.objaverse_lvis.split, "r"))
        self.categories = sorted(np.unique([data['category'] for data in self.objaverse_lvis_split]))
        self.category2idx = {self.categories[i]: i for i in range(len(self.categories))}

        self.clip_cat_feat = np.load(config.objaverse_lvis.clip_feat_path, allow_pickle=True)
        self.text_embed_version = "prompt_avg" if self.use_prompt_engineering else "original"
        if self.use_text_filtering:
            self.gpt4_filtering = json.load(open(config.dataset.gpt4_filtering_path, "r"))

        if split == "train":
            train_file = json.load(open(train_lvis_filtered, "r"))
            data = []
            for k, v in train_file.items():
                data.extend(v)
            self.data = data

        elif split == "test":
            test_file = json.load(open(test_lvis_filtered, "r"))
            data = []
            for k, v in test_file.items():
                data.extend(v)
            self.data = data
        elif split == "all":
            train_file = json.load(open(train_lvis_filtered, "r"))
            test_file = json.load(open(test_lvis_filtered, "r"))
            data = []
            for k, v in train_file.items():
                data.extend(v)
            for k, v in test_file.items():
                data.extend(v)
            self.data = data
        else:
            raise NotImplementedError

    def __getitem__(self, index: int):
        meta = self.data[index]
        data_path = meta["data_path"][1:]
        uid = meta["uid"]

        raw_data = np.load(data_path, allow_pickle=True).item()
        
        n = raw_data['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = raw_data['xyz'][idx]
        rgb = raw_data['rgb'][idx]

        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.use_color:
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz
        
        assert not np.isnan(xyz).any()

        text_feat = []
        texts = []
        if 'text' in self.text_source:
            if not (self.use_text_filtering and self.gpt4_filtering[uid]["flag"] == "N"):
                texts.append(raw_data["text"][0])
                text_feat.append(raw_data["text_feat"][0][self.text_embed_version])
        
        if 'caption' in self.text_source:
            if np.random.rand() < 0.5:
                if len(raw_data["blip_caption"]) > 0:
                    texts.append(raw_data["blip_caption"])
                    text_feat.append(raw_data["blip_caption_feat"][self.text_embed_version])
            else:
                if len(raw_data["msft_caption"]) > 0:
                    texts.append(raw_data["msft_caption"])
                    text_feat.append(raw_data["msft_caption_feat"][self.text_embed_version])
        
        if 'retrieval_text' in self.text_source:
            if len(raw_data["retrieval_text"]) > 0:
                idx = np.random.randint(len(raw_data["retrieval_text"]))
                texts.append(raw_data["retrieval_text"][idx])
                text_feat.append(raw_data["retrieval_text_feat"][idx]["original"]) # no prompt engineering for retrieval text

        if len(text_feat) > 0:
            assert len(text_feat) == len(texts)
            text_idx = np.random.randint(len(texts))
            text_feat = text_feat[text_idx]
            texts = texts[text_idx]
            text_feat = torch.from_numpy(text_feat).type(torch.float32).reshape(-1)
        else:
            text_feat = None
            texts = None

        if np.random.rand() < 0.5:
            img_feat = raw_data['thumbnail_feat']
        else:
            idx = np.random.randint(raw_data['image_feat'].shape[0])
            img_feat = raw_data["image_feat"][idx]

        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "img_feat": torch.from_numpy(img_feat).type(torch.float32).reshape(-1),
            "dataset": "Objaverse",
            "data_path": meta["data_path"],
            "group": meta["group"],
            "name":  uid,
            "category": meta["category"],
            "category2idx": self.category2idx[meta["category"]],
            "texts": texts,
            "text_feat": text_feat,
            "has_text": text_feat is not None,
        }

    def __len__(self):
        return len(self.data)
