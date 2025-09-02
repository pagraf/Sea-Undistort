import os
import random
import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
import yaml
import numpy as np
import torch

class SeaUndistort(Dataset):
    """
    PyTorch Dataset
    
    Config-Parameters:
      - root_dir (str)
      - shader_folders (List[str]): Relavant for multiple Shader Versions.
      - pair_mode (str): Image Pair to use.
      - split (dict): i.e. {"train": 0.7, "val": 0.15, "test": 0.15}.
      - seed (int): Seed.
      - proportional_shaders (bool): If shaders should be split proportional or random.
      - transform_mode (str): "resize" or "crop" or "resizecrop".
      - crop_size (int, optional): Size if you selected crop.
      - use_augmentation (bool): Whether to apply augmentation.
    """
    def __init__(self, config, split_mode="train"):
        """
        Args:
            split_mode (str): "train", "val" oder "test"
        """
        super().__init__()
        self.config = config
        self.split_mode = split_mode.lower()
        assert self.split_mode in ["train", "val", "test"], "split_mode must be one of 'train', 'val' or 'test'."

        self.root_dir = config["root_dir"]
        self.shader_folders = sorted(config["shader_folders"])
        self.pair_mode = config["pair_mode"]
        self.split = config["split"]
        self.seed = config["seed"]
        self.proportional_shaders = config.get("proportional_shaders", True)
        self.transform_mode = config.get("transform_mode", "crop").lower()
        self.img_size = config.get("size", 256)
        self.use_augmentation = config.get("use_augmentation", False)
        self.color_jittering = config.get("color_jittering", "light").lower()
        self.negative_values = config.get("negative_values", False)
        self.add_mask = config.get("add_mask", False)

        if self.transform_mode not in ["resize", "crop", "resizecrop"]:
            raise ValueError("transform_mode must be either 'resize', 'crop' or 'resizecrop'.")

        if self.color_jittering not in ["none", "light", "strong"]:
            raise ValueError("color_jittering must be either 'none', 'light' or 'strong'.")

        # 1) load predefined blacklists and testsets
        self.shader_blacklists = {}
        self.shader_testsets = {}
        for shader in self.shader_folders:
            shader_path = os.path.join(self.root_dir, shader)
            blacklist_path = os.path.join(shader_path, "blacklist.txt")
            testset_path = os.path.join(shader_path, "test_list.txt")
            self.shader_blacklists[shader] = self._load_list_file(blacklist_path)
            self.shader_testsets[shader] = self._load_list_file(testset_path)

        # 2) Collect IDs per Shader
        shader_id_dict = self._collect_ids_per_shader()

        # 3) Split in Train/Val/Test
        self.data_ids = self._create_split(shader_id_dict)

        # 4) Generate samples list
        self.samples = []
        for shader, ids in self.data_ids[self.split_mode].items():
            for img_id in ids:
                self.samples.append((shader, img_id))

        self.samples.sort()

        # 5) Prepare transformations
        self.transform = self._build_transforms()
        self.colorJittering = self._build_color_jittering()

    def _build_transforms(self):
        transform_list = []
        transform_list.append(T.ToImage())
        if self.transform_mode == "resize":
            transform_list.append(T.Resize((self.img_size, self.img_size)))
        elif self.transform_mode == "crop":
            if self.split_mode == "train":
                transform_list.append(T.RandomCrop((self.img_size, self.img_size)))
            else:
                transform_list.append(T.CenterCrop((self.img_size, self.img_size)))
        elif self.transform_mode == "resizecrop":
            if self.split_mode == "train":
                transform_list.append(T.RandomResizedCrop((self.img_size, self.img_size), antialias=True))
            else:
                transform_list.append(T.CenterCrop((self.img_size, self.img_size)))
        else:
            raise ValueError(f"Unknown transform_mode: {self.transform_mode}")

        if self.use_augmentation and self.split_mode == "train":
            transform_list.append(T.RandomHorizontalFlip(p=0.5))
            transform_list.append(T.RandomVerticalFlip(p=0.5))

        transform_list.append(T.ToDtype(torch.float32, scale=True)) #(T.ToTensor())

        return T.Compose(transform_list)

    def _build_color_jittering(self):
        transform_list = []
        transform_list.append(T.ToImage())


        if self.split_mode == "train":
            # light
            if self.color_jittering == "light" and self.split_mode == "train":
                transform_list.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05))

            # strong
            if self.color_jittering == "strong" and self.split_mode == "train":
                transform_list.append(T.ColorJitter(brightness=0.3, contrast=0.6, saturation=0.2, hue=0.05))

        transform_list.append(T.ToDtype(torch.float32, scale=True))

        return T.Compose(transform_list)

    def _load_list_file(self, filepath):
        if not os.path.exists(filepath):
            print(f"[WARNUNG] Datei '{filepath}' nicht gefunden. Rückgabe eines leeren Sets.")
            return set()

        with open(filepath, "r") as f:
            lines = f.readlines()

        ids = set()
        for line in lines:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            ids.add(line.zfill(4))

        return ids

    def _collect_ids_per_shader(self):
        shader_id_dict = {}
        for shader_folder in self.shader_folders:
            full_path = os.path.join(self.root_dir, shader_folder)
            if not os.path.isdir(full_path):
                print(f"[WARNUNG] Shader-Ordner '{full_path}' ist kein Verzeichnis oder existiert nicht.")
                shader_id_dict[shader_folder] = set()
                continue

            # looking for render_*.png
            png_files = glob.glob(os.path.join(full_path, "render_*.png"))
            collected_ids = set()
            for file in png_files:
                filename = os.path.basename(file)
                base, ext = os.path.splitext(filename)  # ('render_0001_no_waves', '.png')
                if base.startswith("render_"):
                    potential_id = base[7:11]
                    if potential_id.isdigit():
                        if potential_id not in self.shader_blacklists.get(shader_folder, set()):
                            collected_ids.add(potential_id)
            shader_id_dict[shader_folder] = collected_ids
        return shader_id_dict

    def _create_split(self, shader_id_dict):
        """
        Returns Dictionary:
        {
          "train": {shader_folder: set_of_ids},
          "val":   {shader_folder: set_of_ids},
          "test":  {shader_folder: set_of_ids}
        }
        """
        # random.seed(self.seed)
        rng = np.random.default_rng(self.seed)

        data_split = {
            "train": {},
            "val": {},
            "test": {}
        }

        for shader, all_ids in shader_id_dict.items():
            test_ids_fixed = all_ids.intersection(self.shader_testsets.get(shader, set()))
            remaining = list(all_ids - self.shader_testsets.get(shader, set()))
            remaining.sort()

            rng.shuffle(remaining)

            n = len(remaining)
            n_train = int(n * self.split["train"])
            n_val = int(n * self.split["val"])
            
            train_ids = remaining[:n_train]
            val_ids = remaining[n_train:n_train+n_val]
            test_ids = remaining[n_train+n_val:]

            
            data_split["train"][shader] = set(train_ids)
            data_split["val"][shader] = set(val_ids)
            data_split["test"][shader] = set(test_ids).union(test_ids_fixed)

            print(f"Shader '{shader}': Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)},\nVal={val_ids},\nTest={test_ids}")

        return data_split

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        shader_folder, img_id = self.samples[idx]

        input_path, label_path = self._get_image_paths(shader_folder, img_id)

        try:
            input_img = Image.open(input_path).convert("RGB")
            label_img = Image.open(label_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Fehler beim Laden der Bilder: {input_path} oder {label_path}. Fehler: {e}")

        if self.add_mask:
            base, ext = os.path.splitext(input_path)
            no_sunglint_path = base + "_no_sunglint" + ext

            if not os.path.exists(no_sunglint_path):
                raise FileNotFoundError(f"Bild ohne Sunglint nicht gefunden: {no_sunglint_path}")

            no_sunglint_img = Image.open(no_sunglint_path).convert("RGB")
            input_gray = np.array(input_img.convert("L"))
            no_sunglint_gray = np.array(no_sunglint_img.convert("L"))
            
            diff = np.abs(input_gray.astype(np.int16) - no_sunglint_gray.astype(np.int16)).astype(np.uint8)

            # -------- New Mask Generation Method --------
            lower = self.config.get("lower_threshold", 30)
            upper = self.config.get("upper_threshold", 60)

            clipped = np.clip(diff, lower, upper)

            norm = (clipped.astype(np.float32) - lower) / float(max(upper - lower, 1))

            mask_np = (norm * 255).astype(np.uint8)

            mask = Image.fromarray(mask_np, mode='L')

            # ------- old method used -------
            # threshold_value = self.config.get("mask_threshold", 45)
            # mask_np = (diff > threshold_value) * 255
            # mask = Image.fromarray(np.uint8(mask_np)).convert("L")
        else:
            mask = None

        if self.transform:
            if self.add_mask:
                input_img, label_img, mask = self.transform(input_img, label_img, mask)
            else:
                input_img, label_img = self.transform(input_img, label_img)

        if self.color_jittering != "none":
            input_img, label_img = self.colorJittering(input_img, label_img)

        if self.negative_values:
            input_img = (input_img - 0.5) * 2 
            label_img = (label_img - 0.5) * 2
            if self.add_mask:
                mask = (mask - 0.5) * 2


        out = {'lq': input_img, 'gt': label_img}
        if self.add_mask:
            out['mask'] = mask

        return out

    def _get_image_paths(self, shader_folder, img_id):
        """
        Ermittelt die Pfade zu Input- und Label-Bild basierend auf pair_mode.
        """
        base_dir = os.path.join(self.root_dir, shader_folder)

        #  - normal: "render_XXXX.png"
        #  - ground: "render_XXXX_ground.png"
        #  - no_sunglint: "render_XXXX_no_sunglint.png"
        #  - no_waves: "render_XXXX_no_waves.png"

        normal_file = f"render_{img_id}.png"
        no_sunglint_file = f"render_{img_id}_no_sunglint.png"
        no_waves_file = f"render_{img_id}_no_waves.png"
        ground_file = f"render_{img_id}_ground.png"

        pair_mode = self.pair_mode.lower()

        # paring modes:
        if pair_mode == "normal_no_waves":
            input_path = os.path.join(base_dir, normal_file)
            label_path = os.path.join(base_dir, no_waves_file)
        elif pair_mode == "normal_no_sunglint":
            input_path = os.path.join(base_dir, normal_file)
            label_path = os.path.join(base_dir, no_sunglint_file)
        elif pair_mode == "no_sunglint_no_waves":
            input_path = os.path.join(base_dir, no_sunglint_file)
            label_path = os.path.join(base_dir, no_waves_file)
        elif pair_mode == "normal_filtered_no_waves":
            input_path = os.path.join(base_dir, normal_file)
            label_path = os.path.join(base_dir, no_waves_file)
        else:
            raise ValueError(f"Unbekannter pair_mode: {self.pair_mode}")

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input-Bild nicht gefunden: {input_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label-Bild nicht gefunden: {label_path}")

        return input_path, label_path

def load_config(path_to_config):
    with open(path_to_config, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Example
    config = load_config("config.yaml")

    train_dataset = SeaUndistort(config, split_mode="train")
    val_dataset   = SeaUndistort(config, split_mode="val")
    test_dataset  = SeaUndistort(config, split_mode="test")

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config["batch_size"],
                                               shuffle=True,
                                               num_workers=4,
                                               pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config["batch_size"],
                                             shuffle=False,
                                             num_workers=4,
                                             pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=config["batch_size"],
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True)

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # i.e.:
        # model_output = model(inputs)
        # loss = criterion(model_output, labels)
        # ...
        pass


if __name__ == "__main__":
    main()
