import json

from PIL import Image
from torch.utils.data import Dataset


class PetManifestDataset(Dataset):
    def __init__(self, manifest_path, split=None, corruption=None, transform=None):
        """
        split: 'train' | 'test' | None (no filter)
        corruption: None -> only clean images (corruption is None)
                    'any' -> only corrupted images
                    'all' -> clean + corrupted
                    'gaussian_noise' etc -> only that corruption type
        """
        with open(manifest_path) as f:
            entries = json.load(f)

        if split is not None:
            entries = [e for e in entries if e["split"] == split]

        if corruption is None:
            entries = [e for e in entries if e["corruption"] is None]
        elif corruption == "any":
            entries = [e for e in entries if e["corruption"] is not None]
        elif corruption != "all":
            entries = [e for e in entries if e["corruption"] == corruption]
        # corruption == "all" -> no filter, keep everything

        # class_index can be None for some corrupted entries if lookup failed; drop those
        entries = [e for e in entries if e["class_index"] is not None]

        self.entries = entries
        self.transform = transform

        idx_to_name = {}
        for e in entries:
            idx_to_name[e["class_index"]] = e["breed"]
        self.classes = [idx_to_name[i] for i in sorted(idx_to_name)]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        img = Image.open(e["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, e["class_index"]
