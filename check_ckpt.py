import torch
ckpt = torch.load("/workspace/checkpoints/epoch-47_miou_67.15.pth", map_location="cpu")
print("Keys:", ckpt.keys() if hasattr(ckpt, "keys") else "not dict")
if 'config' in ckpt:
    print("Config:", ckpt['config'])
