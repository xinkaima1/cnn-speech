import torch, torchaudio
print(torch.__version__, torchaudio.__version__)
print("CUDA 可用:", torch.cuda.is_available())
print("CUDA 版本:", torch.version.cuda)
print("显卡:", torch.cuda.get_device_name(0))
print(torch.randn(2, 3, device="cuda").mean())
