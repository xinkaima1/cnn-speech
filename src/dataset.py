import glob
import random

import torch
import torchaudio                      # torchaudio.load()：读音频文件用
import torchaudio.transforms as T      # transforms = 音频变换工具箱，Mel 频谱在这里
from torch.utils.data import Dataset, DataLoader

# 前置修复：RAVDESS 原生 48 kHz → 降到语音标准 16 kHz
# （MelSpectrogram 的 sample_rate 参数只定频率刻度、不重采样音频，必须手动降，
#   否则频率错位 3 倍、hop=125 每秒产生 384 帧 → 128 帧只剩 0.33 秒语音）
resample = T.Resample(orig_freq=48000, new_freq=16000)   # 48kHz → 16kHz

# 功能：波形→Mel 频谱 | 参数注释：名称 / 含义 / 默认 / 调整指导
melspec = T.MelSpectrogram(
    sample_rate=16000,   # 采样率 / 与重采样后的波形一致 / 固定 16000 / 勿动
    n_fft=1024,          # FFT 窗长 / 1024/16000=64ms，频率分辨率≈15.6Hz / 默认 400 / 勿动
    hop_length=125,      # 帧移 / 每秒帧数=16000/125=128 帧 / 与目标 128 帧匹配 / 勿动
    n_mels=64,           # Mel 频带数 / 输入"图像"的高度 / 默认 64 / 32~128 皆可
)

def wav_to_logmel(wav, target_frames=128):
    # 修复1：数据里有 5 条立体声文件 → 双声道取平均合成单声道（否则拼不进 batch）
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)   # dim=0 是声道维；keepdim 让形状保持 [1,T]
    # 修复2：48kHz → 16kHz（关键一行：不降采样则下面所有参数的解释错位 3 倍）
    wav = resample(wav)
    mel = melspec(wav)                    # ① 波形[1,T] → Mel 频谱 [1, 64, T]，T 随语音长度变化
    mel = torch.log(mel + 1e-6)           # ② log 压缩（+1e-6 防止 log(0) 报错）
    # ③ 固定长度：太长 → 中心裁剪；太短 → 右侧补零（CNN 要求所有输入同尺寸）
    n_frames = mel.shape[2]               # 当前时间帧数（shape 的第 3 个数）
    if n_frames > target_frames:
        start = (n_frames - target_frames) // 2   # 从中间开始裁，保留语音主体
        mel = mel[:, :, start:start + target_frames]
    elif n_frames < target_frames:
        pad = target_frames - n_frames             # 缺多少帧
        mel = torch.nn.functional.pad(mel, (0, pad))  # (0,pad) = 只在最后一维（时间）右侧补零
    return mel                            # [1, 64, 128]
# # 测试代码
# # if __name__ == "__main__": 的意思是"直接运行本文件时才执行下面的代码"，
# # 被 train.py import 时会自动跳过——所以放心留在文件里，不干扰任何后续步骤。
# if __name__ == "__main__":
#     import torchaudio
#     import glob, random
#     import matplotlib
#     matplotlib.use("Agg")           # 无屏幕环境也能存图（WSL 没有图形界面）
#     import matplotlib.pyplot as plt

#     # 1) 随机挑一条真实语音（glob = 按通配符列文件，* 号匹配任意名）
#     files = glob.glob("data/Actor_01/*.wav")
#     path = random.choice(files)

#     # 2) 读文件 → 过你写好的 wav_to_logmel
#     wav, sr = torchaudio.load(path)
#     print("sample rate:", sr)         # 预期 48000——RAVDESS 原生采样率，这正是函数里要重采样的原因
#     logmel = wav_to_logmel(wav)

#     # 3) 对照预期：torch.Size([1, 64, 128])
#     print("file:", path)              # 随机抽中的这条语音的路径
#     print("output shape:", logmel.shape)  # 预期 torch.Size([1, 64, 128])

#     # 4) 存频谱图（不是弹窗口，是存成文件——去项目根目录找它）
#     plt.figure(figsize=(8, 3))
#     plt.imshow(logmel[0].numpy(), origin="lower", aspect="auto")
#     plt.xlabel("Time Frame"); plt.ylabel("Mel Bin")  # 时间帧 / Mel 频带——标签用英文（DejaVu 无汉字字形，中文会画成方块□）
#     plt.savefig("fig_mel_sample.png", dpi=150, bbox_inches="tight")
#     print("saved to fig_mel_sample.png")  # 已保存 = 跑通了，去项目根目录看图
class RavdessDataset(Dataset):
    def __init__(self, file_list):
        # file_list: [(路径, 情感标签int), ...]，由你在 glob + 文件名解析后传入
        self.file_list = file_list    # 原样保存，供下面两个方法用
    def __len__(self):
        return len(self.file_list)    # 样本总数 = 列表长度
    def __getitem__(self, idx):
        path, label = self.file_list[idx]          # 取第 idx 条 (路径, 标签)
        wav, sr = torchaudio.load(path)            # 读音频 → (波形[1,T], 采样率)
        logmel = wav_to_logmel(wav)                # Step 2 的函数 → [1,64,128]
        return logmel, torch.tensor(label, dtype=torch.int64)  # 标签必须 int64

# 划分：打乱后 80% 训练 / 20% 测试，固定 seed=42 保证可复现
# 封装成函数：train.py 直接 import 它 → 训练与验收永远用同一份数据划分
def make_loaders(batch_size=32, seed=42):
    files = glob.glob("data/**/*.wav", recursive=True)       # 递归列出全部 wav
    pairs = [(f, int(f.split("-")[2]) - 1) for f in files]  # 编码01→标签0
    random.Random(seed).shuffle(pairs)             # 固定种子 → 每次运行划分一致
    cut = int(len(pairs) * 0.8)                  # 80% 处切一刀
    train_ds = RavdessDataset(pairs[:cut])         # 前 80% 训练
    test_ds = RavdessDataset(pairs[cut:])          # 后 20% 测试——Step 5/7 都要用
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)   # 训练要打乱
    test_loader = DataLoader(test_ds, batch_size=batch_size)      # 测试不用打乱
    return train_loader, test_loader

# 测试代码
if __name__ == "__main__":
    # 验收测试：直接运行本文件才执行；被 train.py import 时自动跳过
    train_loader, test_loader = make_loaders()          # 上面刚封装好的函数
    x, y = next(iter(train_loader))        # 取出一个 batch（32 条）
    print("x.shape:", x.shape)      # 预期 torch.Size([32, 1, 64, 128])，32=batch 里的样本数
    print("y.shape:", y.shape)      # 预期 torch.Size([32])，每条样本一个整数标签
    print("y dtype:", y.dtype)      # 预期 torch.int64（dtype = data type，张量里数字的类型）
    print("train size:", len(train_loader.dataset))  # 预期 1152（=1440×0.8，1440 条语音的 80%）
    print("test size:", len(test_loader.dataset))    # 预期 288（后 20%）