import torch                    # torch.randn 要用——import torch.nn as nn 不会带出 torch 这个名字！
import torch.nn as nn

class SpeechCNN(nn.Module):
    def __init__(self, n_classes=8):
        super().__init__()
        # 三个卷积块：Conv2d(1→16→32→64, 3×3, padding=1) + ReLU + MaxPool2d(2)
        self.block1 = nn.Sequential(              # 输入 [N, 1, 64, 128]
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # → [N, 16, 64, 128]
            nn.ReLU(),
            nn.MaxPool2d(2),                     # → [N, 16, 32, 64] 高宽减半
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # → [N, 32, 32, 64]
            nn.ReLU(),
            nn.MaxPool2d(2),                     # → [N, 32, 16, 32]
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → [N, 64, 16, 32]
            nn.ReLU(),
            nn.MaxPool2d(2),                     # → [N, 64, 8, 16] 三连减半后 64×8×16
        )
        # 分类头：Flatten → Linear(64*8*16 → 128) + ReLU → Linear(128 → n_classes)
        self.classifier = nn.Sequential(
            nn.Flatten(),                         # → [N, 8192]（= 64×8×16）
            nn.Linear(64 * 8 * 16, 128),          # → [N, 128]（参数量大头：104.9 万）
            nn.ReLU(),
            nn.Linear(128, n_classes),            # → [N, 8] logits
        )
    def forward(self, x):
        # x: [N, 1, 64, 128] → 依次过三个卷积块 → 分类头 → 返回 logits [N, 8]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)                 # 结尾不加 softmax！CrossEntropyLoss 自带