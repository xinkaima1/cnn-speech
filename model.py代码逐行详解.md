# model.py 代码逐行详解（对应 src/model.py，共 35 行）

---

## 零、这个文件在做什么（先看全景）

`model.py` 定义整个项目唯一的**神经网络结构** `SpeechCNN`——把一张 `[64, 128]` 的 log-Mel 频谱图"看"一遍，输出 8 类情感中每一类的得分。

```
dataset.py 的 wav_to_logmel ──提供输入──▶  [N, 1, 64, 128]
                                            │
                                            ▼
                              model.py 的 SpeechCNN（本文件）
                                            │
                                            ▼
                                    logits [N, 8] ──▶ train.py 的 CrossEntropyLoss
```

**一句话理解 CNN**：频谱图本质上是一张"灰度图片"（高 64 = 频带，宽 128 = 时间帧），所以语音情感识别被转化成了一个**图像 8 分类问题**——用图像识别的标配武器（卷积神经网络）来解决。

**设计思想（这个网络在模仿什么）**：三层"卷积 + 池化"逐级提取特征（低级纹理 → 中级模式 → 高级语义），最后用全连接层把这些特征"读"成 8 个情感得分——和图像分类教科书结构（LeNet/AlexNet 的迷你版）同源。

---

## 一、导入部分（第 1-2 行）

```python
import torch                    # torch.randn 要用——import torch.nn as nn 不会带出 torch 这个名字！
import torch.nn as nn
```

#### 第 1 行：`import torch`
- **为什么必须有这行**：第 2 行的 `import torch.nn as nn` **只会引入名字 `nn`**，不会把 `torch` 这个名字带进当前文件
- 本文件用 `torch` 的地方：`torch.randn`（第 109 行注释测试里造假数据用）
- ⚠️ **新手常见坑**：以为 `import torch.nn as nn` 之后就能写 `torch.xxx`——**不行**，会报 `NameError: name 'torch' is not defined`。两个名字是两条独立的导入语句，各管各的

#### 第 2 行：`import torch.nn as nn`
- `torch.nn` = PyTorch 的神经网络组件库：`nn.Module`、`nn.Sequential`、`nn.Conv2d`、`nn.ReLU`、`nn.MaxPool2d`、`nn.Flatten`、`nn.Linear` 全在这里
- `as nn`：全世界通用的约定俗成别名，任何 PyTorch 代码里 `nn.` 都指 `torch.nn.`

---

## 二、类定义骨架（第 4-6 行）

```python
class SpeechCNN(nn.Module):
    def __init__(self, n_classes=8):
        super().__init__()
```

#### 第 4 行：`class SpeechCNN(nn.Module)`
- **继承**：`SpeechCNN(nn.Module)` 表示 `SpeechCNN` 是 PyTorch `nn.Module` 基类的子类
- **为什么要继承**：`nn.Module` 提供了大量免费能力——参数自动登记（`net.parameters()` 能找到所有权重）、`.to(device)` 搬 GPU、`state_dict()` 打包存盘、`net.train()/net.eval()` 模式切换……不继承这些全都得手写
- **命名习惯**：PyTorch 模型通常继承 `nn.Module`（keras/TensorFlow 世界则常用 `Model` 基类，思想相同）

#### 第 5 行：`def __init__(self, n_classes=8)`
- **构造函数**：`SpeechCNN()` 被调用时先执行它
- **参数 `n_classes=8`**：要分几类。RAVDESS 有 8 种情感 → 默认 8。写死数字（8）不如写成参数（n_classes）灵活——想换成 4 分类数据集时，`SpeechCNN(n_classes=4)` 一行搞定，不用改网络代码
- **`self`**：指向"正在被创建的那个实例自己"，Python 类方法的第一个参数固定是它（调用时不用手动传）

#### 第 6 行：`super().__init__()`
- **调用父类 `nn.Module` 的构造函数**，先完成父类的初始化（登记参数的簿记系统等）
- ⚠️ **不写会直接报错**：`AttributeError: cannot assign module before Module.__init__() call`——PyTorch 强制要求先初始化父类，再开始 `self.xxx = ...` 搭积木
- 记法：**盖房子先打地基（父类），再砌墙（自己的层）**

---

## 三、三个卷积块（第 7-22 行）

每个块都是同一个配方：**Conv2d（提特征）→ ReLU（加非线性）→ MaxPool2d（压缩尺寸）**，像流水线上的三道工序。

```python
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
```

### 3.1 `nn.Sequential(...)`（容器，串起一个块）

- **作用**：把多个层按顺序串成一个"块"，调用这个块 = 数据依次流过每一层
- 等价写法对比（`nn.Sequential` 是简写）：

```python
# 写法一：Sequential（本文件用的）
self.block1 = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
)

# 写法二：手动拆开（效果完全相同）
self.conv1   = nn.Conv2d(1, 16, kernel_size=3, padding=1)
self.relu1   = nn.ReLU()
self.pool1   = nn.MaxPool2d(2)
# forward 里就得写三行：x = self.conv1(x); x = self.relu1(x); x = self.pool1(x)
```

- **好处**：三层变一行，`forward` 里每个块只需写一行；块的边界（"一组做一件事"）也更清晰

### 3.2 `nn.Conv2d(进通道, 出通道, kernel_size=3, padding=1)`（特征提取器）

以 block1 的 `nn.Conv2d(1, 16, kernel_size=3, padding=1)` 为例：

| 参数 | 本文件的值 | 含义 |
|---|---|---|
| 第 1 个参数 | 1（block2 是 16，block3 是 32） | **输入通道数**：进来的"图像"有几层。Mel 频谱图是单通道灰度图 → 1 |
| 第 2 个参数 | 16（block2 是 32，block3 是 64） | **输出通道数**：出去有几层 = 这一层学多少种"特征图案" |
| `kernel_size` | 3 | 卷积核是 3×3 的小窗口，在图上滑动扫图 |
| `padding=1` | 1 | 四周各补 1 圈零——**专门为了让输出尺寸不变**（见下） |

- **卷积在做什么**：3×3 的小窗口贴着频谱图滑动，每贴一处就把窗口内 9 个数加权求和（权重就是要学的参数）→ 输出一个数。窗口扫完整张图 → 输出一张"特征响应图"：哪里出现了这个核擅长的图案，哪里就亮
- **通道数为什么 1→16→32→64 翻倍**：越深的层，学到的模式越抽象、需要的"图案种类"越多——这是 CNN 的经典设计（通道翻倍、尺寸减半，VGG 同款思路）
- **参数量怎么算**：`(核高 × 核宽 × 进通道 + 1 偏置) × 出通道`
  - block1：`(3×3×1 + 1) × 16 = 160` 个
  - block2：`(3×3×16 + 1) × 32 = 4640` 个
  - block3：`(3×3×32 + 1) × 64 = 18496` 个

### 3.3 `padding=1` 为什么能让尺寸不变（一行公式）

```
输出尺寸 = (输入尺寸 + 2×padding − kernel_size) / stride + 1
         = (输入尺寸 + 2×1 − 3) / 1 + 1
         = 输入尺寸
```

- **为什么在乎尺寸不变**：卷积只管"提特征"，把尺寸减半的任务全部交给 MaxPool——每层职责单一，形状好推。若 `padding=0`，每过一层尺寸就缩 2（64→62→60…），最后 `64*8*16` 那个数就再也对不上了
- 记法：**kernel 3×3 + padding 1 = 尺寸不变的黄金搭档**（kernel 5×5 配 padding 2 同理）

### 3.4 `nn.ReLU()`（激活函数，加非线性）

- 公式：`ReLU(x) = max(0, x)`——负数归零，正数原样通过
- **为什么必须有**：卷积和全连接都是**线性**运算（乘法+加法），线性叠线性还是线性，再叠一百层也等价于一层——网络就没有"深度"的意义了。夹一个非线性函数，网络才能拟合弯弯曲曲的决策边界
- 类比：只有直尺（线性）画不出圆弧，ReLU 就是那个"折弯器"

### 3.5 `nn.MaxPool2d(2)`（池化，压缩尺寸）

- **2 的含义**：2×2 的窗口内取**最大值**，输出 1 个数 → 高、宽各**减半**
- 三个块的三连减半：`64×128 → 32×64 → 16×32 → 8×16`
- **为什么压缩**：① 后面的全连接层输入从 64×128=8192 直接对应更大的数（若不压缩将是 64×128 通道数连乘，参数爆炸）；② 小的位置平移不敏感——特征稍微挪一点，最大值还是被抓住
- **注意**：池化**没有要学的参数**（只是取 max 的固定操作），所以它不贡献参数量

### 3.6 三个块各自的形状流水账（N = batch 大小）

| 块 | 进 | Conv 后 | MaxPool 后 |
|---|---|---|---|
| block1 | `[N, 1, 64, 128]` | `[N, 16, 64, 128]` | `[N, 16, 32, 64]` |
| block2 | `[N, 16, 32, 64]` | `[N, 32, 32, 64]` | `[N, 32, 16, 32]` |
| block3 | `[N, 32, 16, 32]` | `[N, 64, 16, 32]` | `[N, 64, 8, 16]` |

> 维度含义：`[N, C, H, W]` = `[batch 数, 通道数, 高(频带), 宽(时间帧)]`

---

## 四、分类头 classifier（第 23-29 行）

```python
        # 分类头：Flatten → Linear(64*8*16 → 128) + ReLU → Linear(128 → n_classes)
        self.classifier = nn.Sequential(
            nn.Flatten(),                         # → [N, 8192]（= 64×8×16）
            nn.Linear(64 * 8 * 16, 128),          # → [N, 128]（参数量大头：104.9 万）
            nn.ReLU(),
            nn.Linear(128, n_classes),            # → [N, 8] logits
        )
```

#### `nn.Flatten()`（翻译官：卷积世界 → 全连接世界）
- 把 `[N, 64, 8, 16]` 摊平成 `[N, 8192]`（64×8×16 = 8192，一行 8192 个数）
- **为什么必须摊平**：`nn.Linear` 只吃一维向量，不认三维立方体——Flatten 就是把三维特征"拉直"成一条线
- **形状对账**：这里 `64*8*16` 必须和 block3 的输出严格对上——**改任何一个上游参数（比如 n_mels 或池化层数），这个 8192 都要跟着手改**，这也是 CNN 最常见的崩法（`mat1 and mat2 shapes cannot be multiplied` 报错就是它错了）

#### `nn.Linear(64*8*16, 128)`（全连接层 1，参数量大头）
- 8192 个输入 → 128 个输出，每个输出 = 8192 个输入的加权求和
- 参数量 = `8192×128 + 128(偏置) = 1,048,704` ≈ **105 万**，占全模型 107 万参数的 **98%**——深度不深的小模型里，全连接层扛参数大头是常态
- **为什么不直接 8192 → 8**：先压到 128 再分 8 类，中间多一层非线性（ReLU），分类边界更灵活

#### `nn.ReLU()`
- 同上，给两个 Linear 之间加非线性，否则两层 Linear 叠起来等价于一层

#### `nn.Linear(128, n_classes)`（输出层）
- 128 → 8，输出 `[N, 8]` 的 **logits**：每条样本对 8 种情感的原始得分（可正可负，未归一化）
- **注意**：`n_classes` 用的是 `__init__` 的参数——把参数化贯彻到底

---

## 五、前向传播 `forward`（第 30-35 行）

```python
    def forward(self, x):
        # x: [N, 1, 64, 128] → 依次过三个卷积块 → 分类头 → 返回 logits [N, 8]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)                 # 结尾不加 softmax！CrossEntropyLoss 自带
```

#### 第 30 行：`def forward(self, x)`
- **定义"数据怎么流"**：`__init__` 搭积木（买零件），`forward` 规定积木的拼装顺序（怎么用）
- **不需要手动调**：写 `net(x)` 时，PyTorch 自动调用 `forward`——`nn.Module.__call__` 里做了转发（还顺带执行钩子等内部机制），这是"魔法方法"（双下划线方法）的典型代表

#### 第 32-34 行：依次过三个块
- 每行就是查一次 3.6 节的形状流水账：`[N,1,64,128] → [N,16,32,64] → [N,32,16,32] → [N,64,8,16]`
- `x = ...` 不断覆盖：数据像流水一样过站，每站只保留处理结果

#### 第 35 行：`return self.classifier(x)`（⚠️ 关键设计：不加 softmax）
- 返回的是**原始 logits** `[N, 8]`，**故意不做 softmax 归一化**
- **为什么**：`train.py` 用的 `nn.CrossEntropyLoss` **内部自带** `log_softmax`——你在模型里先做一遍 softmax，损失函数里再做一遍，数值会被压得极端化（梯度消失），训练效果反而变差
- 记法：**模型吐 logits，损失函数管归一**——各司其职，谁也不抢谁的活
- 推理时若真想要概率，自己补一行：`probs = torch.softmax(logits, dim=1)`

---

## 六、参数量总账（一张表算清）

| 层 | 计算式 | 参数量 |
|---|---|---|
| block1 Conv2d | (3×3×1+1)×16 | 160 |
| block2 Conv2d | (3×3×16+1)×32 | 4,640 |
| block3 Conv2d | (3×3×32+1)×64 | 18,496 |
| Linear 1 | (8192+1)×128 | 1,048,704 |
| Linear 2 | (128+1)×8 | 1,032 |
| **合计** | | **1,072,032 ≈ 107 万** |

- 验证方法（dataset.py 第 104-110 行注释测试里就有）：

```python
net = SpeechCNN()
print("total params:", sum(p.numel() for p in net.parameters()))  # 1072032
```

- 存盘大小 ≈ 107 万 × 4 字节 ≈ **4.1 MB**（float32 每个参数 4 字节）——train.py 存的 `speech_cnn.pth` 就是这个量级

---

## 七、数据流全图

```
输入频谱图 [N, 1, 64, 128]        （1 通道 = 单张灰度图）
    │
    ▼ block1: Conv(1→16)+ReLU+MaxPool
[N, 16, 32, 64]                  （16 种局部纹理特征，尺寸减半）
    │
    ▼ block2: Conv(16→32)+ReLU+MaxPool
[N, 32, 16, 32]                  （32 种中级模式组合）
    │
    ▼ block3: Conv(32→64)+ReLU+MaxPool
[N, 64, 8, 16]                   （64 种高级语义特征，尺寸再减半）
    │
    ▼ Flatten
[N, 8192]                        （特征拉直成一条向量）
    │
    ▼ Linear(8192→128)+ReLU
[N, 128]                         （特征综合压缩）
    │
    ▼ Linear(128→8)
logits [N, 8]                    （8 类情感得分，未 softmax）
    │
    └──▶ 交给 train.py 的 CrossEntropyLoss 算损失
```

---

## 八、常见问题 Q&A

### Q1：`__init__` 和 `forward` 有什么区别？各干什么？

| | `__init__` | `forward` |
|---|---|---|
| 何时执行 | `SpeechCNN()` 实例化时，**一次** | 每次 `net(x)` 调用时，**每 batch 一次** |
| 干什么 | 声明有哪些层（搭积木/买零件） | 规定数据怎么流过这些层（拼装顺序） |
| 类比 | 厨房备料 | 炒菜的步骤 |

把层声明放 `__init__`（而不是 `forward` 里）的原因：**层只创建一次，权重才能跨 batch 连续保存和更新**。若写在 `forward` 里，每个 batch 都新建层，学到的参数直接作废。

### Q2：为什么是三个块，不是两个或五个？

经验性的"够用就好"：
- **太少（1-2 块）**：感受野太小，只能看到很局部的纹理，分不出需要全局信息的情感
- **太多（5+ 块）**：每块减半一次尺寸，`128` 宽几个块就减没了（128→64→32→16→8→4→2），再深输入就成一条线了；且参数增多、小数据集（1152 条）更容易过拟合
- 三块后 `8×16` 的特征图仍保留一定的频率×时间结构，够用了

### Q3：`kernel_size=3` 为什么不用更大的核（5×5、7×7）？

- **两层 3×3 的感受野 = 一层 5×5**（5×5 的视野可以由两个 3×3 叠出来），但参数更少（2×9×C < 25×C）、还多了一次非线性
- 这是 VGG 论文确立的现代共识：**深而小的核 优于 浅而大的核**

### Q4：`net(x)` 时 Python 怎么知道要跑 `forward`？

魔法方法 `__call__` 机制：`nn.Module` 定义了 `__call__`，内部会调用 `self.forward(x)`。所以：

```python
net(x)        # 表面
net.forward(x)  # 实际等价（但不建议显式这么写——会绕过 nn.Module 的钩子机制）
```

`__call__` 的转发还顺带执行了 PyTorch 的内部簿记（hooks 等），所以**永远写 `net(x)`，不写 `net.forward(x)`**。

### Q5：怎么单独测试这个模型（不训练）？

文件外的最小验证（或用 dataset.py 第 104-110 行的注释测试）：

```python
from model import SpeechCNN
import torch

net = SpeechCNN()
out = net(torch.randn(2, 1, 64, 128))   # 造 2 条假频谱图
print(out.shape)    # torch.Size([2, 8])——形状对了，网络就是通的
```

`torch.randn(2, 1, 64, 128)` = 造 2 条形状正确的随机噪声——**只验形状、不管数值**，是检查网络结构的标准手法。

### Q6：为什么我改了 `n_mels`（或池化层数）之后模型报错？

报错形如 `mat1 and mat2 shapes cannot be multiplied`，几乎一定是 `nn.Linear(64 * 8 * 16, 128)` 的 **8192 和上游对不上了**：

- `64` 来自 `n_mels`（频带数）
- `8`、`16` 来自 `64×128` 输入经三次减半（64→32→16→8、128→64→32→16）

链条上任何一环变了，`64*8*16` 都要手改。这是全连接层的"死板税"，也是后来全卷积网络（FCN）流行起来的原因之一。

### Q7：logits 是什么？和概率什么关系？

- **logits** = 未经归一化的原始得分，范围 (−∞, +∞)，每类一个数
- **概率** = softmax(logits)：把 8 个得分压成 8 个非负、和为 1 的概率

```
logits:      [2.1, -0.5, 0.3, ...]     （本模型输出）
                │ softmax
                ▼
概率:        [0.62, 0.05, 0.11, ...]   （和 = 1，模型只在自己推理时才需要）
```

本模型返回 logits 而不是概率——因为 `CrossEntropyLoss` 内部自带 softmax（见五、第 35 行的说明）。

### Q8：`MaxPool2d` 和 `Conv2d` 都在"压缩信息"，有什么本质区别？

| | Conv2d | MaxPool2d |
|---|---|---|
| 有没有参数 | **有**（核的权重是要学的） | **无**（固定取 max 的规则） |
| 在网络里的角色 | 提特征 | 压尺寸 + 抗小平移 |
| 通道数 | 可变（1→16→32→64） | 不变 |

一句话：卷积是"学出来的滤波器"，池化是"写死的下采样"。

---

## 九、关键术语表

| 术语 | 英文 | 解释 |
|---|---|---|
| 卷积层 | Conv2d | 3×3 小窗口扫图提特征，权重是要学的参数 |
| 通道 | Channel | 图像的"层数"：灰度图 1 层；通道数 = 该层学几种特征图案 |
| 卷积核 | Kernel / Filter | 那个 3×3 的小窗口本身，权重就是模型的"眼睛" |
| 填充 | Padding | 四周补零，kernel 3 + padding 1 = 输出尺寸不变 |
| 步长 | Stride | 窗口每次滑动几格，默认 1 |
| 激活函数 | Activation (ReLU) | max(0,x)，加非线性，没有它网络叠再深也等价一层 |
| 池化 | MaxPool2d | 窗口内取最大值，高宽减半，无参数 |
| 感受野 | Receptive Field | 一个输出像素"看得见"的原图区域大小 |
| 摊平 | Flatten | [N,C,H,W] → [N, C·H·W]，卷积世界进全连接世界的门票 |
| 全连接层 | Linear | 每个输出 = 所有输入的加权求和，参数量大头 |
| logits | Logits | 未经 softmax 的原始得分，CrossEntropyLoss 的标准输入 |
| nn.Module | Module | PyTorch 所有网络的基类，提供参数登记/搬运/存盘等能力 |
| nn.Sequential | Sequential | 按顺序串层的容器，调用块 = 依次过每层 |
| 继承 | Inheritance | 子类自动获得父类能力（SpeechCNN 继承 nn.Module） |
| super() | super | 调用父类方法，`super().__init__()` 先初始化父类 |
| state_dict | State Dict | 模型参数打包成的字典，train.py 存的 .pth 就是它 |
| 前向传播 | Forward Pass | 数据从输入流到输出的过程，`forward()` 定义路线图 |
