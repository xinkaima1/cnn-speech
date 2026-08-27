# train.py 代码逐行详解

---

## 零、这个文件在做什么（先看全景）

`train.py` 是整个项目的**总指挥**：把前面各步骤的成果组装起来，完成"训练 → 评估 → 记录"的完整闭环。

```
dataset.py 的 make_loaders()  ──提供数据──▶
model.py 的 SpeechCNN         ──提供模型──▶   train.py（本文件）
                                          │
                                          ├── 训练 40 轮（epoch）
                                          ├── 每轮算训练损失 + 训练/测试正确率
                                          └── 记进 history 字典（Step 6 画图用）
```

五个核心角色（深度学习万能五件套）：

| 角色 | 代码 | 类比 |
|---|---|---|
| 数据 | `train_loader / test_loader` | 教材 |
| 模型 | `net = SpeechCNN()` | 学生 |
| 损失函数 | `criterion = nn.CrossEntropyLoss()` | 打分老师 |
| 优化器 | `optimizer = Adam(...)` | 学习方法 |
| 循环 | `for epoch ... for batch ...` | 课程表 |

---

## 一、导入区（第 1-8 行）

```python
# ============ train.py 开头的 import 区（缺一个都 NameError）============
import torch
import torch.nn as nn
from model import SpeechCNN          # 你 Step 4 写的类
from dataset import make_loaders      # Step 3 封装好的数据划分 → 两个 loader
import matplotlib
matplotlib.use("Agg")                   # WSL 无图形界面，用 Agg 后端（存文件不弹窗）
import matplotlib.pyplot as plt      # Step 6 画收敛曲线用
```

#### 第 2 行：`import torch`
- PyTorch 核心：张量运算、设备管理（`torch.device`）、优化器都在这

#### 第 3 行：`import torch.nn as nn`
- 神经网络组件库：损失函数（`nn.CrossEntropyLoss`）在这里
- `as nn` 是社区约定俗成的别名，全世界的 PyTorch 代码都这么写

#### 第 4 行：`from model import SpeechCNN`
- **项目内部导入**！`model.py` 是你自己写的文件（Step 4 的 CNN 结构），和 train.py 同在 `src/` 下
- **导入原理**：Python 运行脚本时会把**脚本所在目录**（`src/`）自动加入搜索路径，所以 `from model import ...` 天然能找到同目录的 model.py
- ⚠️ 注意：这跟"你当前在哪个目录"无关——**从项目根目录跑 `python src/train.py` 照样能导入成功**
- 如果报 `ModuleNotFoundError: No module named 'model'`，说明 model.py 被移动/改名了

#### 第 5 行：`from dataset import make_loaders`
- 同样是项目内部导入，拿 Step 3 封装的数据划分函数
- **关键设计**：训练和验收（dataset.py 的 `__main__` 测试）用的是**同一个函数 → 同一份数据划分**，结果可复现

#### 第 6-8 行：matplotlib 三连
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```
- **`matplotlib.use("Agg")` 的坑**：必须在 `import pyplot` **之前**调用！
  - 因为 pyplot 导入时会锁定后端，先锁了再改就晚了
- `Agg` = 纯画图不出窗口的后端，WSL 没有图形界面，弹窗会报错；Agg 只存文件
- ⚠️ **当前版本画图代码还没写**（Step 6 待补），所以 `plt` 暂时**导入了但没用**——IDE 会提示 "plt imported but unused"，这是正常的，不是 bug。补上画图代码后提示自动消失

---

## 二、全局准备区（第 10-16 行）

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)                            # 预期 cuda（你的 4080）
train_loader, test_loader = make_loaders(batch_size=32)  # 组装数据（与验收同一份划分）
net = SpeechCNN().to(device)                              # 模型搬到 GPU
criterion = nn.CrossEntropyLoss()                         # 式(2)：目标函数
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)    # 式(10)：更新规则
history = {"train_loss": [], "train_acc": [], "test_acc": []}  # Step 6 画图的数据源
```

#### 第 10 行：设备选择（三元表达式）
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
- `torch.cuda.is_available()`：检测有没有可用的 NVIDIA 显卡
- 有 → `"cuda"`（GPU）；没有 → `"cpu"`
- 这是**一行兼容写法**：同一份代码在有卡/没卡的机器上都能跑
- 预期打印 `cuda`（你的 4080）

#### 第 12 行：组装数据
- 调用 dataset.py 的 `make_loaders(batch_size=32)`
- 返回两个 DataLoader：训练用（打乱）、测试用（不打乱）
- 注意这里**没传 seed** → 用默认值 42 → 和 dataset.py 验收测试的划分**完全一致**

#### 第 13 行：`net = SpeechCNN().to(device)`
- `SpeechCNN()`：实例化模型（此时权重是随机初始化的"白痴"状态）
- `.to(device)`：把模型的全部参数搬到 GPU 显存
- **易错点**：模型和数据必须在**同一个设备**上，模型在 GPU、数据在 CPU 会直接报错

#### 第 14 行：损失函数
- `nn.CrossEntropyLoss()` = 交叉熵损失，分类任务标配（对应论文式(2)）
- 内部自动做 softmax + 取负对数，所以模型最后一层**不需要**自己加 softmax
- 输入：模型输出 `logits [32, 8]` + 真标签 `y [32]` → 输出：一个标量 loss

#### 第 15 行：优化器
```python
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
```
- `net.parameters()`：生成模型所有可学习权重（PyTorch 自动登记）
- `Adam`：自适应学习率优化器，几乎不用调参就能用，深度学习默认首选（对应式(10)）
- `lr=1e-3`：学习率 0.001，Adam 的经典默认值
- 优化器的任务：根据 `loss.backward()` 算出的梯度，**真正修改**权重

#### 第 16 行：记录器
- `history` 是一个**字典**，三个 key 各对应一个**空列表**
- 每轮训练往列表里 `append` 一个数，40 轮后每个列表有 40 个数 → 正好是画曲线的数据

---

## 三、评估函数 `evaluate`（第 18-28 行）

```python
def evaluate(net, loader):
    # 遍历一遍 loader 算正确率：eval 模式 + no_grad 不算梯度（省显存提速）
    net.eval(); correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = net(x).argmax(dim=1)      # 每行最大值的下标 = 预测类别
            correct += (pred == y).sum().item()
            total += len(y)
    net.train()                            # 用完切回训练模式
    return correct / total
```

#### 第 20 行：`net.eval()`
- 切到**评估模式**：关闭训练专用的行为（如 Dropout 随机丢弃）
- 不切换的话，Dropout 会继续随机丢神经元 → 评估结果不稳定
- 两个常见模式：`net.eval()`（评估）/ `net.train()`（训练），好习惯是成对出现

#### 第 21 行：`with torch.no_grad():`
- 评估**不需要算梯度**（又不更新权重，要梯度干嘛）
- 关掉梯度追踪 → **省一大截显存和计算**，速度明显变快
- `with` 块内的代码享受这个待遇，出了块自动恢复

#### 第 23 行：数据搬 GPU
- 和训练循环里一样：每个 batch 搬一次

#### 第 24 行：`pred = net(x).argmax(dim=1)`（关键行）
- `net(x)`：前向传播，输出 `logits [32, 8]`（32 条样本 × 8 类的得分）
- `argmax(dim=1)`：**每行**最大值的**下标** → 也就是"模型认为是第几类"
- 区别记忆：`max` 返回值，`argmax` 返回位置；分类要的是**位置**（类别编号）
- `dim=1` 指按行找（dim=0 是按列找，这里不对）

#### 第 25-26 行：数对了几个
- `pred == y`：逐元素比较，得到 `[True, False, True, ...]` 张量
- `.sum()`：True 计 1，加起来 = 答对的数量
- `.item()`：从张量里取出普通 Python 数字（只有单元素张量能用）
- `total` 累加样本总数，最后 `correct / total` = 正确率

#### 第 27 行：`net.train()` 切回训练模式
- **易漏点**！评估完忘记切回来，下一轮训练的 Dropout 就废了
- 该函数被 train_loader 和 test_loader 各调用一次，靠这行保证模式正确

---

## 四、训练主循环（第 30-47 行）+ 存盘（第 48-52 行）

### 4.1 外层：epoch 循环

```python
for epoch in range(40):                                    # epoch: 30~60，看曲线定
    net.train()
    loss_sum, n_batches = 0.0, 0                       # 每个 epoch 重新归零，用于算平均值
```

- **epoch = 全部数据完整过一遍**。40 epoch = 全部 1152 条样本看 40 遍
- 为什么多看几遍：一遍学不会，反复看逐步收敛
- `net.train()`：明确切训练模式（第一轮其实多余，但和 `evaluate` 里的 eval 配对，安全）
- `loss_sum, n_batches`：累加器和计数器，**每个 epoch 归零重来**（算本 epoch 平均损失用）
- 40 不是神圣数字：太少学不会，太多过拟合（train_acc 涨、test_acc 反跌）。**怎么定看曲线**（Step 6 的意义）

### 4.2 内层：batch 循环（训练五步曲）

```python
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)                  # 数据搬到 GPU（每个 batch 搬一次）
        logits = net(x)            # ① 前向
        loss = criterion(logits, y) # ② 计算损失
        optimizer.zero_grad()      # ③ 清旧梯度（否则梯度会累加！）
        loss.backward()            # ④ 反向传播（执行你推导的全部公式）
        optimizer.step()           # ⑤ 更新参数
        loss_sum += loss.item(); n_batches += 1          # .item() = 从张量里取出普通数字
```

- `for x, y in train_loader`：DataLoader 每次吐出一个 batch（32 条样本 + 32 个标签）
- 1152 条 ÷ 32 = 36 个 batch → 内层循环每 epoch 转 36 圈

#### ① 前向：`logits = net(x)`
- 数据流过 CNN，输出 `[32, 8]` 的得分矩阵
- "logits" = 未经 softmax 的原始分数，CrossEntropyLoss 的标准输入

#### ② 计算损失：`loss = criterion(logits, y)`
- 拿得分和正确答案对比，算出一个数字（越低越好）
- 这一个标量就是整个 batch 的"错误程度"

#### ③ 清旧梯度：`optimizer.zero_grad()`（新手最常忘！）
- PyTorch 的梯度默认**累加**不清零（为了支持梯度累积等高级玩法）
- 不清零 → 上一 batch 的梯度混进来 → 更新方向被污染 → 训练发散
- 位置要求：必须在 `backward()` **之前**清（顺序 ③④ 是标准写法，①②③ 里的 ③ 放哪都行，只要在 ④ 前）

#### ④ 反向传播：`loss.backward()`
- 自动求导引擎从 loss 出发，**反向**算出每个权重的梯度（∂loss/∂w）
- 你论文里推导的公式，这一行全包了——这就是用框架的意义

#### ⑤ 更新：`optimizer.step()`
- Adam 拿着梯度真正**修改权重**（w ← w - lr·方向）
- 注意 `backward` 只算不改，`step` 才改——两步是分开的

#### 累计统计
- `loss.item()`：把损失从张量变普通数字才能累加
- 一个 epoch 结束后 `avg_loss = loss_sum / 36`（36 个 batch 的平均损失）

### 4.3 每 epoch 的评估与记录

```python
    avg_loss = loss_sum / n_batches                        # 本 epoch 训练损失
    train_acc = evaluate(net, train_loader)   # 训练集正确率（evaluate 函数在最上面，直接调用）
    test_acc = evaluate(net, test_loader)     # 测试集正确率——Step 7 的最终数字就看它
    history["train_loss"].append(avg_loss)    # 三个数记进 history，Step 6 画图的数据源
    history["train_acc"].append(train_acc)    # append = 往列表末尾追加一个数
    history["test_acc"].append(test_acc)
    print(f"epoch {epoch}  train_loss={avg_loss:.4f}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")  # 每个 epoch 一行仪表盘
```

- 评估两遍：训练集 + 测试集。**看两者的差**是判断过拟合的核心手段：
  - `train_acc` 高、`test_acc` 也高且接近 → 学得好
  - `train_acc` 很高、`test_acc` 明显低 → **过拟合**（死记硬背了）
- `append`：往列表末尾追加，history 三个列表各长一格
- `f"..."`：f-string 格式化字符串，`{avg_loss:.4f}` = 保留 4 位小数
- print 那行 = 每 epoch 一行仪表盘，盯着它就知道训练是否在进步

### 4.4 训练结束：存盘（第 48-52 行）

```python
# ============ 存盘：训练成果落袋为安（放画图之前——后面画图代码若有 bug 崩了，模型也不丢）============
torch.save(net.state_dict(), "speech_cnn.pth")
# save = 存盘 / state_dict = 107 万个参数值打包成的字典 / .pth = PyTorch 模型文件习惯后缀
# 相对路径 → 从 ~/cnn-speech 运行时落在项目根目录（与 data/、src/ 同级），约 4 MB
print("saved: speech_cnn.pth")    # 终端确认；跑完可用 ls -lh speech_cnn.pth 核对大小
```

#### 第 49 行：`torch.save(net.state_dict(), "speech_cnn.pth")`（关键行）
- **`net.state_dict()`**：把模型全部参数值（约 107 万个数）打包成一个**字典**——key 是层的名字，value 是该层的权重张量
  - 只存**参数值**，不存模型结构——所以加载时需要先建好同样的 `SpeechCNN()` 再灌进去
- **`torch.save`**：把字典序列化写入磁盘文件
- **`.pth`**：PyTorch 模型文件的惯用后缀（无强制要求，但全社区都这么写）
- **相对路径的坑**：`"speech_cnn.pth"` 落在**当前工作目录**——从根目录运行就落在项目根目录；从别处运行会存到别处（和 `data/` 的相对路径问题同理）

#### 第 52 行：print 确认
- 终端打一行确认，防止"以为存了其实没存"
- 训练完可核对：`ls -lh speech_cnn.pth`，预期约 4 MB（107 万参数 × 4 字节）

#### 为什么要"先存盘、后画图"
- 注释里写了设计意图：**存盘放在画图代码之前**——后面 Step 6 画图若报 bug 崩溃，训练了 40 轮的模型已经落袋为安，不用重训
- 顺序思想：**昂贵成果先持久化，易错步骤放后面**

---

## 五、训练五步曲总结（必须背下来）

```
① logits = net(x)           前向：算得分
② loss = criterion(logits,y) 打分：算错误
③ optimizer.zero_grad()      清梯度：擦黑板
④ loss.backward()            反向：算每个权重该怎么改
⑤ optimizer.step()           更新：真的改
```

顺序固定 ①②③④⑤，循环往复。**忘了 ③ 是新手第一大坑**（梯度累加 → 训练发疯）。

---

## 六、数据流全图

```
train_loader (1152 条, 36 个 batch)
    │
    ▼ 每个 batch:
x[32,1,64,128] ──GPU──▶ SpeechCNN ──▶ logits[32,8]
                                          │
                       y[32] ──────▶ CrossEntropyLoss ──▶ loss(标量)
                                                            │
                        zero_grad ◀──────────────────────── │
                                                            ▼
                                                        backward()
                                                            │
                                                            ▼
                                                        step() ──▶ 权重更新
（36 个 batch 转完 = 1 个 epoch，共 40 个 epoch）
    │
    ▼ 每 epoch 结束:
evaluate(train_loader) → train_acc ─┐
evaluate(test_loader)  → test_acc ──┼──▶ history 字典 ──▶ (Step 6 画曲线)
avg_loss             ──────────────┘
    │
    ▼ 40 epoch 全部结束后:
torch.save(net.state_dict(), "speech_cnn.pth")   ──▶ 模型存盘（~4 MB，落袋为安）
```

---

## 七、常见问题 Q&A

### Q1：为什么 `zero_grad()` 在 `backward()` 之前？不写会怎样？

PyTorch 的梯度是**累加设计**：每次 `backward()` 算出的梯度会**加**到已有的梯度上，而不是覆盖。

- 正常单步更新：清零 → 算 → 用，梯度永远是"本 batch 的"
- 忘了清零：本 batch 梯度 = 前 35 个 batch 梯度之和 → 更新方向严重跑偏 → loss 不降反升、发散成 NaN

记法：**backward 是"算"，zero_grad 是"擦"，step 是"用"。每轮先擦黑板再写。**

### Q2：`net.eval()` / `net.train()` 到底切换了什么？

切换**层的运行模式**，不碰权重：

| 模式 | Dropout | BatchNorm |
|---|---|---|
| `net.train()` | 随机丢弃神经元（增强泛化） | 用当前 batch 统计量 |
| `net.eval()` | 全部保留（稳定输出） | 用历史累积统计量 |

评估时若忘切 `eval()`，Dropout 继续随机丢 → 同一份数据测两次结果不同，指标不可信。

### Q3：`argmax(dim=1)` 为什么是 dim=1？

logits 形状 `[32, 8]`：
- `dim=0` = 沿"样本"方向（列）→ 算每**类**的最高分样本，不是我们要的
- `dim=1` = 沿"类别"方向（行）→ 算每**样本**得分最高的类别 ✓

一句话：**每一行是一张图的 8 类得分，答案取每行最大位置 → dim=1**。

### Q4：`.item()` 是干嘛的？直接用 `loss` 不行吗？

`loss` 是一个**张量**（哪怕只含一个数）。直接 `loss_sum += loss` 会有两个问题：
- 张量累加会一直挂着计算图引用 → **显存越占越多**（内存泄漏式错误）
- 类型不匹配，累加出的是张量不是数字

`.item()` = 把单元素张量**剥离**成普通 Python float，干干净净。

### Q5：`torch.device("cuda" if ... else "cpu")` 这行语法是什么？

Python 的**三元表达式**（条件表达式）：

```python
A if 条件 else B     # 条件成立取 A，否则取 B
```

等价于：

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
```

一行写完分支，常见于"有没有 GPU 都能跑"的兼容代码。

### Q6：模型和数据为什么都要 `.to(device)`？只搬一个行吗？

不行。**运算要求双方在同一设备**：
- 数据在 CPU、模型在 GPU → 报 `Expected all tensors to be on the same device` 错误
- 本代码在两处搬数据：训练循环每个 batch 搬一次（第 34 行）、evaluate 里也搬（第 23 行）

记法：**模型搬一次（第 13 行），数据每 batch 搬一次**——模型常驻 GPU，数据像流水一样过一遍就走。

### Q7：history 记这些数干什么用？

history 是 Step 6（画收敛曲线）的**唯一数据源**：

- `train_loss` 曲线：看**是否在学**（应持续下降）
- `train_acc` vs `test_acc` 两条曲线放一起：看**是否过拟合**（两线张口就是）
- 决定 `epoch=40` 是否合理、要不要早停，全靠看图说话

所以训练时一个数都不能漏记。

### Q8：存盘存的是什么？以后怎么加载回来用？

**存的是参数，不是模型**。`state_dict()` 是一个字典：

```
{"block1.0.weight": 张量, "block1.0.bias": 张量, ..., "classifier.3.weight": 张量, ...}
```

key 是层的名字，value 是该层学到的权重值——约 107 万个数，序列化成 ~4 MB 文件。

以后加载（如写 predict.py 推理时）是**两步走**：

```python
net = SpeechCNN()                          # ① 先建好同样结构的空模型
net.load_state_dict(torch.load("speech_cnn.pth"))  # ② 把存的参数灌进去
```

因为只存了参数值、没存结构，所以 ① 不可省——文件里不知道"这些数该摆成什么形状的模型"。

**顺带的工程价值**：下次想调 epoch 数或换学习率重训前，可以先加载这份参数接着练，不用从零开始。

---

## 八、运行方式与预期输出

**必须在项目根目录运行**（不是 `src/` 里！）：

```bash
cd ~/cnn-speech
python src/train.py
```

**为什么必须在根目录**：dataset.py 第 84 行的 `glob.glob("data/**/*.wav")` 是**相对当前工作目录**的路径——
- 在根目录跑 → 找到 `data/`，1440 个文件正常加载
- 在 `src/` 里跑 → `src/data/` 不存在 → files 为空 → **训练空转不报错**（更隐蔽的坑）

import 不用担心：Python 会把脚本所在目录 `src/` 自动加进搜索路径，所以从根目录跑 `python src/train.py` 时 `from model import ...` 照样成功。

预期输出（每 epoch 一行，最后存盘确认）：

```
device: cuda
epoch 0  train_loss=2.0134  train_acc=0.2543  test_acc=0.2396
epoch 1  train_loss=1.7221  train_acc=0.3312  test_acc=0.2986
...
epoch 39  train_loss=0.1102  train_acc=0.9626  test_acc=0.5833
saved: speech_cnn.pth
```

**看什么**：
- `train_loss` 应一路下降
- `test_acc` 先升后可能平台/回落（过拟合信号）
- 最终报告用 `test_acc`（Step 7 的数字）
- 最后一行 `saved:` 出现 = 模型已存到项目根目录，用 `ls -lh speech_cnn.pth` 核对（~4 MB）

---

## 九、关键术语表

| 术语 | 英文 | 解释 |
|---|---|---|
| epoch | Epoch | 全部训练数据完整过一遍 |
| batch | Batch | 一次送入模型的一小撮样本（这里 32 条） |
| logits | Logits | 模型输出、未经 softmax 的原始得分 |
| 损失函数 | Loss Function | 量化"模型错多少"的函数，训练的目标是把它变小 |
| 交叉熵 | Cross Entropy | 分类任务标准损失：预测概率偏离真标签越多，惩罚越大 |
| 优化器 | Optimizer | 根据梯度更新权重的算法（Adam 最常用） |
| 学习率 | Learning Rate (lr) | 每次权重更新的步长，太大发散、太小学不动 |
| 梯度 | Gradient | loss 对每个权重的偏导，指出"权重往哪调能降 loss" |
| 反向传播 | Backward | 自动求导引擎，从 loss 反推出所有梯度 |
| 评估模式 | eval mode | 关 Dropout/BatchNorm 训练行为的模式，评估专用 |
| no_grad | No Grad | 关闭梯度追踪，省显存提速，评估专用 |
| 过拟合 | Overfitting | train_acc 高、test_acc 低：模型在死记硬背 |
| f-string | Formatted String | `f"{x:.4f}"` 格式化字符串语法 |
| 三元表达式 | Ternary | `A if cond else B` 一行分支写法 |
| state_dict | State Dict | 模型所有参数打包成的字典（key=层名，value=权重），存盘/加载的标准格式 |
| checkpoint | Checkpoint | 训练存档：模型参数（+可选优化器状态），可恢复训练或推理 |
