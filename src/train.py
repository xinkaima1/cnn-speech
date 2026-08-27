# ============ train.py 开头的 import 区（缺一个都 NameError）============
import torch
import torch.nn as nn
from model import SpeechCNN          # 你 Step 4 写的类
from dataset import make_loaders      # Step 3 封装好的数据划分 → 两个 loader
import matplotlib
matplotlib.use("Agg")                   # WSL 无图形界面，用 Agg 后端（存文件不弹窗）
import matplotlib.pyplot as plt      # Step 6 画收敛曲线用

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)                            # 预期 cuda（你的 4080）
train_loader, test_loader = make_loaders(batch_size=32)  # 组装数据（与验收同一份划分）
net = SpeechCNN().to(device)                              # 模型搬到 GPU
criterion = nn.CrossEntropyLoss()                         # 式(2)：目标函数
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)    # 式(10)：更新规则
history = {"train_loss": [], "train_acc": [], "test_acc": []}  # Step 6 画图的数据源

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

for epoch in range(40):                                    # epoch: 30~60，看曲线定
    net.train()
    loss_sum, n_batches = 0.0, 0                       # 每个 epoch 重新归零，用于算平均值
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)                  # 数据搬到 GPU（每个 batch 搬一次）
        logits = net(x)            # ① 前向
        loss = criterion(logits, y) # ② 计算损失
        optimizer.zero_grad()      # ③ 清旧梯度（否则梯度会累加！）
        loss.backward()            # ④ 反向传播（执行你推导的全部公式）
        optimizer.step()           # ⑤ 更新参数
        loss_sum += loss.item(); n_batches += 1          # .item() = 从张量里取出普通数字
    avg_loss = loss_sum / n_batches                        # 本 epoch 训练损失
    train_acc = evaluate(net, train_loader)   # 训练集正确率（evaluate 函数在最上面，直接调用）
    test_acc = evaluate(net, test_loader)     # 测试集正确率——Step 7 的最终数字就看它
    history["train_loss"].append(avg_loss)    # 三个数记进 history，Step 6 画图的数据源
    history["train_acc"].append(train_acc)    # append = 往列表末尾追加一个数
    history["test_acc"].append(test_acc)
    print(f"epoch {epoch}  train_loss={avg_loss:.4f}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")  # 每个 epoch 一行仪表盘