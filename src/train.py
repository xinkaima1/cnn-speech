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
# ============ 存盘：训练成果落袋为安（放画图之前——后面画图代码若有 bug 崩了，模型也不丢）============
torch.save(net.state_dict(), "speech_cnn.pth")
# save = 存盘 / state_dict = 107 万个参数值打包成的字典 / .pth = PyTorch 模型文件习惯后缀
# 相对路径 → 从 ~/cnn-speech 运行时落在项目根目录（与 data/、src/ 同级），约 4 MB
print("saved: speech_cnn.pth")    # 终端确认；跑完可用 ls -lh speech_cnn.pth 核对大小
# ============ 以下追加到 train.py 末尾（Step 6：收敛曲线，接在 torch.save 之后）============
plt.figure(figsize=(7, 4))                 # 新开一张画布（7×4 英寸）
plt.plot(history["train_loss"])            # 每 epoch 一个点（x 轴自动为 0,1,2,…）
plt.xlabel("Epoch"); plt.ylabel("Cross-Entropy Loss")
plt.title("Training Loss Convergence")
plt.savefig("loss.png", dpi=150, bbox_inches="tight")

plt.figure(figsize=(7, 4))                 # 再开一张新画布，两张图互不干扰
plt.plot(history["train_acc"], label="train acc")
plt.plot(history["test_acc"], label="test acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.title("Accuracy Convergence")
plt.legend()                               # 图例：把 label 显示出来
plt.savefig("acc.png", dpi=150, bbox_inches="tight")
print("saved: loss.png / acc.png")
# ============ 以下追加到 train.py 末尾（Step 7：终评+混淆矩阵，接在画图代码后面）============
from sklearn.metrics import ConfusionMatrixDisplay   # 混淆矩阵工具（8/26 已装好 sklearn）
net.eval()                                          # 评估模式
y_true, y_pred = [], []                             # 边跑边收，混淆矩阵要用
with torch.no_grad():                               # 不建计算图，省显存提速
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = net(x).argmax(dim=1)                 # 每行最大值下标 = 预测类别
        y_true += y.cpu().tolist()                  # 转普通列表（sklearn 不认 GPU 张量）
        y_pred += pred.cpu().tolist()
acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
print(f"test accuracy = {acc:.4f}")                 # 随机基线 0.125——应为它的数倍
ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
plt.savefig("fig_confusion.png", dpi=150, bbox_inches="tight")
print("saved: fig_confusion.png")