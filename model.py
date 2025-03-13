import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import scipy.io
import sys

class ImprovedSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(ImprovedSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 线性变换矩阵
        self.U_qkv = nn.Linear(embed_dim, 3 * embed_dim)  # 用于生成Q, K, V
        self.U_sum = nn.Linear(embed_dim, embed_dim)      # 用于求和操作
        self.U_copy = nn.Linear(embed_dim, embed_dim)     # 用于复制操作
        self.U_proj = nn.Linear(embed_dim, embed_dim)     # 最终的投影矩阵

        # 动态缩放矩阵 D
        self.D = nn.Parameter(torch.ones(num_heads, self.head_dim))

    def forward(self, x):
        B, N, C = x.shape
        # 1. 线性变换生成 Q, K, V
        qkv = self.U_qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # 2. 对 K 和 V 进行 L2 归一化
        k = k / torch.norm(k, dim=-1, keepdim=True)
        v = v / torch.norm(v, dim=-1, keepdim=True)
        # 3. 计算归一化后的 K 和 V 的乘积
        kv_product = k * v
        # 4. 应用线性变换 U_copy 到 K*V 的结果上
        kv_product = self.U_copy(kv_product)
        # 5. 应用线性变换 U_sum 到 K*V 的结果上
        kv_product = self.U_sum(kv_product)
        # 6. 动态缩放 Q
        q = q * self.D.view(1, self.num_heads, 1, self.head_dim)
        # 7. 计算 D ⊙ Q ⊙ (U_copy(K ⊙ V) U_sum)
        qkv_product = q * kv_product
        # 8. 应用线性变换 U_proj 到最终结果上
        out = self.U_proj(qkv_product.permute(0, 2, 1, 3).reshape(B, N, C))

        return out


class ImprovedTransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        super(ImprovedTransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = ImprovedSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

    def forward(self, x):
        x = self.norm1(x)
        x = x + self.attn(x)
        x = self.norm2(x)
        x = x + self.mlp(x)
        return x
    
class ConvolutionalLayers(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalLayers, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.flatten(2)  # Reshape to (B, C, H*W)
        x = x.permute(0, 2, 1)  # Reshape to (B, H*W, C)
        return x
    
class ClassificationLayer(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(ClassificationLayer, self).__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x
    
class TPRO_NET(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads, num_classes):
        super(TPRO_NET, self).__init__()
        self.conv_layers = ConvolutionalLayers(in_channels, embed_dim)
        self.transformer_layers = nn.ModuleList([
            ImprovedTransformerEncoder(embed_dim, num_heads) for _ in range(15)
        ])
        # 为每个指标定义单独的分类层
        self.classifier_valence = ClassificationLayer(embed_dim, num_classes)
        self.classifier_arousal = ClassificationLayer(embed_dim, num_classes)
        self.classifier_dominance = ClassificationLayer(embed_dim, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)  # (B, H*W, C)
        for layer in self.transformer_layers:
            x = layer(x)
        # 为每个指标获取输出
        output_valence = self.classifier_valence(x)
        output_arousal = self.classifier_arousal(x)
        output_dominance = self.classifier_dominance(x)
        return output_valence, output_arousal, output_dominance
    
# 加载 DREAMER 数据集
data = scipy.io.loadmat('/n04dat/jyzhou/emotion/data/DREAMER.mat')
eeg_data = data['DREAMER'][0, 0]  # 假设数据存储在 'DREAMER' 变量中
sampling_rate = 128  # DREAMER 数据集的采样率
all_sec = np.zeros((23,18))
for subject in range(23):
        for trial in range(18):
            sti = eeg_data["Data"][0, subject]["EEG"][0, 0]["stimuli"][0,0][trial,0]
            all_sec[subject][trial] = sti.shape[0]/sampling_rate
    
# 加载数据
data_path = "/n04dat/jyzhou/emotion/data/processed_DREAMER_features.npy"
data = np.load(data_path)

# 假设数据形状为 (23, 18, 时间步数, 8, 9, 9)
# 每个受试者和视频的时间步数可能不同

# 加载标签
label_path = "/n04dat/jyzhou/emotion/data/label.txt"
labels = []
with open(label_path, "r") as file:
    for line in file:
        parts = line.strip().split()
        valence = int(parts[2])
        arousal = int(parts[3])
        dominance = int(parts[4])
        labels.append([valence, arousal, dominance])

# 将标签转换为 numpy 数组
labels = np.array(labels).reshape(23, 18, 3)

# 将数据和标签组织成列表，每个元素对应一个受试者和一个视频
subjects = []
sec = 0
for subject in range(23):
    trials = []
    for trial in range(18):
        # 获取当前受试者和视频的数据
        subject_data = data[sec:sec + all_sec[subject][trial]]
        print(subject_data.shape)
        # 获取当前受试者和视频的标签
        subject_label = labels[subject, trial]
        trials.append((subject_data, subject_label))
        sec = sec + all_sec[subject][trial]
    subjects.append(trials)

# 将数据和标签转换为 PyTorch 张量
dataset = []
for subject in subjects:
    for trial in subject:
        data_tensor = torch.tensor(trial[0], dtype=torch.float32)
        label_tensor = torch.tensor(trial[1], dtype=torch.long)
        dataset.append((data_tensor, label_tensor))

# 确保数据和标签的对应关系正确
# 数据的形状为 (23*18*时间步数, 8, 9, 9)
# 标签的形状为 (23*18, 3)
# 每个 trial 的数据对应一个标签，因此需要将标签扩展到与数据的时间步数相同的维度

# 计算每个 trial 的数据数量
num_subjects = 23
num_trials = 18
num_time_steps = data.shape[0] // (num_subjects * num_trials)

# 将标签扩展到与数据的时间步数相同的维度
# 标签的形状变为 (23*18*时间步数, 3)
labels_expanded = labels.repeat(num_time_steps, 1)

# 现在，data[i] 对应 labels_expanded[i]
#######################################
# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义模型
in_channels = 8  # 输入通道数
embed_dim = 32  # Transformer 嵌入维度
num_heads = 8  # 多头注意力头数
num_classes = 5  # DREAMER 数据集的分类数

model = TPRO_NET(in_channels, embed_dim, num_heads, num_classes)
model = model.to(device)  # 将模型移动到设备上
################################
# 定义 k-fold 交叉验证
k = 5
splits = KFold(n_splits=k, shuffle=True, random_state=42)

# 存储每个 fold 的评估结果
valence_accuracies = []
arousal_accuracies = []
dominance_accuracies = []
valence_precisions = []
arousal_precisions = []
dominance_precisions = []
valence_recalls = []
arousal_recalls = []
dominance_recalls = []
valence_specificities = []
arousal_specificities = []
dominance_specificities = []
valence_f1_scores = []
arousal_f1_scores = []
dominance_f1_scores = []

# 执行 k-fold 交叉验证
for fold, (train_indices, val_indices) in enumerate(splits.split(dataset)):
    print(f"Fold {fold + 1}/{k}")
    
    # 划分训练集和验证集
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    
    # 定义数据加载器
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False)
    
    # 初始化模型、损失函数和优化器
    model = TPRO_NET(in_channels, embed_dim, num_heads, num_classes)
    model = model.to(device)  # 将模型移动到设备上
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # 训练循环
    num_epochs = 50  # 根据 DREAMER 数据集设置
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for data_tensor, label_tensor in train_loader:
            inputs = data_tensor.to(device)  # 将数据移动到设备上
            targets_valence = label_tensor[:, 0].to(device)
            targets_arousal = label_tensor[:, 1].to(device)
            targets_dominance = label_tensor[:, 2].to(device)
            
            # 前向传播
            outputs_valence, outputs_arousal, outputs_dominance = model(inputs)
            
            # 计算损失
            loss_valence = criterion(outputs_valence, targets_valence)
            loss_arousal = criterion(outputs_arousal, targets_arousal)
            loss_dominance = criterion(outputs_dominance, targets_dominance)
            loss = loss_valence + loss_arousal + loss_dominance
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")
################################    
# 重定向输出到文件
sys.stdout = open('/path/a.txt', 'w')

# 评估模型
model.eval()
correct_valence = 0
correct_arousal = 0
correct_dominance = 0
total_samples = 0

# 初始化指标统计变量
tp_valence = np.zeros(5)
fp_valence = np.zeros(5)
tn_valence = np.zeros(5)
fn_valence = np.zeros(5)

tp_arousal = np.zeros(5)
fp_arousal = np.zeros(5)
tn_arousal = np.zeros(5)
fn_arousal = np.zeros(5)

tp_dominance = np.zeros(5)
fp_dominance = np.zeros(5)
tn_dominance = np.zeros(5)
fn_dominance = np.zeros(5)

with torch.no_grad():
    for data_tensor, label_tensor in val_loader:
        inputs = data_tensor.to(device)  # 将数据移动到设备上
        targets_valence = label_tensor[:, 0].to(device)
        targets_arousal = label_tensor[:, 1].to(device)
        targets_dominance = label_tensor[:, 2].to(device)
        
        # 前向传播
        outputs_valence, outputs_arousal, outputs_dominance = model(inputs)
        
        # 统计正确预测数
        _, predicted_valence = torch.max(outputs_valence, 1)
        _, predicted_arousal = torch.max(outputs_arousal, 1)
        _, predicted_dominance = torch.max(outputs_dominance, 1)
        
        correct_valence += (predicted_valence == targets_valence).sum().item()
        correct_arousal += (predicted_arousal == targets_arousal).sum().item()
        correct_dominance += (predicted_dominance == targets_dominance).sum().item()
        total_samples += targets_valence.size(0)
        
        # 更新指标统计
        for c in range(5):
            # Valence
            tp_valence[c] += ((predicted_valence == c) & (targets_valence == c)).sum().item()
            fp_valence[c] += ((predicted_valence == c) & (targets_valence != c)).sum().item()
            tn_valence[c] += ((predicted_valence != c) & (targets_valence != c)).sum().item()
            fn_valence[c] += ((predicted_valence != c) & (targets_valence == c)).sum().item()
            
            # Arousal
            tp_arousal[c] += ((predicted_arousal == c) & (targets_arousal == c)).sum().item()
            fp_arousal[c] += ((predicted_arousal == c) & (targets_arousal != c)).sum().item()
            tn_arousal[c] += ((predicted_arousal != c) & (targets_arousal != c)).sum().item()
            fn_arousal[c] += ((predicted_arousal != c) & (targets_arousal == c)).sum().item()
            
            # Dominance
            tp_dominance[c] += ((predicted_dominance == c) & (targets_dominance == c)).sum().item()
            fp_dominance[c] += ((predicted_dominance == c) & (targets_dominance != c)).sum().item()
            tn_dominance[c] += ((predicted_dominance != c) & (targets_dominance != c)).sum().item()
            fn_dominance[c] += ((predicted_dominance != c) & (targets_dominance == c)).sum().item()

# 计算每个类别的 Precision、Recall、Specificity 和 F1-score
def calculate_metrics(tp, fp, tn, fn):
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return precision, recall, specificity, f1

# 计算每个类别的指标
valence_metrics = []
arousal_metrics = []
dominance_metrics = []

for c in range(5):
    precision_valence, recall_valence, specificity_valence, f1_valence = calculate_metrics(tp_valence[c], fp_valence[c], tn_valence[c], fn_valence[c])
    precision_arousal, recall_arousal, specificity_arousal, f1_arousal = calculate_metrics(tp_arousal[c], fp_arousal[c], tn_arousal[c], fn_arousal[c])
    precision_dominance, recall_dominance, specificity_dominance, f1_dominance = calculate_metrics(tp_dominance[c], fp_dominance[c], tn_dominance[c], fn_dominance[c])
    
    valence_metrics.append((precision_valence, recall_valence, specificity_valence, f1_valence))
    arousal_metrics.append((precision_arousal, recall_arousal, specificity_arousal, f1_arousal))
    dominance_metrics.append((precision_dominance, recall_dominance, specificity_dominance, f1_dominance))

# 打印每个类别的指标
for c in range(5):
    print(f"Valence Class {c+1}:")
    print(f"  Precision: {valence_metrics[c][0]:.4f}, Recall: {valence_metrics[c][1]:.4f}, Specificity: {valence_metrics[c][2]:.4f}, F1-score: {valence_metrics[c][3]:.4f}")
    print(f"Arousal Class {c+1}:")
    print(f"  Precision: {arousal_metrics[c][0]:.4f}, Recall: {arousal_metrics[c][1]:.4f}, Specificity: {arousal_metrics[c][2]:.4f}, F1-score: {arousal_metrics[c][3]:.4f}")
    print(f"Dominance Class {c+1}:")
    print(f"  Precision: {dominance_metrics[c][0]:.4f}, Recall: {dominance_metrics[c][1]:.4f}, Specificity: {dominance_metrics[c][2]:.4f}, F1-score: {dominance_metrics[c][3]:.4f}")

# 计算总体指标
accuracy_valence = correct_valence / total_samples
accuracy_arousal = correct_arousal / total_samples
accuracy_dominance = correct_dominance / total_samples

print("Final Results:")
print(f"Valence Accuracy: {accuracy_valence:.4f}")
print(f"Arousal Accuracy: {accuracy_arousal:.4f}")
print(f"Dominance Accuracy: {accuracy_dominance:.4f}")

# 恢复默认输出
sys.stdout.close()
sys.stdout = sys.__stdout__
