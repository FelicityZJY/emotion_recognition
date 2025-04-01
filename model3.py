import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import scipy.io
import sys
from sklearn.model_selection import KFold, train_test_split
import sys
import os
from thop import profile, clever_format

class EPA(nn.Module):
    """
        Efficient Paired Attention Block, based on: "Shaker et al.,
        UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
        """
    def __init__(self, hidden_size, proj_size, num_heads=3, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)

        # E and F are projection matrices with shared weights used in spatial attention module to project
        # keys and values from HWD-dimension to P-dimension
        self.E = self.F = nn.Linear(hidden_size // num_heads, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))

    def forward(self, x):
        B, C, N = x.shape
        #let hidden_size=N
        qkvv = self.qkvv(x).reshape(B, C, 4, self.num_heads, self.hidden_size // self.num_heads) #(B,C,4hidden_size)

        qkvv = qkvv.permute(2, 0, 3, 1, 4) #(4,B,num_heads,C, hidden_size // self.num_heads)

        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3] #(B,num_heads,C, hidden_size // num_heads)

        v_CA = v_CA.transpose(-2, -1) #(B,num_heads, hidden_size // self.num_heads, C)

        k_shared_projected = self.E(k_shared)#(B, num_heads,C, proj_size)
        k_shared = k_shared.transpose(-2, -1)  # (B, num_heads, hidden_size // num_heads, C)

        v_SA_projected = self.F(v_SA)#(B, num_heads,C, proj_size)
        v_SA_projected = v_SA_projected.transpose(-2, -1)#(B, num_heads, proj_size,C)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared) * self.temperature #(B, num_heads, C, C)
        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)
        x_CA = (attn_CA @ v_CA.transpose(-2,-1)).permute(0, 2, 1, 3).reshape(B, C, N) #(B, C, N)
        #(B,num_heads,C,hidden_size//num_heads)->(B,C,num_heads,hidden_size//num_heads)->(B, C, N)
        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2
        #(B,num_heads,C, hidden_size // num_heads)->(B,num_heads, hidden_size // num_heads,C)->(B,num_heads,hidden_size//num_heads,proj_size)
        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)
        x_SA = (attn_SA @ v_SA_projected).permute(0, 3, 1, 2).reshape(B, C, N)
        #(B,num_heads,hidden_size//num_heads,C)->(B,C,num_heads,hidden_size//num_heads)->(B, C, N)

        # Concat fusion
        x_SA = x_SA + x
        x_CA = x_CA + x
        x = torch.cat((x_SA, x_CA), dim=-1)
        return x #(B, C, 2N)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature', 'temperature2'}



class ImprovedTransformerEncoder(nn.Module):
    def __init__(self, embed_dim,proj_size, num_heads, out_size,mlp_ratio=4):
        super(ImprovedTransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = EPA(embed_dim,proj_size, num_heads)
        self.norm2 = nn.LayerNorm(2*embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2*embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), out_size)
        )

    def forward(self, x):
        x = self.norm1(x)
        x = self.attn(x)
        x = self.norm2(x)
        x = self.mlp(x)
        return x


class ConvolutionalLayers(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalLayers, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))  # (240,32,9,9)
        x = self.relu(self.conv2(x))  # (240,8,9,9)
        x = x.flatten(2)  # Reshape to (B, C, H*W)
        # x = x.permute(0, 2, 1)  # Reshape to (B, H*W, C)
        return x


class ClassificationLayer(nn.Module):
    def __init__(self, C, N, num_classes):
        super(ClassificationLayer, self).__init__()
        # 线性变换层，将 C 维度映射到 1 维度
        self.linear_transform = nn.Linear(C, 1)
        # 全连接层，将 N 维度映射到 num_classes 维度
        self.fc = nn.Linear(2*N, num_classes)
        # SoftMax 层
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # self.linear_transform 将 C 维度映射到 1 维度
        B, C, N = x.shape
        # 调整 x 的形状为 (B * N, C)
        x = x.permute(0, 2, 1)  # 形状: (B, N, C)
        x = x.reshape(-1, C)  # 形状: (B * N, C)
        # 线性变换得到 z
        z = self.linear_transform(x)  # 形状: (B * N, 1)
        # 恢复 z 的形状为 (B, N, 1)
        z = z.reshape(B, N, 1)  # 形状: (B, N, 1)
        # 去掉多余的维度
        z = z.squeeze(dim=2)  # 形状: (B, N)
        # 全连接层
        x = self.fc(z)  # 形状: (B, num_classes)
        # SoftMax 层
        x = self.softmax(x)  # 形状不变: (B, num_classes)
        return x


class TPRO_NET(nn.Module):
    def __init__(self, in_channels, embed_dim, out_size,num_heads,num_classes):
        super(TPRO_NET, self).__init__()
        self.conv_layers = ConvolutionalLayers(in_channels, embed_dim)
        self.downsample = nn.MaxPool1d(kernel_size=3)  # 下采样层
        self.transformer_layers = nn.ModuleList([
            ImprovedTransformerEncoder(embed_dim=81, proj_size=9, out_size= out_size,num_heads=num_heads),
            ImprovedTransformerEncoder(embed_dim=27, proj_size=3, out_size= out_size, num_heads=num_heads),
            ImprovedTransformerEncoder(embed_dim=9, proj_size=1, out_size= out_size, num_heads=num_heads),
            ImprovedTransformerEncoder(embed_dim=3, proj_size=1, out_size= out_size, num_heads=num_heads)
        ])
        # 为每个指标定义单独的分类层
        self.classifier_valence = ClassificationLayer(in_channels, embed_dim, num_classes)
        self.classifier_arousal = ClassificationLayer(in_channels, embed_dim, num_classes)
        self.classifier_dominance = ClassificationLayer(in_channels, embed_dim, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)  # (B, C, N)
        o0 = self.transformer_layers[0](x)  # (B, C, 2N)
        x1 = self.downsample(x)  # (B, C, N/3)
        o1 = self.transformer_layers[1](x1)  # (B, C, 2N)
        x2 = self.downsample(x1)  # (B, C, N/9)
        o2 = self.transformer_layers[2](x2) # (B, C, 2N)
        x3 = self.downsample(x2)  # (B, C, N/27)
        o3 = self.transformer_layers[3](x3)  # (B, C, 2N)
        x = o0 + o1 + o2 + o3  # (B, C, 2N)
        # 为每个指标获取输出
        output_valence = self.classifier_valence(x)  # (B,C,class)
        output_arousal = self.classifier_arousal(x)
        output_dominance = self.classifier_dominance(x)
        return output_valence, output_arousal, output_dominance


#########################################################
# 计算每个类别的 Precision、Recall、Specificity 和 F1-score
def calculate_metrics(tp, fp, tn, fn):
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return precision, recall, specificity, f1


##########################################################
# 在脚本开头打开文件
original_stdout = sys.stdout  # 保存原始的标准输出
sys.stdout = open('c-2.txt', 'w')  # 打开文件并设置为新的标准输出

# 加载 DREAMER 数据集
data = scipy.io.loadmat('DREAMER.mat')
eeg_data = data['DREAMER'][0, 0]  # 假设数据存储在 'DREAMER' 变量中
sampling_rate = 128  # DREAMER 数据集的采样率
all_sec = np.zeros((23, 18))
for subject in range(23):
    for trial in range(18):
        sti = eeg_data["Data"][0, subject]["EEG"][0, 0]["stimuli"][0, 0][trial, 0]
        all_sec[subject][trial] = sti.shape[0] / sampling_rate

# 加载数据
data_path = "processed_DREAMER_features.npy"
data = np.load(data_path)  # (85744,2,4,9,9)

# 加载标签
label_path = "label.txt"
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
# 先看这个加载数据对不对！！！！！
C = 8
H = 9
W = 9
dataset = []
subjects = []
for subject in range(23):
    for trial in range(18):
        for sec in range(int(all_sec[subject][trial])):
            second = subject * 18 + trial
            data_tensor = torch.tensor(data[second], dtype=torch.float32)
            data_tensor = data_tensor.reshape(8, 9, 9)
            label_tensor = torch.tensor(labels[subject, trial], dtype=torch.long)
            dataset.append((data_tensor, label_tensor))

# 外层划分训练集和测试集
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

#######################################
# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义模型
in_channels = 8  # 输入通道数
embed_dim = H * W  # Transformer 嵌入维度
num_heads = 3  # 多头注意力头数
num_classes = 5  # DREAMER 数据集的分类数

# 初始化模型、损失函数和优化器
model = TPRO_NET(in_channels=in_channels, embed_dim=embed_dim, out_size=2*embed_dim, num_heads=num_heads, num_classes=num_classes)
model = model.to(device)  # 将模型移动到设备上
################################
'''
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
for fold, (train_indices, val_indices) in enumerate(splits.split(train_dataset)):
    print(f"Fold {fold + 1}/{k}")

    # 划分训练集和验证集
    fold_train_dataset = [train_dataset[i] for i in train_indices]
    fold_val_dataset = [train_dataset[i] for i in val_indices]

    # 定义数据加载器
    train_loader = DataLoader(fold_train_dataset, batch_size=240, shuffle=True)
    val_loader = DataLoader(fold_val_dataset, batch_size=240, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # 训练循环
    num_epochs = 50  # 根据 DREAMER 数据集设置
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for data_tensor, label_tensor in train_loader:
            inputs = data_tensor.to(device)  # (240,8,9,9) 将数据移动到设备上
            targets_valence = label_tensor[:, 0].to(device) - 1
            targets_arousal = label_tensor[:, 1].to(device) - 1
            targets_dominance = label_tensor[:, 2].to(device) - 1

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

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")
################################
# 在所有 fold 的训练和评估完成后，使用测试集进行最终评估
print("Final Evaluation on Test Set:")
test_loader = DataLoader(test_dataset, batch_size=240, shuffle=False)

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
    for data_tensor, label_tensor in test_loader:
        inputs = data_tensor.to(device)  # 将数据移动到设备上
        targets_valence = label_tensor[:, 0].to(device)
        targets_arousal = label_tensor[:, 1].to(device)
        targets_dominance = label_tensor[:, 2].to(device)

        # 前向传播
        outputs_valence, outputs_arousal, outputs_dominance = model(inputs)

        # 统计正确预测数
        predicted_valence = torch.argmax(outputs_valence, dim=1)
        predicted_valence = predicted_valence + 1
        predicted_arousal = torch.argmax(outputs_arousal, dim=1)
        predicted_arousal = predicted_arousal + 1
        predicted_dominance = torch.argmax(outputs_dominance, dim=1)
        predicted_dominance = predicted_dominance + 1

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

# 计算每个类别的指标
valence_metrics = []
arousal_metrics = []
dominance_metrics = []

for c in range(5):
    precision_valence, recall_valence, specificity_valence, f1_valence = calculate_metrics(tp_valence[c], fp_valence[c],
                                                                                           tn_valence[c], fn_valence[c])
    precision_arousal, recall_arousal, specificity_arousal, f1_arousal = calculate_metrics(tp_arousal[c], fp_arousal[c],
                                                                                           tn_arousal[c], fn_arousal[c])
    precision_dominance, recall_dominance, specificity_dominance, f1_dominance = calculate_metrics(tp_dominance[c],
                                                                                                   fp_dominance[c],
                                                                                                   tn_dominance[c],
                                                                                                   fn_dominance[c])

    valence_metrics.append((precision_valence, recall_valence, specificity_valence, f1_valence))
    arousal_metrics.append((precision_arousal, recall_arousal, specificity_arousal, f1_arousal))
    dominance_metrics.append((precision_dominance, recall_dominance, specificity_dominance, f1_dominance))

# 打印每个类别的指标
for c in range(5):
    print(f"Valence Class {c + 1}:")
    print(
        f"  Precision: {valence_metrics[c][0]:.4f}, Recall: {valence_metrics[c][1]:.4f}, Specificity: {valence_metrics[c][2]:.4f}, F1-score: {valence_metrics[c][3]:.4f}")
    print(f"Arousal Class {c + 1}:")
    print(
        f"  Precision: {arousal_metrics[c][0]:.4f}, Recall: {arousal_metrics[c][1]:.4f}, Specificity: {arousal_metrics[c][2]:.4f}, F1-score: {arousal_metrics[c][3]:.4f}")
    print(f"Dominance Class {c + 1}:")
    print(
        f"  Precision: {dominance_metrics[c][0]:.4f}, Recall: {dominance_metrics[c][1]:.4f}, Specificity: {dominance_metrics[c][2]:.4f}, F1-score: {dominance_metrics[c][3]:.4f}")

# 计算总体指标
accuracy_valence = correct_valence / total_samples
accuracy_arousal = correct_arousal / total_samples
accuracy_dominance = correct_dominance / total_samples

print("Final Results:")
print(f"Valence Accuracy: {accuracy_valence:.4f}")
print(f"Arousal Accuracy: {accuracy_arousal:.4f}")
print(f"Dominance Accuracy: {accuracy_dominance:.4f}")
'''
# 定义输入张量
B, C, H, W = 240, 8, 9, 9  # 示例输入形状
input_tensor = torch.randn(B, C, H, W).to(device)
# 计算 FLOPS 和参数量
flops, params = profile(model, inputs=(input_tensor, ), verbose=False)
flops, params = clever_format([flops, params], "%.3f")
print(f"模型的 FLOPS: {flops}")
print(f"模型的参数量: {params}")

# 脚本结束前恢复标准输出
sys.stdout.close()  # 关闭文件
sys.stdout = original_stdout  # 恢复原标准输出
