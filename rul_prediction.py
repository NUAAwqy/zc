#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
剩余使用寿命（RUL - Remaining Useful Life）预测模块

基于特征退化趋势和深度学习的RUL预测
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
import json
import os
from feature_extraction import feature_extraction
from data_preprocess import diagnosis_stage_prepro

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================== RUL预测模型 ==========================

class RULModel(nn.Module):
    """基于LSTM的RUL预测模型"""
    def __init__(self, input_size, hidden_size=128, num_layers=3):
        super(RULModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        out = torch.sigmoid(out)
        return out


class RULCNNModel(nn.Module):
    """基于CNN的RUL预测模型"""
    def __init__(self, input_channels, sequence_length):
        super(RULCNNModel, self).__init__()
        # input_channels: 输入特征数（如9个健康特征）
        # sequence_length: 序列长度（如50个时间步）
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)
        
        # 计算卷积后的大小
        conv_out_size = self._get_conv_output_size(sequence_length)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(conv_out_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def _get_conv_output_size(self, seq_length):
        """计算卷积层输出大小"""
        size = seq_length
        # Conv1 + MaxPool (kernel=3, stride=1, padding=1)
        size = ((size + 2*1 - 3) // 1 + 1) // 2
        # Conv2 + MaxPool  
        size = ((size + 2*1 - 3) // 1 + 1) // 2
        # Conv3 + MaxPool
        size = ((size + 2*1 - 3) // 1 + 1) // 2
        return 256 * size
        
    def forward(self, x):
        # 输入: (batch, seq_len, features)
        # 转换为: (batch, features, seq_len) 用于Conv1d
        x = x.transpose(1, 2)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.dropout(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = self.dropout(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.dropout(x)
        
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


# ========================== 训练RUL模型 ==========================
class TrainingHistory:
    """记录训练历史"""
    def __init__(self):
        self.history = {
            'loss': [],
            'acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def update(self, loss, acc, val_loss, val_acc):
        self.history['loss'].append(loss)
        self.history['acc'].append(acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)


def calc_rul_accuracy(outputs, targets, threshold=0.1):
    """
    计算RUL预测的近似准确率
    误差在阈值范围内算作预测正确
    """
    outputs = outputs.detach().cpu().numpy().flatten()
    targets = targets.detach().cpu().numpy().flatten()
    # 绝对误差
    abs_error = np.abs(outputs - targets)
    # 阈值（真实值的10%）
    tol = np.maximum(threshold * np.abs(targets), 1e-3)
    correct = (abs_error <= tol).sum()
    return correct / len(targets)


def training_rul_model(X_train, y_train, X_valid, y_valid, X_test, y_test, model_type='LSTM', 
                       batch_size=64, epochs=100, learn_rate=0.001):
    """
    训练RUL预测模型
    
    参数:
        training_data_path: 训练数据路径
        model_type: 模型类型 ('LSTM' or 'CNN')
        batch_size: 批次大小
        epochs: 训练轮数
        max_sequence_length: 最大序列长度
        learn_rate: 学习率
        train_ratio: 训练集比例 (默认0.7)
        valid_ratio: 验证集比例 (默认0.2)，测试集比例 = 1 - train_ratio - valid_ratio

    返回:
        model: 训练好的模型
        history: 训练历史
        test_metrics: 测试集指标
    """
    # ========== 转换为PyTorch张量 ==========
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1)
    
    X_valid = torch.FloatTensor(X_valid)
    y_valid = torch.FloatTensor(y_valid).reshape(-1, 1)
    
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1)
    
    # ========== 创建模型 ==========
    input_size = X_train.shape[2]
    seq_length = X_train.shape[1]
    
    if model_type == 'LSTM':
        model = RULModel(input_size).to(device)
    else:  # CNN
        model = RULCNNModel(input_size, seq_length).to(device)
        
    # ========== 创建数据加载器 ==========
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # ========== 定义优化器和调度器 ==========
    optimizer = optim.AdamW(model.parameters(), lr=learn_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # 训练历史
    history = TrainingHistory()
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10
    
    # ========== 训练模型 ==========
    for epoch in trange(epochs, desc='epochs', unit='epoch', colour='#448844'):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        train_samples = 0
        train_acc = 0.0
        
        for batch_data in train_loader:
            inputs, targets = batch_data
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += loss.item() * inputs.size(0)
            train_mae += torch.abs(outputs - targets).sum().item()
            train_samples += inputs.size(0)
            batch_acc = calc_rul_accuracy(outputs, targets, threshold=0.5)
            train_acc += batch_acc * inputs.size(0)
        
        train_loss = train_loss / train_samples
        train_mae = train_mae / train_samples
        train_acc = train_acc / train_samples
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_samples = 0
        val_acc = 0.0
        
        with torch.no_grad():
            for batch_data in valid_loader:
                inputs, targets = batch_data
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = F.mse_loss(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                val_mae += torch.abs(outputs - targets).sum().item()
                val_samples += inputs.size(0)
                batch_acc = calc_rul_accuracy(outputs, targets, threshold=0.5)
                val_acc += batch_acc * inputs.size(0)
        
        val_loss = val_loss / val_samples
        val_mae = val_mae / val_samples
        val_acc = val_acc / val_samples
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录历史
        history.update(train_loss, train_acc, val_loss, val_acc)
        
        # # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"早停于 epoch {epoch+1}")
                break
        
        print(f'Epoch {epoch+1}/{epochs}: '
                f'Train Loss: {train_loss}, Train MAE: {train_mae}, '
                f'Val Loss: {val_loss}, Val MAE: {val_mae}')
    
    # ========== 测试阶段 ==========
    model.eval()
    test_loss = 0.0
    test_mae = 0.0
    test_samples = 0
    test_acc = 0.0
    
    with torch.no_grad():
        for batch_data in test_loader:
            inputs, targets = batch_data
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)
            
            test_loss += loss.item() * inputs.size(0)
            test_mae += torch.abs(outputs - targets).sum().item()
            test_samples += inputs.size(0)
            batch_acc = calc_rul_accuracy(outputs, targets, threshold=0.5)
            test_acc += batch_acc * inputs.size(0)
    
    test_loss = test_loss / test_samples
    test_mae = test_mae / test_samples
    test_acc = test_acc / test_samples
    
    print(f'\n测试集 - Loss: {test_loss}, MAE: {test_mae}, Accuracy: {test_acc}')

    # _save_checkpoint(model, epoch, save_path)
    
    return model, history, [test_loss, test_acc]


def generate_rul_data(json_file_dir, max_sequence_length=None, sample_length=None):
    """
    生成RUL训练数据，使用累积序列（下三角矩阵形式）
    
    参数:
        json_file_dir: JSON文件目录
        max_sequence_length: 最大序列长度（None表示使用所有数据）
    
    返回:
        X: (N, max_seq_len, features) - 带padding的序列
        y: (N, 1) - RUL标签
    """
    features = []
    labels = []
    max_sequence_length = 50

    if sample_length is not None:
        sample_length = sample_length
    else:
        sample_length = len(os.listdir(json_file_dir))

    json_files = sorted(os.listdir(json_file_dir))[:sample_length]
    for idx, json_file_name in tqdm(enumerate(json_files), desc='Loading JSON files', unit='file', colour='#448844'):
        horizontal_signals = []
        vertical_signals = []

        with open(os.path.join(json_file_dir, json_file_name), 'r') as f:
            data = json.load(f)
            for item in data:
                horizontal_signals.append(item['horizontal_signal'])
                vertical_signals.append(item['vertical_signal'])
        features.append(extract_health_features(horizontal_signals, vertical_signals))
        labels.append((len(os.listdir(json_file_dir)) - idx) / len(os.listdir(json_file_dir)))

    n_samples = len(labels)
    n_features = features[0].shape[0]
    
    # 确定最大序列长度
    if max_sequence_length is None:
        max_seq_len = n_samples
    else:
        max_seq_len = min(max_sequence_length, n_samples)

    X = np.zeros((n_samples, max_seq_len, n_features), dtype=np.float32)
    y = torch.tensor(np.array(labels, dtype=np.float32)).unsqueeze(1)
    
    for i in range(n_samples):
        seq_len = min(i + 1, max_seq_len)
        start_idx = max(0, i + 1 - max_seq_len)
        end_idx = i + 1
        
        sequence = features[start_idx:end_idx]
        X[i, -seq_len:, :] = sequence  # 右对齐填充
    
    return X, y


# ========================== RUL预测 ==========================

def predict_rul(model_path, data_path):
    """
    使用训练好的RUL模型预测剩余寿命
    
    参数:
        model_path: RUL模型路径
        data_path: 数据文件路径
        sequence_length: 序列长度（使用多少个历史数据点）
    
    返回:
        预测结果字典
    """
    # 加载模型
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    # 加载并预处理数据
    TRAINED_SEQUENCE_LENGTH = 50
    SAMPLE_LENGTH = 50
    X, y = generate_rul_data(data_path, max_sequence_length=TRAINED_SEQUENCE_LENGTH, sample_length=SAMPLE_LENGTH)
    input_sequence = X[-1, :, :]
    input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(device)  # (1, seq_len, features)

    # 预测
    with torch.no_grad():
        rul_prediction = model(input_tensor).item()

    # 计算剩余寿命
    living_time = SAMPLE_LENGTH * 10 / (1 - rul_prediction + 1e-6)
    rul = (living_time - SAMPLE_LENGTH * 10) / 3600  # 转换为小时
    rul = int(rul * 100)
    
    # 计算置信度（基于特征的标准差）
    feature_std = np.std(input_sequence, axis=0).mean()
    confidence = max(0.0, min(1.0, 1.0 - feature_std / 10.0))
    
    # 计算健康指数（0-100）
    health_index = rul_prediction * 100
    
    return {
        'rul': max(0, rul),
        'confidence': confidence,
        'health_index': health_index,
        'status': get_health_status(health_index)
    }
        


def extract_health_features(horizontal_signal, vertical_signal):
    """
    从信号中提取健康特征
    
    参数:
        signal: 振动信号
    
    返回:
        特征向量
    """
    # 时域特征
    # 均值，信号的平均值
    mean_h = np.mean(horizontal_signal)
    # 标准差，信号的波动程度
    std_h = np.std(horizontal_signal)
    # 均方根，反映信号能量
    rms_h = np.sqrt(np.mean(np.array(horizontal_signal)**2))
    # 峰值，信号的最大绝对值
    peak_h = np.max(np.abs(np.array(horizontal_signal)))
    # 峭度，衡量信号尖锐程度（异常检测常用）
    kurtosis_h = np.mean((np.array(horizontal_signal) - mean_h)**4) / (std_h**4) if std_h > 0 else 0
    # 偏度，衡量信号分布的对称性
    skewness_h = np.mean((np.array(horizontal_signal) - mean_h)**3) / (std_h**3) if std_h > 0 else 0
    # 峰值因子，峰值与均方根之比，反映冲击性
    crest_factor_h = peak_h / rms_h if rms_h > 0 else 0

    mean_v = np.mean(vertical_signal)
    std_v = np.std(vertical_signal)
    rms_v = np.sqrt(np.mean(np.array(vertical_signal)**2))
    peak_v = np.max(np.abs(np.array(vertical_signal)))
    kurtosis_v = np.mean((np.array(vertical_signal) - mean_v)**4) / (std_v**4) if std_v > 0 else 0
    skewness_v = np.mean((np.array(vertical_signal) - mean_v)**3) / (std_v**3) if std_v > 0 else 0
    crest_factor_v = peak_v / rms_v if rms_v > 0 else 0
    
    # 频域特征
    # 信号的傅里叶变换幅值
    fft_values_h = np.abs(np.fft.fft(horizontal_signal))
    # 频域均值
    fft_mean_h = np.mean(fft_values_h)
    # 频域标准差
    fft_std_h = np.std(fft_values_h)

    fft_values_v = np.abs(np.fft.fft(vertical_signal))
    fft_mean_v = np.mean(fft_values_v)
    fft_std_v = np.std(fft_values_v)

    mean = (mean_h + mean_v) / 2
    std = (std_h + std_v) / 2
    rms = (rms_h + rms_v) / 2
    peak = (peak_h + peak_v) / 2
    kurtosis = (kurtosis_h + kurtosis_v) / 2
    skewness = (skewness_h + skewness_v) / 2
    crest_factor = (crest_factor_h + crest_factor_v) / 2
    fft_mean = (fft_mean_h + fft_mean_v) / 2
    fft_std = (fft_std_h + fft_std_v) / 2
    
    return np.array([mean, std, rms, peak, kurtosis, skewness, 
                    crest_factor, fft_mean, fft_std])


def calculate_health_index(features):
    """
    根据特征计算健康指数（0-100）
    
    参数:
        features: 特征向量
    
    返回:
        健康指数 (0-100)
    """
    # 归一化特征并计算健康指数
    # 这里使用简单的加权方法
    # RMS, 峰值, 峭度 越小越健康
    
    # 正常化各个特征（这里使用经验阈值）
    rms_score = max(0, 100 - features[2] * 100)  # RMS
    peak_score = max(0, 100 - features[3] * 50)   # Peak
    kurtosis_score = max(0, 100 - abs(features[4] - 3) * 10)  # Kurtosis
    
    # 加权平均
    health_index = (rms_score * 0.4 + peak_score * 0.3 + kurtosis_score * 0.3)
    
    return max(0, min(100, health_index))


def get_health_status(health_index):
    """根据健康指数返回状态描述"""
    if health_index >= 80:
        return "健康"
    elif health_index >= 60:
        return "良好"
    elif health_index >= 40:
        return "警告"
    elif health_index >= 20:
        return "危险"
    else:
        return "严重故障"


# 测试代码：只有在直接运行此文件时才会执行
if __name__ == '__main__':
    model_path = 'models/RUL_CNN_20251114_172031.pth'
    data_path = 'uploads/Test_dataset/bearing1_3'
    result = predict_rul(model_path, data_path)
    print(result)
