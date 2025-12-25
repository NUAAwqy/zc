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
from torch.utils.data import TensorDataset, DataLoader
from scipy.io import loadmat
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
        return out


class RULCNNModel(nn.Module):
    """基于CNN的RUL预测模型"""
    def __init__(self, input_size):
        super(RULCNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=20, stride=4, padding=10)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=10, stride=2, padding=5)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)
        
        # 计算卷积后的大小
        conv_out_size = self._get_conv_output_size(input_size)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(conv_out_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def _get_conv_output_size(self, input_size):
        """计算卷积层输出大小"""
        size = input_size
        # Conv1 + MaxPool
        size = ((size + 2*10 - 20) // 4 + 1) // 2
        # Conv2 + MaxPool  
        size = ((size + 2*5 - 10) // 2 + 1) // 2
        # Conv3 + MaxPool
        size = ((size + 2*2 - 5) // 2 + 1) // 2
        return 256 * size
        
    def forward(self, x):
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
        return x


# ========================== 训练RUL模型 ==========================

def training_rul_model(X_train, y_train, X_valid, y_valid, X_test, y_test, 
                      model_type='LSTM', batch_size=64, epochs=100):
    """
    训练RUL预测模型
    
    参数:
        X_train: 训练集特征 (N, seq_len, features) for LSTM or (N, features) for CNN
        y_train: 训练集RUL标签 (N, 1)
        X_valid: 验证集
        y_valid: 验证集标签
        X_test: 测试集
        y_test: 测试集标签
        model_type: 模型类型 ('LSTM' or 'CNN')
        batch_size: 批次大小
        epochs: 训练轮数
    
    返回:
        model: 训练好的模型
        history: 训练历史
        test_loss: 测试集损失
    """
    
    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1)
    X_valid = torch.FloatTensor(X_valid)
    y_valid = torch.FloatTensor(y_valid).reshape(-1, 1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).reshape(-1, 1)
    
    # 调整输入维度
    if model_type == 'LSTM':
        if len(X_train.shape) == 2:
            X_train = X_train.unsqueeze(1)  # (N, 1, features)
            X_valid = X_valid.unsqueeze(1)
            X_test = X_test.unsqueeze(1)
        input_size = X_train.shape[2]
        model = RULModel(input_size).to(device)
    else:  # CNN
        if len(X_train.shape) == 2:
            X_train = X_train.unsqueeze(1)  # (N, 1, features)
            X_valid = X_valid.unsqueeze(1)
            X_test = X_test.unsqueeze(1)
        input_size = X_train.shape[2]
        model = RULCNNModel(input_size).to(device)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    
    # 训练模型
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        train_samples = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_mae += torch.abs(outputs - targets).sum().item()
            train_samples += inputs.size(0)
        
        train_loss = train_loss / train_samples
        train_mae = train_mae / train_samples
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                val_mae += torch.abs(outputs - targets).sum().item()
                val_samples += inputs.size(0)
        
        val_loss = val_loss / val_samples
        val_mae = val_mae / val_samples
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"早停于 epoch {epoch+1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.2f}, '
                  f'Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f}')
    
    # 测试阶段
    model.eval()
    test_loss = 0.0
    test_mae = 0.0
    test_samples = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item() * inputs.size(0)
            test_mae += torch.abs(outputs - targets).sum().item()
            test_samples += inputs.size(0)
    
    test_loss = test_loss / test_samples
    test_mae = test_mae / test_samples
    
    print(f'\n测试集 - Loss: {test_loss:.4f}, MAE: {test_mae:.2f}')
    
    return model, history, {'test_loss': test_loss, 'test_mae': test_mae}


# ========================== RUL预测 ==========================

def predict_rul(model_path, data_path, sequence_length=10):
    """
    使用训练好的RUL模型预测剩余寿命
    
    参数:
        model_path: RUL模型路径
        data_path: 数据文件路径
        sequence_length: 序列长度（使用多少个历史数据点）
    
    返回:
        预测结果字典
    """
    try:
        # 加载模型
        model = torch.load(model_path, map_location=device)
        model.eval()
        
        # 加载并预处理数据
        # 这里我们使用诊断数据预处理函数
        diagnosis_samples = diagnosis_stage_prepro(data_path, 2048, 100, normal=True)
        
        # 提取时域特征作为健康指标
        features_list = []
        for sample in diagnosis_samples:
            features = extract_health_features(sample)
            features_list.append(features)
        
        features_array = np.array(features_list)
        
        # 如果有足够的历史数据，使用序列
        if len(features_array) >= sequence_length:
            # 使用最近的sequence_length个数据点
            sequence = features_array[-sequence_length:]
        else:
            # 不够就填充
            padding = np.tile(features_array[0], (sequence_length - len(features_array), 1))
            sequence = np.vstack([padding, features_array])
        
        # 转换为tensor
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)  # (1, seq_len, features)
        
        # 预测
        with torch.no_grad():
            rul_prediction = model(input_tensor).item()
        
        # 计算置信度（基于特征的标准差）
        feature_std = np.std(features_array, axis=0).mean()
        confidence = max(0.0, min(1.0, 1.0 - feature_std / 10.0))
        
        # 计算健康指数（0-100）
        health_index = calculate_health_index(features_array[-1])
        
        return {
            'rul': max(0, rul_prediction),  # RUL不能为负
            'confidence': confidence,
            'health_index': health_index,
            'status': get_health_status(health_index)
        }
        
    except Exception as e:
        # 如果RUL模型不存在或出错，使用基于规则的方法
        return rule_based_rul_estimation(data_path)


def extract_health_features(signal):
    """
    从信号中提取健康特征
    
    参数:
        signal: 振动信号
    
    返回:
        特征向量
    """
    # 时域特征
    mean = np.mean(signal)
    std = np.std(signal)
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    kurtosis = np.mean((signal - mean)**4) / (std**4) if std > 0 else 0
    skewness = np.mean((signal - mean)**3) / (std**3) if std > 0 else 0
    crest_factor = peak / rms if rms > 0 else 0
    
    # 频域特征
    fft_values = np.abs(np.fft.fft(signal))
    fft_mean = np.mean(fft_values)
    fft_std = np.std(fft_values)
    
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
    kurtosis_score = max(0, 100 - abs(features[4] - 3) * 10)  # Kurtosis (正常约为3)
    
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


def rule_based_rul_estimation(data_path):
    """
    基于规则的RUL估算（当没有训练好的RUL模型时使用）
    
    参数:
        data_path: 数据文件路径
    
    返回:
        预测结果字典
    """
    try:
        # 加载并预处理数据
        diagnosis_samples = diagnosis_stage_prepro(data_path, 2048, 100, normal=True)
        
        # 提取特征
        features_list = []
        for sample in diagnosis_samples:
            features = extract_health_features(sample)
            features_list.append(features)
        
        features_array = np.array(features_list)
        
        # 计算当前健康指数
        current_health = calculate_health_index(features_array[-1])
        
        # 基于健康指数估算RUL（简单线性模型）
        # 假设健康指数每小时下降0.1
        if current_health > 20:
            estimated_rul = (current_health - 20) / 0.1  # 小时
        else:
            estimated_rul = 0
        
        confidence = 0.6  # 规则方法置信度较低
        
        return {
            'rul': estimated_rul,
            'confidence': confidence,
            'health_index': current_health,
            'status': get_health_status(current_health),
            'method': 'rule-based'
        }
        
    except Exception as e:
        return {
            'rul': 0,
            'confidence': 0,
            'health_index': 0,
            'status': '未知',
            'error': str(e)
        }


# ========================== 测试代码 ==========================

if __name__ == '__main__':
    # 生成模拟RUL数据用于测试
    print("RUL预测模块测试")
    
    # 模拟训练数据
    n_samples = 1000
    seq_len = 10
    n_features = 9
    
    # 生成模拟数据：随着时间推移，RUL线性下降
    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    y = np.linspace(100, 0, n_samples).astype(np.float32)  # RUL从100到0
    
    # 划分数据集
    train_size = int(0.7 * n_samples)
    valid_size = int(0.2 * n_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_valid = X[train_size:train_size+valid_size]
    y_valid = y[train_size:train_size+valid_size]
    X_test = X[train_size+valid_size:]
    y_test = y[train_size+valid_size:]
    
    print(f"训练集: {X_train.shape}, {y_train.shape}")
    print(f"验证集: {X_valid.shape}, {y_valid.shape}")
    print(f"测试集: {X_test.shape}, {y_test.shape}")
    
    # 训练模型
    print("\n开始训练LSTM RUL模型...")
    model, history, test_metrics = training_rul_model(
        X_train, y_train, X_valid, y_valid, X_test, y_test,
        model_type='LSTM', batch_size=64, epochs=50
    )
    
    print(f"\n测试结果: MAE = {test_metrics['test_mae']:.2f}")

