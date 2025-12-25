#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


@Desc  : 训练模型的相关函数 - PyTorch版本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from feature_extraction import feature_extraction

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================== 模型定义 ==========================

class CNN1D(nn.Module):
    """1D CNN模型"""
    def __init__(self, input_size, num_classes=10):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=20, stride=8, padding=10)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # 计算卷积后的大小
        conv_out_size = ((input_size + 2*10 - 20) // 8 + 1) // 4
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * conv_out_size, 100)
        self.fc2 = nn.Linear(100, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LSTMModel(nn.Module):
    """LSTM模型"""
    def __init__(self, input_size, num_classes=10):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=64, batch_first=True)
        self.dropout1 = nn.Dropout(0.5)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.dropout2 = nn.Dropout(0.5)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=256, batch_first=True)
        self.dropout3 = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x, _ = self.lstm3(x)
        x = self.dropout3(x)
        # 只取最后一个时间步的输出
        x = x[:, -1, :]
        x = self.fc(x)
        return x


class GRUModel(nn.Module):
    """GRU模型"""
    def __init__(self, input_size, num_classes=10):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=64, batch_first=True)
        self.dropout1 = nn.Dropout(0.5)
        self.gru2 = nn.GRU(input_size=64, hidden_size=128, batch_first=True)
        self.dropout2 = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x, _ = self.gru1(x)
        x = self.dropout1(x)
        x, _ = self.gru2(x)
        x = self.dropout2(x)
        # 只取最后一个时间步的输出
        x = x[:, -1, :]
        x = self.fc(x)
        return x


# ========================== 训练辅助类 ==========================

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


# ========================== 训练函数 ==========================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        _, true_labels = torch.max(labels.data, 1)
        total += labels.size(0)
        correct += (predicted == true_labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            _, true_labels = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted == true_labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def training_with_1D_CNN(X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size=128, epochs=20, num_classes=10):
    '''
    使用 1D_CNN 进行训练
    :param X_train: 训练集
    :param y_train: 训练集标签
    :param X_valid: 验证集
    :param y_valid: 验证集标签
    :param X_test: 测试集
    :param y_test: 测试集标签
    :param batch_size: 模型训练的 批次大小
    :param epochs: 模型训练的轮数
    :param num_classes: 分类数
    :return:
            model：训练完成的模型
            history：模型训练的历史记录
            score：模型在测试集上的得分
    '''
    # 数据转换为PyTorch张量
    X_train = torch.FloatTensor(X_train).unsqueeze(1)  # (N, 1, L)
    X_valid = torch.FloatTensor(X_valid).unsqueeze(1)
    X_test = torch.FloatTensor(X_test).unsqueeze(1)
    y_train = torch.FloatTensor(y_train)
    y_valid = torch.FloatTensor(y_valid)
    y_test = torch.FloatTensor(y_test)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    input_size = X_train.shape[2]
    model = CNN1D(input_size, num_classes).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 训练历史
    history = TrainingHistory()
    
    # 训练模型
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, valid_loader, criterion, device)
        
        history.update(train_loss, train_acc, val_loss, val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # 评估模型
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    score = [test_loss, test_acc]
    
    return model, history, score


def training_with_LSTM(X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size=128, epochs=60, num_classes=10):
    '''
    使用 LSTM 进行训练
    :param X_train: 训练集
    :param y_train: 训练集标签
    :param X_valid: 验证集
    :param y_valid: 验证集标签
    :param X_test: 测试集
    :param y_test: 测试集标签
    :param batch_size: 模型训练的 批次大小
    :param epochs: 模型训练的轮数
    :param num_classes: 分类数
    :return:
            model：训练完成的模型
            history：模型训练的历史记录
            score：模型在测试集上的得分
    '''
    # 数据转换为PyTorch张量 - LSTM需要 (batch, seq_len, features)
    X_train = torch.FloatTensor(X_train).unsqueeze(1)  # (N, 1, L)
    X_valid = torch.FloatTensor(X_valid).unsqueeze(1)
    X_test = torch.FloatTensor(X_test).unsqueeze(1)
    y_train = torch.FloatTensor(y_train)
    y_valid = torch.FloatTensor(y_valid)
    y_test = torch.FloatTensor(y_test)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    input_size = X_train.shape[2]
    model = LSTMModel(input_size, num_classes).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练历史
    history = TrainingHistory()
    
    # 训练模型
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, valid_loader, criterion, device)
        
        history.update(train_loss, train_acc, val_loss, val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # 评估模型
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    score = [test_loss, test_acc]
    
    return model, history, score


def training_with_GRU(X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size=128, epochs=60, num_classes=10):
    '''
    使用 GRU 进行训练
    :param X_train: 训练集
    :param y_train: 训练集标签
    :param X_valid: 验证集
    :param y_valid: 验证集标签
    :param X_test: 测试集
    :param y_test: 测试集标签
    :param batch_size: 模型训练的 批次大小
    :param epochs: 模型训练的轮数
    :param num_classes: 分类数
    :return:
            model：训练完成的模型
            history：模型训练的历史记录
            score：模型在测试集上的得分
    '''
    # 数据转换为PyTorch张量
    X_train = torch.FloatTensor(X_train).unsqueeze(1)  # (N, 1, L)
    X_valid = torch.FloatTensor(X_valid).unsqueeze(1)
    X_test = torch.FloatTensor(X_test).unsqueeze(1)
    y_train = torch.FloatTensor(y_train)
    y_valid = torch.FloatTensor(y_valid)
    y_test = torch.FloatTensor(y_test)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    input_size = X_train.shape[2]
    model = GRUModel(input_size, num_classes).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练历史
    history = TrainingHistory()
    
    # 训练模型
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, valid_loader, criterion, device)
        
        history.update(train_loss, train_acc, val_loss, val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # 评估模型
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    score = [test_loss, test_acc]
    
    return model, history, score


def training_with_random_forest(X_train, y_train, X_valid, y_valid, X_test, y_test):
    '''
    使用 随机森林 进行训练
    :param X_train: 训练集
    :param y_train: 训练集标签
    :param X_valid: 验证集
    :param y_valid: 验证集标签
    :param X_test: 测试集
    :param y_test: 测试集标签
    :return:
            clf_rfc：训练完成的模型
            score：模型在验证集上的得分
            X_train_feature_extraction：将原数据进行了特征提取过的训练集
            X_test_feature_extraction：将原数据进行了特征提取过的测试集
    '''
    # 把训练集和验证集合并，全部用作训练集
    X_train = np.vstack((X_train, X_valid))
    y_train = np.vstack((y_train, y_valid))

    # 将one-hot编码了的标签解码（这里不需要one-hot编码）
    y_train = [np.argmax(item) for item in y_train]
    y_train = np.array(y_train)
    y_test = [np.argmax(item) for item in y_test]
    y_test = np.array(y_test)

    loader = np.empty(shape=[X_train.shape[0], 16])
    for i in range(X_train.shape[0]):
        loader[i] = feature_extraction(X_train[i])
    X_train_feature_extraction = loader

    loader = np.empty(shape=[X_test.shape[0], 16])
    for i in range(X_test.shape[0]):
        loader[i] = feature_extraction(X_test[i])
    X_test_feature_extraction = loader

    clf_rfc = RandomForestClassifier(n_estimators=17, max_depth=21, criterion='gini', min_samples_split=2,
                                       max_features=9, random_state=60 )
    clf_rfc.fit(X_train_feature_extraction, y_train)
    score = clf_rfc.score(X_test_feature_extraction, y_test)
    return clf_rfc, score, X_train_feature_extraction, X_test_feature_extraction