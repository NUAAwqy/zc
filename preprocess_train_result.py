#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

@Desc  : 对模型的训练结果进行处理：
            训练集和验证集损失及正确率曲线
            绘制混淆矩阵
            分类报告
            绘制 ROC曲线，精度召回曲线
        PyTorch版本
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端，用于在线程中保存图片
import matplotlib.pyplot as plt
import torch

import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import scikitplot as skplt

from training_model import CNN1D, LSTMModel, GRUModel, device


def plot_history_curcvs(history, save_path, model_name):
    '''
    绘制 训练集 和 验证集 的 损失 及 正确率 曲线
    :param history: 模型训练的历史记录对象
    :param save_path: 生成图片的保存路径
    :param model_name: 模型名称
    :return:
    '''
    import os
    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    acc = history.history['acc']  # 每一轮 在 训练集 上的 精度
    val_acc = history.history['val_acc']  # 每一轮 在 验证集 上的 精度
    loss = history.history['loss']  # 每一轮 在 训练集 上的 损失
    val_loss = history.history['val_loss']  # 每一轮 在 验证集 上的 损失

    epochs = range(len(acc))

    # 绘制正确率曲线
    fig1 = plt.figure(figsize=(8, 6))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    acc_path = save_path + '/' + model_name + '_train_valid_acc.png'
    plt.savefig(acc_path, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f'正确率曲线已保存至: {acc_path}')

    # 绘制损失曲线
    fig2 = plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    loss_path = save_path + '/' + model_name + '_train_valid_loss.png'
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f'损失曲线已保存至: {loss_path}')


def plot_confusion_matrix(model, model_name, save_path, X_test, y_test):
    '''
    绘制混淆矩阵
    :param model: 模型
    :param model_name: 模型名称
    :param save_path: 生成图片的保存路径
    :param X_test: 测试集
    :param y_test: 测试集标签
    :return:
    '''
    import os
    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 随机森林使用sklearn
    if 'random_forest' == model_name:
        y_preds = model.predict(X_test)
    else:  # PyTorch模型
        model.eval()
        X_test_tensor = torch.FloatTensor(X_test)
        
        # 根据模型类型调整输入维度
        if isinstance(model, CNN1D):
            X_test_tensor = X_test_tensor.unsqueeze(1)  # (N, 1, L)
        elif isinstance(model, (LSTMModel, GRUModel)):
            X_test_tensor = X_test_tensor.unsqueeze(1)  # (N, 1, L)
        
        with torch.no_grad():
            X_test_tensor = X_test_tensor.to(device)
            outputs = model(X_test_tensor)
            _, y_preds = torch.max(outputs, 1)
            y_preds = y_preds.cpu().numpy()

    y_test = [np.argmax(item) for item in y_test]  # one-hot解码

    # 绘制混淆矩阵
    con_mat = confusion_matrix(y_test, y_preds)

    con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]  # 归一化
    con_mat_norm = np.around(con_mat_norm, decimals=2)  # np.around(): 四舍五入

    fig = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_norm,
                annot=True,  # annot: 默认为False，为True的话，会在格子上显示数字
                cmap='Blues'  # 热力图颜色
                )

    plt.ylim(0, 10)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    cm_path = save_path + '/' + model_name + '_confusion_matrix.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'混淆矩阵已保存至: {cm_path}')


def brief_classification_report(model, model_name, X_test, y_test):
    '''
    计算 分类报告
    :param model: 模型
    :param model_name:  模型名称
    :param X_test: 测试集
    :param y_test: 测试集标签
    :return: classification_report：分类报告
    '''
    # 随机森林使用sklearn
    if 'random_forest' == model_name:
        y_preds = model.predict(X_test)
    else:  # PyTorch模型
        model.eval()
        X_test_tensor = torch.FloatTensor(X_test)
        
        # 根据模型类型调整输入维度
        if isinstance(model, CNN1D):
            X_test_tensor = X_test_tensor.unsqueeze(1)  # (N, 1, L)
        elif isinstance(model, (LSTMModel, GRUModel)):
            X_test_tensor = X_test_tensor.unsqueeze(1)  # (N, 1, L)
        
        with torch.no_grad():
            X_test_tensor = X_test_tensor.to(device)
            outputs = model(X_test_tensor)
            _, y_preds = torch.max(outputs, 1)
            y_preds = y_preds.cpu().numpy()

    y_test = [np.argmax(item) for item in y_test]  # one-hot解码
    classification_report = metrics.classification_report(y_test, y_preds)

    return classification_report


def plot_metrics(model, model_name, save_path, X_test, y_test):
    '''
    绘制 ROC曲线 和 精度召回曲线
    :param model: 模型
    :param model_name: 模型名称
    :param save_path: 生成图片的保存路径
    :param X_test: 测试集
    :param y_test: 测试集标签
    :return:
    '''
    import os
    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 随机森林使用sklearn的predict_proba
    if 'random_forest' == model_name:
        y_probas = model.predict_proba(X_test)
    else:  # PyTorch模型
        model.eval()
        X_test_tensor = torch.FloatTensor(X_test)
        
        # 根据模型类型调整输入维度
        if isinstance(model, CNN1D):
            X_test_tensor = X_test_tensor.unsqueeze(1)  # (N, 1, L)
        elif isinstance(model, (LSTMModel, GRUModel)):
            X_test_tensor = X_test_tensor.unsqueeze(1)  # (N, 1, L)
        
        with torch.no_grad():
            X_test_tensor = X_test_tensor.to(device)
            outputs = model(X_test_tensor)
            # 使用softmax获取概率
            y_probas = torch.softmax(outputs, dim=1)
            y_probas = y_probas.cpu().numpy()
    
    y_test = [np.argmax(item) for item in y_test]  # one-hot解码

    # 绘制"ROC曲线"
    skplt.metrics.plot_roc(y_test, y_probas, title=model_name+' ROC Curves', figsize=(7, 7))
    roc_path = save_path + '/' + model_name + '_ROC_Curves.png'
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'ROC曲线已保存至: {roc_path}')

    # 绘制"精度召回曲线"
    skplt.metrics.plot_precision_recall(y_test, y_probas, title=model_name+' Precision-Recall Curves', figsize=(7, 7))
    pr_path = save_path + '/' + model_name + '_Precision_Recall_Curves.png'
    plt.savefig(pr_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'精度召回曲线已保存至: {pr_path}')