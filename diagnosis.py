#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


@Desc  : 故障诊断的相关函数 - PyTorch版本
"""
import numpy as np
import joblib
import torch

from feature_extraction import feature_extraction
from training_model import CNN1D, LSTMModel, GRUModel, device


def diagnosis(diagnosis_samples, model_file_path):
    """
    故障诊断
    :param diagnosis_samples: 数据样本
    :param model_file_path: 模型路径
    :return: pred_result：诊断结果
    """
    suffix = model_file_path.split('/')[-1].split('.')[-1]  # 获得所选模型的后缀名
    if 'm' == suffix:  # 说明是随机森林
        # 提取特征
        loader = np.empty(shape=[diagnosis_samples.shape[0], 16])
        for i in range(diagnosis_samples.shape[0]):
            loader[i] = feature_extraction(diagnosis_samples[i])
        diagnosis_samples_feature_extraction = loader

        # 加载模型
        model = joblib.load(model_file_path)
        # 使用模型进行诊断
        y_preds = model.predict(diagnosis_samples_feature_extraction)
    else:  # PyTorch模型 (.pth 或 .pt)
        # 加载模型
        model = torch.load(model_file_path, map_location=device, weights_only=False)
        model.eval()
        
        # 转换数据为张量
        diagnosis_samples_tensor = torch.FloatTensor(diagnosis_samples)
        
        # 判断是CNN还是LSTM/GRU
        if isinstance(model, CNN1D):
            diagnosis_samples_tensor = diagnosis_samples_tensor.unsqueeze(1)  # (N, 1, L)
        elif isinstance(model, (LSTMModel, GRUModel)):
            diagnosis_samples_tensor = diagnosis_samples_tensor.unsqueeze(1)  # (N, 1, L)
        
        # 预测
        with torch.no_grad():
            diagnosis_samples_tensor = diagnosis_samples_tensor.to(device)
            outputs = model(diagnosis_samples_tensor)
            _, y_preds = torch.max(outputs, 1)
            y_preds = y_preds.cpu().numpy()

    y_preds = list(y_preds)
    # 计算这些样本诊断结果中出现次数最多的结果作为最后结果
    y_pred = max(y_preds, key=y_preds.count)
    pred_result = result_decode(y_pred)

    return pred_result


def result_decode(y_pred):
    '''
    将数字表示的诊断结果解码为文字
    :param y_pred:
    :return:
    '''
    if 0 == y_pred:
        pred_result = '滚动体故障：0.1778mm'
    elif 1 == y_pred:
        pred_result = '滚动体故障：0.3556mm'
    elif 2 == y_pred:
        pred_result = '滚动体故障：0.5334mm'
    elif 3 == y_pred:
        pred_result = '内圈故障：0.1778mm'
    elif 4 == y_pred:
        pred_result = '内圈故障：0.3556mm'
    elif 5 == y_pred:
        pred_result = '内圈故障：0.5334mm'
    elif 6 == y_pred:
        pred_result = '外圈故障（6点方向）：0.1778mm'
    elif 7 == y_pred:
        pred_result = '外圈故障（6点方向）：0.3556mm'
    elif 8 == y_pred:
        pred_result = '外圈故障（6点方向）：0.5334mm'
    elif 9 == y_pred:
        pred_result = '正常'

    return pred_result