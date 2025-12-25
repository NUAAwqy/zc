#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轴承故障诊断与剩余寿命预测 Web 应用
基于Flask的后端服务器

功能模块:
1. 数据接入与处理
2. 模型训练
3. 故障诊断
4. 剩余寿命预测
5. 设备管理
6. 模型管理
7. 系统管理
"""

# 设置matplotlib后端为非交互式（必须在导入matplotlib之前）
import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import json
import torch
import joblib
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename
import threading
import uuid
import shutil

# 导入项目模块
from data_preprocess import training_stage_prepro, diagnosis_stage_prepro
from training_model import training_with_1D_CNN, training_with_LSTM, training_with_GRU, training_with_random_forest
from diagnosis import diagnosis, result_decode
from preprocess_train_result import plot_history_curcvs, plot_confusion_matrix, brief_classification_report, plot_metrics
from utils import generate_md5
from rul_prediction import predict_rul, training_rul_model

# 创建Flask应用
app = Flask(__name__, 
            static_folder='web_static',
            template_folder='web_templates')
CORS(app)

# 配置
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 最大上传100MB
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MODEL_FOLDER'] = './models'
app.config['CACHE_FOLDER'] = './cache'
app.config['DATABASE_FILE'] = './database.json'

# 确保必要的文件夹存在
for folder in [app.config['UPLOAD_FOLDER'], app.config['MODEL_FOLDER'], 
               app.config['CACHE_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# 全局变量：存储训练任务状态
training_tasks = {}
diagnosis_tasks = {}
rul_tasks = {}

# 数据库（使用JSON文件简单实现）
def load_database():
    """加载数据库"""
    if os.path.exists(app.config['DATABASE_FILE']):
        with open(app.config['DATABASE_FILE'], 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'devices': [],
        'models': [],
        'diagnoses': [],
        'rul_predictions': []
    }

def save_database(db):
    """保存数据库"""
    with open(app.config['DATABASE_FILE'], 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

# ==================== 路由：主页 ====================
@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

# ==================== API：数据管理 ====================
@app.route('/api/upload_data', methods=['POST'])
def upload_data():
    """上传数据文件"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': '没有文件'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': '没有选择文件'})
        
        if not file.filename.endswith('.mat'):
            return jsonify({'success': False, 'message': '只支持.mat文件'})
        
        # 保存文件
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'message': '文件上传成功',
            'filename': unique_filename,
            'filepath': filepath
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'上传失败: {str(e)}'})

@app.route('/api/upload_dataset', methods=['POST'])
def upload_dataset():
    """上传数据集（多个文件）"""
    try:
        if 'files[]' not in request.files:
            return jsonify({'success': False, 'message': '没有文件'})
        
        files = request.files.getlist('files[]')
        dataset_name = request.form.get('dataset_name', 'dataset')
        
        # 创建数据集文件夹
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{dataset_name}")
        os.makedirs(dataset_dir, exist_ok=True)
        
        uploaded_files = []
        for file in files:
            if file and file.filename.endswith('.mat'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(dataset_dir, filename)
                file.save(filepath)
                uploaded_files.append(filename)
        
        return jsonify({
            'success': True,
            'message': f'成功上传{len(uploaded_files)}个文件',
            'dataset_path': dataset_dir,
            'files': uploaded_files
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'上传失败: {str(e)}'})

# ==================== API：模型训练 ====================
def train_model_thread(task_id, model_type, data_path, params):
    """模型训练线程函数"""
    try:
        training_tasks[task_id]['status'] = 'running'
        training_tasks[task_id]['message'] = '正在训练模型...'
        
        # 数据预处理
        X_train, y_train, X_valid, y_valid, X_test, y_test, scaler_info = training_stage_prepro(
            data_path,
            signal_length=params['signal_length'],
            signal_number=params['signal_number'],
            normal=params['normal'],
            rate=params['rate'],
            enhance=params.get('enhance', False)
        )
        
        training_tasks[task_id]['message'] = '数据预处理完成，开始训练...'
        
        # 训练模型
        if model_type == '1D_CNN':
            model, history, score = training_with_1D_CNN(
                X_train, y_train, X_valid, y_valid, X_test, y_test,
                batch_size=params['batch_size'],
                epochs=params['epochs'],
                num_classes=10
            )
        elif model_type == 'LSTM':
            model, history, score = training_with_LSTM(
                X_train, y_train, X_valid, y_valid, X_test, y_test,
                batch_size=params['batch_size'],
                epochs=params['epochs'],
                num_classes=10
            )
        elif model_type == 'GRU':
            model, history, score = training_with_GRU(
                X_train, y_train, X_valid, y_valid, X_test, y_test,
                batch_size=params['batch_size'],
                epochs=params['epochs'],
                num_classes=10
            )
        elif model_type == 'random_forest':
            model, score, X_train_feature, X_test_feature = training_with_random_forest(
                X_train, y_train, X_valid, y_valid, X_test, y_test
            )
            history = None
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        training_tasks[task_id]['message'] = '训练完成，正在生成图表...'
        
        # 生成图表
        cache_path = os.path.join(app.config['CACHE_FOLDER'], task_id)
        os.makedirs(cache_path, exist_ok=True)
        
        if model_type != 'random_forest':
            plot_history_curcvs(history, cache_path, model_type)
            plot_confusion_matrix(model, model_type, cache_path, X_test, y_test)
            classification_report = brief_classification_report(model, model_type, X_test, y_test)
            plot_metrics(model, model_type, cache_path, X_test, y_test)
        else:
            plot_confusion_matrix(model, model_type, cache_path, X_test_feature, y_test)
            classification_report = brief_classification_report(model, model_type, X_test_feature, y_test)
            plot_metrics(model, model_type, cache_path, X_test_feature, y_test)
        
        # 保存模型
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"{model_type}_{timestamp}"
        
        if model_type == 'random_forest':
            model_path = os.path.join(app.config['MODEL_FOLDER'], f'{model_filename}.m')
            joblib.dump(model, model_path)
        else:
            model_path = os.path.join(app.config['MODEL_FOLDER'], f'{model_filename}.pth')
            torch.save(model, model_path)
        
        # 保存配置文件
        config_path = os.path.join(app.config['MODEL_FOLDER'], f'{model_filename}.json')
        md5 = generate_md5(model_path)
        model_config = {
            'mean': scaler_info['mean'],
            'std': scaler_info['std'],
            'md5': md5,
            'model_type': model_type,
            'signal_length': params['signal_length'],
            'signal_number': params['signal_number'],
            'train_time': timestamp,
            'test_accuracy': score[1] if isinstance(score, list) else score
        }
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # 更新数据库
        db = load_database()
        db['models'].append({
            'id': task_id,
            'name': model_filename,
            'type': model_type,
            'path': model_path,
            'config_path': config_path,
            'accuracy': score[1] if isinstance(score, list) else score,
            'train_time': timestamp,
            'status': 'completed'
        })
        save_database(db)
        
        # 更新任务状态
        training_tasks[task_id]['status'] = 'completed'
        training_tasks[task_id]['message'] = '训练完成！'
        training_tasks[task_id]['result'] = {
            'model_path': model_path,
            'config_path': config_path,
            'accuracy': score[1] if isinstance(score, list) else score,
            'classification_report': classification_report,
            'cache_path': cache_path
        }
        
    except Exception as e:
        training_tasks[task_id]['status'] = 'failed'
        training_tasks[task_id]['message'] = f'训练失败: {str(e)}'
        training_tasks[task_id]['error'] = str(e)

@app.route('/api/train_model', methods=['POST'])
def train_model():
    """启动模型训练"""
    try:
        data = request.json
        model_type = data.get('model_type')
        data_path = data.get('data_path')
        
        # 设置默认参数
        if model_type == 'random_forest':
            params = {
                'signal_length': data.get('signal_length', 500),
                'signal_number': data.get('signal_number', 1000),
                'normal': False,
                'rate': [0.6, 0.2, 0.2],
                'batch_size': 128,
                'epochs': 1
            }
        else:
            params = {
                'signal_length': data.get('signal_length', 2048),
                'signal_number': data.get('signal_number', 1000),
                'normal': True,
                'rate': [0.7, 0.2, 0.1],
                'batch_size': data.get('batch_size', 128),
                'epochs': data.get('epochs', 20 if model_type == '1D_CNN' else 60),
                'enhance': data.get('enhance', False)
            }
        
        # 创建训练任务
        task_id = str(uuid.uuid4())
        training_tasks[task_id] = {
            'status': 'pending',
            'message': '任务已创建',
            'model_type': model_type,
            'create_time': datetime.now().isoformat()
        }
        
        # 启动训练线程
        thread = threading.Thread(
            target=train_model_thread,
            args=(task_id, model_type, data_path, params)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': '训练任务已启动',
            'task_id': task_id
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'启动训练失败: {str(e)}'})

@app.route('/api/training_status/<task_id>', methods=['GET'])
def training_status(task_id):
    """查询训练状态"""
    if task_id in training_tasks:
        return jsonify({
            'success': True,
            'task': training_tasks[task_id]
        })
    else:
        return jsonify({
            'success': False,
            'message': '任务不存在'
        })

# ==================== API：故障诊断 ====================
def diagnose_thread(task_id, model_path, data_path):
    """诊断线程函数"""
    try:
        diagnosis_tasks[task_id]['status'] = 'running'
        diagnosis_tasks[task_id]['message'] = '正在进行诊断...'
        
        # 加载配置
        config_path = os.path.splitext(model_path)[0] + '.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 确定参数
        suffix = os.path.splitext(model_path)[1]
        if suffix == '.m':
            signal_length = 500
            signal_number = 500
            normal = False
            mean = None
            std = None
        else:
            signal_length = 2048
            signal_number = 500
            normal = True
            mean = config.get('mean')
            std = config.get('std')
        
        # 数据预处理
        diagnosis_samples = diagnosis_stage_prepro(
            data_path, signal_length, signal_number, normal, mean, std
        )
        
        # 进行诊断
        result = diagnosis(diagnosis_samples, model_path)
        
        # 更新数据库
        db = load_database()
        db['diagnoses'].append({
            'id': task_id,
            'model_path': model_path,
            'data_path': data_path,
            'result': result,
            'time': datetime.now().isoformat()
        })
        save_database(db)
        
        # 更新任务状态
        diagnosis_tasks[task_id]['status'] = 'completed'
        diagnosis_tasks[task_id]['message'] = '诊断完成！'
        diagnosis_tasks[task_id]['result'] = result
        
    except Exception as e:
        diagnosis_tasks[task_id]['status'] = 'failed'
        diagnosis_tasks[task_id]['message'] = f'诊断失败: {str(e)}'
        diagnosis_tasks[task_id]['error'] = str(e)

@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    """启动故障诊断"""
    try:
        data = request.json
        model_path = data.get('model_path')
        data_path = data.get('data_path')
        
        # 创建诊断任务
        task_id = str(uuid.uuid4())
        diagnosis_tasks[task_id] = {
            'status': 'pending',
            'message': '任务已创建',
            'create_time': datetime.now().isoformat()
        }
        
        # 启动诊断线程
        thread = threading.Thread(
            target=diagnose_thread,
            args=(task_id, model_path, data_path)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': '诊断任务已启动',
            'task_id': task_id
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'启动诊断失败: {str(e)}'})

@app.route('/api/diagnosis_status/<task_id>', methods=['GET'])
def diagnosis_status(task_id):
    """查询诊断状态"""
    if task_id in diagnosis_tasks:
        return jsonify({
            'success': True,
            'task': diagnosis_tasks[task_id]
        })
    else:
        return jsonify({
            'success': False,
            'message': '任务不存在'
        })

# ==================== API：剩余寿命预测 ====================
def rul_prediction_thread(task_id, model_path, data_path):
    """RUL预测线程函数"""
    try:
        rul_tasks[task_id]['status'] = 'running'
        rul_tasks[task_id]['message'] = '正在预测剩余寿命...'
        
        # 进行RUL预测
        rul_result = predict_rul(model_path, data_path)
        
        # 更新数据库
        db = load_database()
        db['rul_predictions'].append({
            'id': task_id,
            'model_path': model_path,
            'data_path': data_path,
            'rul': rul_result['rul'],
            'confidence': rul_result.get('confidence', 0),
            'time': datetime.now().isoformat()
        })
        save_database(db)
        
        # 更新任务状态
        rul_tasks[task_id]['status'] = 'completed'
        rul_tasks[task_id]['message'] = '预测完成！'
        rul_tasks[task_id]['result'] = rul_result
        
    except Exception as e:
        rul_tasks[task_id]['status'] = 'failed'
        rul_tasks[task_id]['message'] = f'预测失败: {str(e)}'
        rul_tasks[task_id]['error'] = str(e)

@app.route('/api/predict_rul', methods=['POST'])
def predict_rul_api():
    """启动剩余寿命预测"""
    try:
        data = request.json
        model_path = data.get('model_path')
        data_path = data.get('data_path')
        
        # 创建RUL预测任务
        task_id = str(uuid.uuid4())
        rul_tasks[task_id] = {
            'status': 'pending',
            'message': '任务已创建',
            'create_time': datetime.now().isoformat()
        }
        
        # 启动预测线程
        thread = threading.Thread(
            target=rul_prediction_thread,
            args=(task_id, model_path, data_path)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'RUL预测任务已启动',
            'task_id': task_id
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'启动预测失败: {str(e)}'})

@app.route('/api/rul_status/<task_id>', methods=['GET'])
def rul_status(task_id):
    """查询RUL预测状态"""
    if task_id in rul_tasks:
        return jsonify({
            'success': True,
            'task': rul_tasks[task_id]
        })
    else:
        return jsonify({
            'success': False,
            'message': '任务不存在'
        })

# ==================== API：模型管理 ====================
@app.route('/api/models', methods=['GET'])
def list_models():
    """获取模型列表"""
    try:
        db = load_database()
        return jsonify({
            'success': True,
            'models': db['models']
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/model/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """删除模型"""
    try:
        db = load_database()
        model = next((m for m in db['models'] if m['id'] == model_id), None)
        
        if model:
            # 删除文件
            if os.path.exists(model['path']):
                os.remove(model['path'])
            if os.path.exists(model['config_path']):
                os.remove(model['config_path'])
            
            # 从数据库删除
            db['models'] = [m for m in db['models'] if m['id'] != model_id]
            save_database(db)
            
            return jsonify({'success': True, 'message': '模型已删除'})
        else:
            return jsonify({'success': False, 'message': '模型不存在'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# ==================== API：设备管理 ====================
@app.route('/api/devices', methods=['GET'])
def list_devices():
    """获取设备列表"""
    try:
        db = load_database()
        return jsonify({
            'success': True,
            'devices': db['devices']
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/device', methods=['POST'])
def add_device():
    """添加设备"""
    try:
        data = request.json
        db = load_database()
        
        device = {
            'id': str(uuid.uuid4()),
            'name': data.get('name'),
            'type': data.get('type'),
            'location': data.get('location'),
            'status': 'normal',
            'create_time': datetime.now().isoformat()
        }
        
        db['devices'].append(device)
        save_database(db)
        
        return jsonify({'success': True, 'message': '设备已添加', 'device': device})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/device/<device_id>', methods=['DELETE'])
def delete_device(device_id):
    """删除设备"""
    try:
        db = load_database()
        db['devices'] = [d for d in db['devices'] if d['id'] != device_id]
        save_database(db)
        return jsonify({'success': True, 'message': '设备已删除'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# ==================== API：历史记录 ====================
@app.route('/api/diagnosis_history', methods=['GET'])
def diagnosis_history():
    """获取诊断历史"""
    try:
        db = load_database()
        return jsonify({
            'success': True,
            'diagnoses': db['diagnoses']
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/rul_history', methods=['GET'])
def rul_history():
    """获取RUL预测历史"""
    try:
        db = load_database()
        return jsonify({
            'success': True,
            'predictions': db['rul_predictions']
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# ==================== API：图表和文件下载 ====================
@app.route('/api/chart/<task_id>/<filename>')
def get_chart(task_id, filename):
    """获取训练图表"""
    try:
        cache_path = os.path.join(app.config['CACHE_FOLDER'], task_id)
        return send_from_directory(cache_path, filename)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 404

@app.route('/api/download_model/<model_id>')
def download_model(model_id):
    """下载模型文件"""
    try:
        db = load_database()
        model = next((m for m in db['models'] if m['id'] == model_id), None)
        
        if model and os.path.exists(model['path']):
            return send_file(model['path'], as_attachment=True)
        else:
            return jsonify({'success': False, 'message': '文件不存在'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ==================== 启动服务器 ====================
if __name__ == '__main__':
    print("""
    ============================================================
    轴承故障诊断与剩余寿命预测 Web 服务
    ============================================================
    服务器地址: http://localhost:5000
    ============================================================
    """)
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False, threaded=True)

