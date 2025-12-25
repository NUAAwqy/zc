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
from sklearn.model_selection import train_test_split
matplotlib.use('Agg')

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, session
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
from functools import wraps

# 导入项目模块
from data_preprocess import training_stage_prepro, diagnosis_stage_prepro
from training_model import training_with_1D_CNN, training_with_LSTM, training_with_GRU, training_with_random_forest
from diagnosis import diagnosis, result_decode
from preprocess_train_result import plot_history_curcvs, plot_confusion_matrix, brief_classification_report, plot_metrics
from utils import generate_md5
from rul_prediction import predict_rul, training_rul_model, generate_rul_data

# 导入MySQL认证模块
from mysql_auth import verify_user, add_user, change_password, delete_user, list_users, init_database

# 创建Flask应用
app = Flask(__name__, 
            static_folder='web_static',
            template_folder='web_templates')
CORS(app)

# 配置
app.config['SECRET_KEY'] = 'secret_key'
# 设置最大上传大小为50GB（支持大文件上传）
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MODEL_FOLDER'] = './models'
app.config['CACHE_FOLDER'] = './cache'
app.config['DATABASE_FILE'] = './database.json'

# 确保必要的文件夹存在
for folder in [app.config['UPLOAD_FOLDER'], app.config['MODEL_FOLDER'], 
               app.config['CACHE_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# 初始化MySQL数据库
init_database()

# 全局变量：存储训练任务状态
training_tasks = {}
diagnosis_tasks = {}
rul_tasks = {}

# ==================== 装饰器：登录验证 ====================
def login_required(f):
    """登录验证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return jsonify({'success': False, 'message': '请先登录', 'code': 401}), 401
        return f(*args, **kwargs)
    return decorated_function

# 模型数据库（使用JSON文件简单实现）
def load_database():
    """加载模型数据库"""
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
    """保存模型数据库"""
    with open(app.config['DATABASE_FILE'], 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

# ==================== 路由：主页 ====================
@app.route('/index')
def index():
    """主页"""
    username = session.get('username', '未登录')
    return render_template('index.html', username=username)


@app.route('/')
def login_page():
    """登录页面"""
    return render_template('login.html')

# ==================== API：用户认证 ====================
@app.route('/api/login', methods=['POST'])
def login():
    """用户登录"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            return jsonify({'success': False, 'message': '用户名和密码不能为空'})
        
        # 验证用户
        if verify_user(username, password):
            session['username'] = username
            return jsonify({
                'success': True, 
                'message': '登录成功',
                'username': username
            })
        else:
            return jsonify({'success': False, 'message': '用户名或密码错误'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'登录失败: {str(e)}'})


@app.route('/api/logout', methods=['POST'])
def logout():
    """用户登出"""
    session.pop('username', None)
    return jsonify({'success': True, 'message': '已退出登录'})


@app.route('/api/check_login', methods=['GET'])
def check_login():
    """检查登录状态"""
    if 'username' in session:
        return jsonify({
            'success': True, 
            'logged_in': True,
            'username': session['username']
        })
    else:
        return jsonify({
            'success': True,
            'logged_in': False
        })


@app.route('/api/change_password', methods=['POST'])
@login_required
def api_change_password():
    """修改密码"""
    try:
        data = request.get_json()
        old_password = data.get('old_password', '')
        new_password = data.get('new_password', '')
        
        if not old_password or not new_password:
            return jsonify({'success': False, 'message': '密码不能为空'})
        
        if len(new_password) < 6:
            return jsonify({'success': False, 'message': '新密码长度不能少于6位'})
        
        username = session['username']
        success, message = change_password(username, old_password, new_password)
        
        return jsonify({'success': success, 'message': message})
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'修改密码失败: {str(e)}'})


# ==================== API：用户管理 ====================
@app.route('/api/users', methods=['POST'])
def create_user():
    """注册用户"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            return jsonify({'success': False, 'message': '用户名和密码不能为空'})
        
        if len(password) < 6:
            return jsonify({'success': False, 'message': '密码长度不能少于6位'})
        
        success, message = add_user(username, password)
        return jsonify({'success': success, 'message': message})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
    

# ==================== API：数据管理 ====================
@app.route('/api/upload_data', methods=['POST'])
@login_required
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
@login_required
def upload_dataset():
    """上传数据集（文件夹）"""
    import gc  # 用于垃圾回收
    import psutil  # 用于内存监控（如果可用）
    
    try:
        # 检查请求大小
        if request.content_length and request.content_length > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({
                'success': False, 
                'message': f'上传失败: 数据大小 ({request.content_length / (1024*1024):.2f} MB) 超过限制 ({app.config["MAX_CONTENT_LENGTH"] / (1024*1024*1024):.2f} GB)。请尝试分批上传。'
            }), 413
        
        # 检查内存使用（如果可用）
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            # 如果内存使用超过2GB，警告
            if memory_mb > 2048:
                return jsonify({
                    'success': False,
                    'message': f'服务器内存使用过高 ({memory_mb:.0f} MB)。请尝试分批上传或稍后重试。'
                }), 503
        except ImportError:
            pass  # psutil不可用，跳过检查
        
        if 'files[]' not in request.files:
            return jsonify({'success': False, 'message': '没有文件'})
        
        files = request.files.getlist('files[]')
        dataset_name = request.form.get('dataset_name', 'dataset')
        
        if not files or len(files) == 0:
            return jsonify({'success': False, 'message': '没有选择文件'})
        
        # 限制单次上传文件数量，防止内存溢出
        MAX_FILES_PER_UPLOAD = 500
        
        # 检查是否指定了数据集目录（用于分批上传合并）
        existing_dataset_dir = request.form.get('dataset_dir', None)
        is_batch_upload = existing_dataset_dir and os.path.exists(existing_dataset_dir)
        
        if is_batch_upload:
            # 使用已存在的数据集目录（分批上传）
            dataset_dir = existing_dataset_dir
        else:
            # 创建新的数据集文件夹
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_dir = os.path.normpath(app.config['UPLOAD_FOLDER'])
            dataset_dir = os.path.join(base_dir, f"{timestamp}_{dataset_name}")
            os.makedirs(dataset_dir, exist_ok=True)
        
        # 如果文件数量超过限制，且不是分批上传，则拒绝（前端会自动分批）
        if len(files) > MAX_FILES_PER_UPLOAD and not is_batch_upload:
            return jsonify({
                'success': False,
                'message': f'文件数量过多 ({len(files)} 个)。单次最多上传 {MAX_FILES_PER_UPLOAD} 个文件。系统将自动分批上传。'
            }), 413
        
        uploaded_files = []
        failed_files = []
        
        # 流式处理文件，逐个保存，避免全部加载到内存
        for idx, file in enumerate(files):
            try:
                # 检查文件是否存在
                if not file or not file.filename:
                    continue

                # file.filename 现在是相对路径，例如 "MyFolder/data/train.mat"
                relative_path = file.filename
                
                # 提取原始目录和文件名
                original_dir = os.path.dirname(relative_path)
                original_filename = os.path.basename(relative_path)

                # 清理文件名（防止恶意文件名，如 "file.exe.mat"）
                safe_filename = secure_filename(original_filename)

                # 构建安全的目标目录路径
                target_dir = os.path.join(dataset_dir, original_dir)

                #  规范化路径（解析 ".." 等）
                safe_target_dir = os.path.normpath(target_dir)

                # 创建目标目录（如果不存在）
                os.makedirs(safe_target_dir, exist_ok=True)
                
                # 保存文件（流式写入，不全部加载到内存）
                filepath = os.path.join(safe_target_dir, safe_filename)
                file.save(filepath)
                
                # 记录保存的相对路径（相对于 dataset_dir）
                uploaded_files.append(os.path.join(original_dir, safe_filename))
                
                # 每处理100个文件，强制垃圾回收一次，释放内存
                if (idx + 1) % 100 == 0:
                    gc.collect()
                    
            except MemoryError:
                failed_files.append({
                    'filename': file.filename if file else 'unknown',
                    'error': '内存不足'
                })
                # 内存不足，停止处理
                break
            except Exception as file_error:
                failed_files.append({
                    'filename': file.filename if file else 'unknown',
                    'error': str(file_error)
                })
        
        # 最后进行一次垃圾回收
        gc.collect()
        
        if len(uploaded_files) == 0:
            return jsonify({
                'success': False,
                'message': '没有文件成功上传',
                'failed_files': failed_files
            })
        
        message = f'成功上传{len(uploaded_files)}个文件'
        if failed_files:
            message += f'，{len(failed_files)}个文件上传失败'
        
        return jsonify({
            'success': True,
            'message': message,
            'dataset_path': dataset_dir,
            'files': uploaded_files,
            'failed_files': failed_files if failed_files else None,
            'total_uploaded': len(uploaded_files),
            'total_failed': len(failed_files)
        })
    except MemoryError:
        return jsonify({
            'success': False,
            'message': '上传失败: 服务器内存不足。请尝试分批上传（每次不超过500个文件）。'
        }), 507
    except Exception as e:
        return jsonify({'success': False, 'message': f'上传失败: {str(e)}'})

@app.route('/api/load_rul_datasets', methods=['GET'])
def load_rul_datasets():
    """加载RUL数据集列表"""
    try:
        datasets = []
        base_dir = os.path.normpath(app.config['UPLOAD_FOLDER'])
        
        for dirs in os.listdir(base_dir):
            if 'bearing' in dirs:
                dataset_path = os.path.join(base_dir, dirs)
                for dataset in os.listdir(dataset_path):
                    datasets.append(os.path.join(dataset_path, dataset))
        
        return jsonify({
            'success': True,
            'datasets': datasets
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'加载数据集失败: {str(e)}'})

@app.route('/api/uploaded_files', methods=['GET'])
@login_required
def get_uploaded_files():
    """获取已上传的单个文件列表（用于诊断）"""
    try:
        files = []
        base_dir = os.path.normpath(app.config['UPLOAD_FOLDER'])
        
        # 从数据库获取历史诊断记录中的文件路径
        db = load_database()
        seen_paths = set()
        
        # 从诊断历史中获取文件路径
        for diagnosis in db.get('diagnoses', []):
            data_path = diagnosis.get('data_path', '')
            if data_path and data_path not in seen_paths:
                # 检查文件是否存在
                if os.path.exists(data_path):
                    files.append(data_path)
                    seen_paths.add(data_path)
        
        # 从uploads目录中扫描所有.mat文件
        if os.path.exists(base_dir):
            for filename in os.listdir(base_dir):
                filepath = os.path.join(base_dir, filename)
                if os.path.isfile(filepath) and filename.endswith('.mat'):
                    if filepath not in seen_paths:
                        files.append(filepath)
                        seen_paths.add(filepath)
        
        # 按文件名排序
        files.sort(key=lambda x: os.path.basename(x))
        
        return jsonify({
            'success': True,
            'files': files
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'加载文件列表失败: {str(e)}'})

@app.route('/api/uploaded_datasets', methods=['GET'])
@login_required
def get_uploaded_datasets():
    """获取已上传的数据集列表（用于训练）"""
    try:
        datasets = []
        base_dir = os.path.normpath(app.config['UPLOAD_FOLDER'])
        
        if not os.path.exists(base_dir):
            return jsonify({
                'success': True,
                'datasets': []
            })
        
        # 扫描uploads目录下的所有文件夹（数据集）
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                # 递归检查文件夹中是否存在文件（支持嵌套目录）
                has_file = False
                for root, dirs, files in os.walk(item_path):
                    # 只要发现任意文件（如 .mat/.json 等）即可认为是数据集
                    if any(files):
                        has_file = True
                        break
                if has_file:
                    datasets.append(item_path)
        
        # 按文件夹名称排序
        datasets.sort(key=lambda x: os.path.basename(x), reverse=True)
        
        return jsonify({
            'success': True,
            'datasets': datasets
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'加载数据集列表失败: {str(e)}'})

# ==================== API：模型训练 ====================
def train_model_thread(task_id, model_type, data_path, params):
    """模型训练线程函数"""
    try:
        training_tasks[task_id]['status'] = 'running'
        training_tasks[task_id]['message'] = '正在处理数据...'
        
        scaler_info = None
        # 数据预处理
        if model_type != 'RUL_LSTM' and model_type != 'RUL_CNN':
            X_train, y_train, X_valid, y_valid, X_test, y_test, scaler_info = training_stage_prepro(
                data_path,
                signal_length=params['signal_length'],
                signal_number=params['signal_number'],
                normal=params['normal'],
                rate=params['rate'],
                enhance=params.get('enhance', False)
            )
        else :
            # ========== 生成数据 ==========
            X, y = generate_rul_data(data_path, params['signal_length'])
            if params['signal_number'] is not None:
                X = X[:min(len(X), params['signal_number'])]
                y = y[:min(len(y), params['signal_number'])]
            
            # ========== 划分数据集 ==========
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=params['rate'][1], random_state=42
            )

            X_train, X_valid, y_train, y_valid = train_test_split(
                X_temp, y_temp, test_size=params['rate'][2], random_state=42
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
        elif model_type == 'RUL_LSTM':
            model, history, score = training_rul_model(
                X_train, y_train, X_valid, y_valid, X_test, y_test,
                model_type='LSTM',
                batch_size=params['batch_size'],
                epochs=params['epochs'],
                learn_rate=0.001,
            )
        elif model_type == 'RUL_CNN':
            model, history, score = training_rul_model(
                X_train, y_train, X_valid, y_valid, X_test, y_test,
                model_type='CNN',
                batch_size=params['batch_size'],
                epochs=params['epochs'],
                learn_rate=0.001,
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        training_tasks[task_id]['message'] = '训练完成，正在生成图表...'
        
        # 生成图表
        cache_path = os.path.join(app.config['CACHE_FOLDER'], task_id)
        os.makedirs(cache_path, exist_ok=True)
        
        classification_report = None
        if model_type != 'random_forest' and model_type != 'RUL_LSTM' and model_type != 'RUL_CNN':
            plot_history_curcvs(history, cache_path, model_type)
            plot_confusion_matrix(model, model_type, cache_path, X_test, y_test)
            classification_report = brief_classification_report(model, model_type, X_test, y_test)
            plot_metrics(model, model_type, cache_path, X_test, y_test)
        elif model_type == 'RUL_LSTM' or model_type == 'RUL_CNN':
            plot_history_curcvs(history, cache_path, model_type)
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
            'md5': md5,
            'model_type': model_type,
            'signal_length': params['signal_length'],
            'signal_number': params['signal_number'],
            'train_time': timestamp,
            'test_accuracy': score[1] if isinstance(score, list) else score
        }

        # 只有非RUL模型才保存mean/std
        if scaler_info is not None:
            model_config['mean'] = scaler_info['mean']
            model_config['std'] = scaler_info['std']
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
            'cache_path': cache_path,
            'model_type': model_type  # 添加模型类型信息
        }
        
    except Exception as e:
        training_tasks[task_id]['status'] = 'failed'
        training_tasks[task_id]['message'] = f'训练失败: {str(e)}'
        training_tasks[task_id]['error'] = str(e)

@app.route('/api/train_model', methods=['POST'])
@login_required
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
        elif model_type == 'RUL_LSTM' or model_type == 'RUL_CNN':
            params = {
                'signal_length': data.get('signal_length', 2048),
                'signal_number': data.get('signal_number', 1000),
                'rate': [0.6, 0.2, 0.2],
                'batch_size': data.get('batch_size', 32),
                'epochs': data.get('epochs', 50)
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
@login_required
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
        print(data_path)
        rul_result = predict_rul(model_path, data_path)
        
        # 更新数据库
        db = load_database()
        db['rul_predictions'].append({
            'id': task_id,
            'model_path': model_path,
            'data_path': data_path,
            'rul': rul_result['rul'],
            'confidence': rul_result.get('confidence', 0),
            'time': datetime.now().isoformat(),
            'status': rul_result.get('status', '未知')
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
@login_required
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
@login_required
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
@login_required
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
@login_required
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
@login_required
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
@login_required
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
@login_required
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
@login_required
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
@login_required
def get_chart(task_id, filename):
    """获取训练图表"""
    try:
        cache_path = os.path.join(app.config['CACHE_FOLDER'], task_id)
        # 规范化路径，确保使用正确的路径分隔符
        cache_path = os.path.normpath(cache_path)
        file_path = os.path.join(cache_path, filename)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'message': f'图表文件不存在: {filename}'}), 404
        
        return send_from_directory(cache_path, filename)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 404

@app.route('/api/download_model/<model_id>')
@login_required
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

# ==================== 错误处理 ====================
@app.errorhandler(413)
def request_entity_too_large(error):
    """处理413错误（请求实体过大）"""
    return jsonify({
        'success': False, 
        'message': '上传失败: 文件大小超过限制。请尝试分批上传或联系管理员增加上传限制。'
    }), 413

@app.errorhandler(500)
def internal_error(error):
    """处理500错误"""
    return jsonify({
        'success': False,
        'message': f'服务器内部错误: {str(error)}'
    }), 500

# ==================== 启动服务器 ====================
if __name__ == '__main__':
    # 检查内存（如果可用）
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        print(f"当前内存使用: {memory_mb:.0f} MB")
    except ImportError:
        print("提示: 安装 psutil 可以监控内存使用: pip install psutil")
    except Exception:
        pass
    
    print("""
    ============================================================
    轴承故障诊断与剩余寿命预测 Web 服务
    ============================================================
    服务器地址: http://localhost:5000
    
    重要提示:
    - 单次最多上传 500 个文件（防止内存溢出）
    - 对于大量文件，请分批上传
    - 建议使用 Gunicorn 等生产级服务器部署
    ============================================================
    """)
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False, threaded=True)

