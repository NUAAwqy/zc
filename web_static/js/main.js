// 全局变量
const API_BASE = '';
let currentTrainingTask = null;
let currentDiagnosisTask = null;
let currentRULTask = null;

// 页面初始化
document.addEventListener('DOMContentLoaded', function() {
    initNavigation();
    updateTime();
    loadDashboard();
    
    // 文件上传监听
    setupFileUploadListeners();
    
    // 定时更新时间
    setInterval(updateTime, 1000);
});

// 导航功能
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            const pageName = this.dataset.page;
            switchPage(pageName);
            
            // 更新导航状态
            navItems.forEach(nav => nav.classList.remove('active'));
            this.classList.add('active');
            
            // 更新页面标题
            document.getElementById('page-title').textContent = this.querySelector('span').textContent;
        });
    });
}

// 页面切换
function switchPage(pageName) {
    const pages = document.querySelectorAll('.page');
    pages.forEach(page => page.classList.remove('active'));
    
    const targetPage = document.getElementById(`${pageName}-page`);
    if (targetPage) {
        targetPage.classList.add('active');
        
        // 页面切换时加载相应数据
        switch(pageName) {
            case 'dashboard':
                loadDashboard();
                break;
            case 'models':
                loadModels();
                break;
            case 'devices':
                loadDevices();
                break;
            case 'history':
                loadHistory();
                break;
            case 'training':
                loadTrainingDatasets();
                break;
            case 'diagnosis':
                loadDiagnosisFiles();
                loadModels();
                break;
            case 'rul': 
                loadRulDatasets();
                loadRulModels();
                break;
        }
    }
}

// 更新时间
function updateTime() {
    const now = new Date();
    const timeStr = now.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
    document.getElementById('current-time').textContent = timeStr;
}

// 加载仪表盘数据
async function loadDashboard() {
    try {
        // 加载模型数量
        const modelsRes = await fetch(`${API_BASE}/api/models`);
        const modelsData = await modelsRes.json();
        document.getElementById('model-count').textContent = modelsData.success ? modelsData.models.length : 0;
        
        // 加载设备数量
        const devicesRes = await fetch(`${API_BASE}/api/devices`);
        const devicesData = await devicesRes.json();
        document.getElementById('device-count').textContent = devicesData.success ? devicesData.devices.length : 0;
        
        // 加载诊断历史
        const diagnosisRes = await fetch(`${API_BASE}/api/diagnosis_history`);
        const diagnosisData = await diagnosisRes.json();
        document.getElementById('diagnosis-count').textContent = diagnosisData.success ? diagnosisData.diagnoses.length : 0;
        
        // 数据集数量（这里简化处理）
        document.getElementById('dataset-count').textContent = '10';
    } catch (error) {
        console.error('加载仪表盘数据失败:', error);
    }
}

// 文件上传监听
function setupFileUploadListeners() {
    // 单个文件上传
    document.getElementById('single-file').addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            document.getElementById('single-file-info').innerHTML = `
                <i class="fas fa-file"></i> ${file.name} (${formatFileSize(file.size)})
                <button class="btn btn-primary" style="margin-left: 10px;" onclick="uploadSingleFile()">
                    <i class="fas fa-upload"></i> 上传
                </button>
            `;
        }
    });
    
    // 数据集文件上传
    document.getElementById('dataset-files').addEventListener('change', function(e) {
        const files = e.target.files;
        if (files.length > 0) {
            let info = `<p>已选择 ${files.length} 个文件:</p><ul>`;
            for (let file of files) {
                info += `<li><i class="fas fa-file"></i> ${file.name}</li>`;
            }
            info += '</ul>';
            document.getElementById('dataset-files-info').innerHTML = info;
        }
    });
    
    // 诊断文件上传
    document.getElementById('diagnosis-file').addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            // 先上传文件
            uploadDiagnosisFile(file);
        }
    });
    
    // RUL文件上传
    document.getElementById('rul-file').addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            uploadRULFile(file);
        }
    });
}

// 上传单个文件
async function uploadSingleFile() {
    const fileInput = document.getElementById('single-file');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('请先选择文件');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(`${API_BASE}/api/upload_data`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            alert('上传成功！');
            document.getElementById('single-file-info').innerHTML += `
                <p style="color: var(--success-color); margin-top: 10px;">
                    <i class="fas fa-check-circle"></i> 文件已保存: ${result.filename}
                </p>
            `;
        } else {
            alert('上传失败: ' + result.message);
        }
    } catch (error) {
        alert('上传失败: ' + error.message);
    }
}

// 上传数据集（支持自动分批）
async function uploadDataset() {
    const fileInput = document.getElementById('dataset-files');
    const datasetNameInput = document.getElementById('dataset-name');
    let datasetName = datasetNameInput.value;
    
    const files = Array.from(fileInput.files);

    if (files.length === 0) {
        alert('请选择一个文件夹');
        return;
    }

    if (!datasetName && files.length > 0 && files[0].webkitRelativePath) {
        datasetName = files[0].webkitRelativePath.split('/')[0];
        datasetNameInput.value = datasetName;
    }
    
    if (!datasetName) {
        alert('请输入数据集名称');
        return;
    }
    
    // 显示上传进度
    const infoDiv = document.getElementById('dataset-files-info');
    const originalContent = infoDiv.innerHTML;
    
    // 计算总大小
    let totalSize = 0;
    for (let file of files) {
        totalSize += file.size;
    }
    
    // 每批最多500个文件
    const MAX_FILES_PER_BATCH = 500;
    const totalBatches = Math.ceil(files.length / MAX_FILES_PER_BATCH);
    const needsBatching = files.length > MAX_FILES_PER_BATCH;
    
    // 显示上传界面
    infoDiv.innerHTML = `
        <div style="margin-top: 10px;">
            <p><i class="fas fa-spinner fa-spin"></i> 正在上传 ${files.length} 个文件${needsBatching ? `（自动分为 ${totalBatches} 批）` : ''}，请稍候...</p>
            <div style="background: #f0f0f0; border-radius: 4px; height: 20px; margin-top: 10px; overflow: hidden;">
                <div id="upload-progress" style="background: var(--primary-color); height: 100%; width: 0%; transition: width 0.3s;"></div>
            </div>
            <p id="upload-status" style="margin-top: 5px; font-size: 12px; color: #666;">准备上传...</p>
            ${needsBatching ? `<p id="batch-status" style="margin-top: 5px; font-size: 12px; color: #888;">批次进度: 0/${totalBatches}</p>` : ''}
        </div>
    `;
    
    let datasetDir = null;
    let totalUploaded = 0;
    let totalFailed = 0;
    const allFailedFiles = [];
    
    try {
        // 分批上传
        for (let batchIndex = 0; batchIndex < totalBatches; batchIndex++) {
            const startIdx = batchIndex * MAX_FILES_PER_BATCH;
            const endIdx = Math.min(startIdx + MAX_FILES_PER_BATCH, files.length);
            const batchFiles = files.slice(startIdx, endIdx);
            
            // 更新批次状态和进度
            if (needsBatching) {
                document.getElementById('batch-status').textContent = 
                    `批次进度: ${batchIndex + 1}/${totalBatches} (${startIdx + 1}-${endIdx}/${files.length} 个文件)`;
            }
            
            // 更新总体进度（基于已完成的批次）
            const overallProgress = (batchIndex / totalBatches) * 100;
            document.getElementById('upload-progress').style.width = overallProgress + '%';
            document.getElementById('upload-status').textContent = 
                `正在上传第 ${batchIndex + 1}/${totalBatches} 批 (${batchFiles.length} 个文件)...`;
            
            // 创建表单数据
            const formData = new FormData();
            formData.append('dataset_name', datasetName);
            
            // 如果已有数据集目录，使用它（合并批次）
            if (datasetDir) {
                formData.append('dataset_dir', datasetDir);
            }
            
            // 添加本批文件
            for (let file of batchFiles) {
                formData.append('files[]', file, file.webkitRelativePath);
            }
            
            // 创建AbortController用于超时控制
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 30 * 60 * 1000);
            
            try {
                const response = await fetch(`${API_BASE}/api/upload_dataset`, {
                    method: 'POST',
                    body: formData,
                    signal: controller.signal,
                });
                
                clearTimeout(timeoutId);
                
                if (!response.ok) {
                    throw new Error(`HTTP错误: ${response.status} ${response.statusText}`);
                }
                
                const result = await response.json();
                
                if (result.success) {
                    // 保存数据集目录（第一次）
                    if (!datasetDir) {
                        datasetDir = result.dataset_path;
                    }
                    
                    // 累计统计
                    totalUploaded += result.total_uploaded || result.files.length;
                    totalFailed += result.total_failed || (result.failed_files ? result.failed_files.length : 0);
                    
                    if (result.failed_files) {
                        allFailedFiles.push(...result.failed_files);
                    }
                    
                    // 更新进度（批次完成）
                    const completedProgress = ((batchIndex + 1) / totalBatches) * 100;
                    document.getElementById('upload-progress').style.width = completedProgress + '%';
                } else {
                    throw new Error(result.message || '上传失败');
                }
            } catch (error) {
                clearTimeout(timeoutId);
                
                // 记录本批失败的文件
                for (let file of batchFiles) {
                    allFailedFiles.push({
                        filename: file.name || file.webkitRelativePath,
                        error: error.message || '上传失败'
                    });
                }
                
                // 如果是网络错误，停止后续批次
                if (error.name === 'AbortError' || error.message.includes('Failed to fetch')) {
                    throw error;
                }
                
                // 其他错误继续下一批
                console.error(`批次 ${batchIndex + 1} 上传失败:`, error);
            }
        }
        
        // 所有批次完成
        document.getElementById('upload-progress').style.width = '100%';
        document.getElementById('upload-status').textContent = '上传完成！';
        if (needsBatching) {
            document.getElementById('batch-status').textContent = `所有批次完成: ${totalBatches}/${totalBatches}`;
        }
        
        // 显示最终结果
        infoDiv.innerHTML = originalContent + `
            <div style="color: var(--success-color); margin-top: 10px; padding: 10px; background: #f0f9ff; border-radius: 4px;">
                <i class="fas fa-check-circle"></i> <strong>上传完成！</strong><br>
                数据集路径: ${datasetDir}<br>
                总文件数: ${files.length} 个<br>
                成功上传: ${totalUploaded} 个文件
                ${totalFailed > 0 ? `<br><span style="color: var(--warning-color);">失败: ${totalFailed} 个文件</span>` : ''}
                ${needsBatching ? `<br><small style="color: #666;">已自动分为 ${totalBatches} 批上传</small>` : ''}
            </div>
        `;
        
        // 自动填充到训练页面
        if (document.getElementById('training-data-path')) {
            document.getElementById('training-data-path').value = datasetDir;
        }
        
    } catch (error) {
        let errorMessage = '上传失败: ';
        if (error.name === 'AbortError') {
            errorMessage += '请求超时（超过30分钟）。';
        } else if (error.message.includes('Failed to fetch')) {
            errorMessage += '网络连接失败。请检查网络连接或服务器状态。';
        } else {
            errorMessage += error.message;
        }
        
        infoDiv.innerHTML = originalContent + `
            <div style="color: var(--danger-color); margin-top: 10px; padding: 10px; background: #fff5f5; border-radius: 4px;">
                <i class="fas fa-exclamation-circle"></i> <strong>${errorMessage}</strong><br>
                ${totalUploaded > 0 ? `<small>已成功上传 ${totalUploaded} 个文件，失败 ${totalFailed} 个文件</small>` : ''}
            </div>
        `;
    }
}

// 开始训练
async function startTraining() {
    const modelType = document.getElementById('model-type').value;
    const dataPath = document.getElementById('training-data-path').value;
    const signalLength = parseInt(document.getElementById('signal-length').value);
    const signalNumber = parseInt(document.getElementById('signal-number').value);
    const batchSize = parseInt(document.getElementById('batch-size').value);
    const epochs = parseInt(document.getElementById('epochs').value);
    const enhance = document.getElementById('data-enhance').checked;
    
    if (!dataPath) {
        alert('请输入训练数据路径');
        return;
    }
    
    const requestData = {
        model_type: modelType,
        data_path: dataPath,
        signal_length: signalLength,
        signal_number: signalNumber,
        batch_size: batchSize,
        epochs: epochs,
        enhance: enhance
    };
    
    try {
        const response = await fetch(`${API_BASE}/api/train_model`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            currentTrainingTask = result.task_id;
            showTrainingProgress();
            monitorTrainingProgress();
        } else {
            alert('启动训练失败: ' + result.message);
        }
    } catch (error) {
        alert('启动训练失败: ' + error.message);
    }
}

// 显示训练进度
function showTrainingProgress() {
    document.getElementById('training-progress').style.display = 'block';
    document.getElementById('training-result').style.display = 'none';
}

// 监控训练进度
async function monitorTrainingProgress() {
    if (!currentTrainingTask) return;
    
    try {
        const response = await fetch(`${API_BASE}/api/training_status/${currentTrainingTask}`);
        const result = await response.json();
        
        if (result.success) {
            const task = result.task;
            document.getElementById('training-status').textContent = task.message;
            
            if (task.status === 'completed') {
                // 训练完成
                document.getElementById('training-progress').style.display = 'none';
                showTrainingResult(task.result);
            } else if (task.status === 'failed') {
                // 训练失败
                document.getElementById('training-progress').style.display = 'none';
                alert('训练失败: ' + task.message);
            } else {
                // 继续监控
                setTimeout(monitorTrainingProgress, 2000);
            }
        }
    } catch (error) {
        console.error('查询训练状态失败:', error);
        setTimeout(monitorTrainingProgress, 2000);
    }
}

// 显示训练结果
function showTrainingResult(result) {
    const resultDiv = document.getElementById('training-result');
    const contentDiv = document.getElementById('training-result-content');
    
    // 处理路径分隔符（兼容Windows和Linux）
    const modelPath = result.model_path.replace(/\\/g, '/');
    const modelName = modelPath.split('/').pop().split('.')[0];
    
    // 获取模型类型（从result中获取，如果没有则从路径推断）
    const modelType = result.model_type || modelName.split('_')[0];
    
    // 根据模型类型决定显示哪些图表
    let chartHTML = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; margin-top: 16px;">';
    
    // RUL模型只显示训练曲线
    if (modelType === 'RUL_LSTM' || modelType === 'RUL_CNN') {
        chartHTML += `
            <img src="${API_BASE}/api/chart/${currentTrainingTask}/${modelType}_train_valid_acc.png" 
                 onerror="this.style.display='none'; this.nextElementSibling.style.display='block';" 
                 style="width: 100%; border-radius: 8px;">
            <div style="display: none; padding: 20px; text-align: center; color: #999;">图表加载失败</div>
            <img src="${API_BASE}/api/chart/${currentTrainingTask}/${modelType}_train_valid_loss.png" 
                 onerror="this.style.display='none'; this.nextElementSibling.style.display='block';" 
                 style="width: 100%; border-radius: 8px;">
            <div style="display: none; padding: 20px; text-align: center; color: #999;">图表加载失败</div>
        `;
    } else {
        // 其他模型显示所有图表
        chartHTML += `
            <img src="${API_BASE}/api/chart/${currentTrainingTask}/${modelType}_train_valid_acc.png" 
                 onerror="this.style.display='none'; this.nextElementSibling.style.display='block';" 
                 style="width: 100%; border-radius: 8px;">
            <div style="display: none; padding: 20px; text-align: center; color: #999;">图表加载失败</div>
            <img src="${API_BASE}/api/chart/${currentTrainingTask}/${modelType}_train_valid_loss.png" 
                 onerror="this.style.display='none'; this.nextElementSibling.style.display='block';" 
                 style="width: 100%; border-radius: 8px;">
            <div style="display: none; padding: 20px; text-align: center; color: #999;">图表加载失败</div>
            <img src="${API_BASE}/api/chart/${currentTrainingTask}/${modelType}_confusion_matrix.png" 
                 onerror="this.style.display='none'; this.nextElementSibling.style.display='block';" 
                 style="width: 100%; border-radius: 8px;">
            <div style="display: none; padding: 20px; text-align: center; color: #999;">图表加载失败</div>
        `;
        
        // 如果有ROC和Precision-Recall曲线，也显示
        chartHTML += `
            <img src="${API_BASE}/api/chart/${currentTrainingTask}/${modelType}_ROC_Curves.png" 
                 onerror="this.style.display='none'; this.nextElementSibling.style.display='block';" 
                 style="width: 100%; border-radius: 8px;">
            <div style="display: none; padding: 20px; text-align: center; color: #999;">图表加载失败</div>
            <img src="${API_BASE}/api/chart/${currentTrainingTask}/${modelType}_Precision_Recall_Curves.png" 
                 onerror="this.style.display='none'; this.nextElementSibling.style.display='block';" 
                 style="width: 100%; border-radius: 8px;">
            <div style="display: none; padding: 20px; text-align: center; color: #999;">图表加载失败</div>
        `;
    }
    
    chartHTML += '</div>';
    
    // 分类报告部分（RUL模型可能没有）
    let classificationReportHTML = '';
    if (result.classification_report && result.classification_report !== 'null' && result.classification_report.trim() !== '') {
        classificationReportHTML = `
            <div class="result-card" style="grid-column: 1 / -1;">
                <h4>分类报告</h4>
                <pre style="font-size: 12px; overflow-x: auto;">${result.classification_report}</pre>
            </div>
        `;
    }
    
    contentDiv.innerHTML = `
        <div class="result-card">
            <h4>模型路径</h4>
            <p style="font-size: 14px; word-break: break-all;">${result.model_path}</p>
        </div>
        <div class="result-card">
            <h4>测试准确率</h4>
            <p>${(result.accuracy * 100).toFixed(2)}%</p>
        </div>
        ${classificationReportHTML}
        <div class="result-card" style="grid-column: 1 / -1;">
            <h4>训练图表</h4>
            ${chartHTML}
        </div>
    `;
    
    resultDiv.style.display = 'block';
}

// 加载模型列表
async function loadModels() {
    try {
        const response = await fetch(`${API_BASE}/api/models`);
        const result = await response.json();
        
        if (result.success) {
            // 更新模型表格
            const tbody = document.getElementById('models-list');
            tbody.innerHTML = '';
            
            result.models.forEach(model => {
                const row = `
                    <tr>
                        <td>${model.name}</td>
                        <td>${model.type}</td>
                        <td>${(model.accuracy * 100).toFixed(2)}%</td>
                        <td>${model.train_time}</td>
                        <td>
                            <button class="btn btn-danger" onclick="deleteModel('${model.id}')">
                                <i class="fas fa-trash"></i> 删除
                            </button>
                        </td>
                    </tr>
                `;
                tbody.innerHTML += row;
            });
            
            // 更新诊断页面的模型下拉框
            const diagnosisSelect = document.getElementById('diagnosis-model');
            diagnosisSelect.innerHTML = '<option value="">-- 请选择模型 --</option>';
            result.models.forEach(model => {
                diagnosisSelect.innerHTML += `<option value="${model.path}">${model.name} (${model.type})</option>`;
            });
        }
    } catch (error) {
        console.error('加载模型列表失败:', error);
    }
}

// 删除模型
async function deleteModel(modelId) {
    if (!confirm('确定要删除此模型吗？')) return;
    
    try {
        const response = await fetch(`${API_BASE}/api/model/${modelId}`, {
            method: 'DELETE'
        });
        
        const result = await response.json();
        
        if (result.success) {
            alert('删除成功');
            loadModels();
        } else {
            alert('删除失败: ' + result.message);
        }
    } catch (error) {
        alert('删除失败: ' + error.message);
    }
}

// 上传诊断文件
async function uploadDiagnosisFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(`${API_BASE}/api/upload_data`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            document.getElementById('diagnosis-data-path').value = result.filepath;
            // 上传成功后刷新文件列表
            loadDiagnosisFiles();
        } else {
            alert('上传失败: ' + result.message);
        }
    } catch (error) {
        alert('上传失败: ' + error.message);
    }
}

// 加载训练数据集列表
async function loadTrainingDatasets() {
    try {
        const response = await fetch(`${API_BASE}/api/uploaded_datasets`);
        const result = await response.json();
        
        const selectElement = document.getElementById('training-data-path-select');
        
        if (result.success && result.datasets) {
            // 清空现有选项（保留第一个选项）
            while (selectElement.options.length > 1) {
                selectElement.remove(1);
            }
            
            // 添加数据集选项
            result.datasets.forEach(datasetPath => {
                const option = document.createElement('option');
                option.value = datasetPath;
                // 显示文件夹名称
                const folderName = datasetPath.split(/[/\\]/).pop();
                option.textContent = folderName + ' (' + datasetPath + ')';
                selectElement.appendChild(option);
            });
        } else {
            console.error('加载数据集列表失败:', result.message);
        }
    } catch (error) {
        console.error('加载数据集列表时发生错误:', error);
    }
}

// 更新训练数据路径（从下拉框选择）
function updateTrainingDataPath() {
    const selectElement = document.getElementById('training-data-path-select');
    const inputElement = document.getElementById('training-data-path');
    if (selectElement.value) {
        inputElement.value = selectElement.value;
    }
}

// 加载诊断文件列表
async function loadDiagnosisFiles() {
    try {
        const response = await fetch(`${API_BASE}/api/uploaded_files`);
        const result = await response.json();
        
        const selectElement = document.getElementById('diagnosis-data-path-select');
        
        if (result.success && result.files) {
            // 清空现有选项（保留第一个选项）
            while (selectElement.options.length > 1) {
                selectElement.remove(1);
            }
            
            // 添加文件选项
            result.files.forEach(filePath => {
                const option = document.createElement('option');
                option.value = filePath;
                // 显示文件名
                const fileName = filePath.split(/[/\\]/).pop();
                option.textContent = fileName + ' (' + filePath + ')';
                selectElement.appendChild(option);
            });
        } else {
            console.error('加载文件列表失败:', result.message);
        }
    } catch (error) {
        console.error('加载文件列表时发生错误:', error);
    }
}

// 更新诊断数据路径（从下拉框选择）
function updateDiagnosisDataPath() {
    const selectElement = document.getElementById('diagnosis-data-path-select');
    const inputElement = document.getElementById('diagnosis-data-path');
    if (selectElement.value) {
        inputElement.value = selectElement.value;
    }
}

// 开始诊断
async function startDiagnosis() {
    const modelPath = document.getElementById('diagnosis-model').value;
    const dataPath = document.getElementById('diagnosis-data-path').value;
    
    if (!modelPath) {
        alert('请选择诊断模型');
        return;
    }
    
    if (!dataPath) {
        alert('请输入数据文件路径或上传文件');
        return;
    }
    
    const requestData = {
        model_path: modelPath,
        data_path: dataPath
    };
    
    try {
        const response = await fetch(`${API_BASE}/api/diagnose`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            currentDiagnosisTask = result.task_id;
            showDiagnosisProgress();
            monitorDiagnosisProgress();
        } else {
            alert('启动诊断失败: ' + result.message);
        }
    } catch (error) {
        alert('启动诊断失败: ' + error.message);
    }
}

// 显示诊断进度
function showDiagnosisProgress() {
    document.getElementById('diagnosis-progress').style.display = 'block';
    document.getElementById('diagnosis-result').style.display = 'none';
}

// 监控诊断进度
async function monitorDiagnosisProgress() {
    if (!currentDiagnosisTask) return;
    
    try {
        const response = await fetch(`${API_BASE}/api/diagnosis_status/${currentDiagnosisTask}`);
        const result = await response.json();
        
        if (result.success) {
            const task = result.task;
            document.getElementById('diagnosis-status').textContent = task.message;
            
            if (task.status === 'completed') {
                // 诊断完成
                document.getElementById('diagnosis-progress').style.display = 'none';
                showDiagnosisResult(task.result);
            } else if (task.status === 'failed') {
                // 诊断失败
                document.getElementById('diagnosis-progress').style.display = 'none';
                alert('诊断失败: ' + task.message);
            } else {
                // 继续监控
                setTimeout(monitorDiagnosisProgress, 1000);
            }
        }
    } catch (error) {
        console.error('查询诊断状态失败:', error);
        setTimeout(monitorDiagnosisProgress, 1000);
    }
}

// 显示诊断结果
function showDiagnosisResult(result) {
    const resultDiv = document.getElementById('diagnosis-result');
    const contentDiv = document.getElementById('diagnosis-result-content');
    
    // 根据结果设置图标和颜色
    let icon = 'fa-check-circle';
    let color = 'var(--success-color)';
    
    if (result.includes('故障')) {
        icon = 'fa-exclamation-triangle';
        color = 'var(--danger-color)';
    }
    
    contentDiv.innerHTML = `
        <div class="result-icon" style="color: ${color};">
            <i class="fas ${icon}"></i>
        </div>
        <div class="result-text" style="color: ${color};">${result}</div>
        <div class="result-description">诊断完成时间: ${new Date().toLocaleString('zh-CN')}</div>
    `;
    
    resultDiv.style.display = 'block';
}

// 加载RUL模型列表
async function loadRulModels() {
    const selectElement = document.getElementById('rul-model');

    while (selectElement.options.length > 1) {
        selectElement.remove(1);
    }

    try {
        const response = await fetch(`${API_BASE}/api/models`);
        
        if (!response.ok) {
            console.error('获取模型列表失败:', response.statusText);
            // 可以在此处向用户显示错误
            selectElement.options[0].textContent = '加载模型失败!';
            return;
        }

        const data = await response.json();

        if (data.success && data.models) {
            // 过滤出 RUL 相关的模型
            const rulModels = data.models.filter(model => 
                model.type === 'RUL_LSTM' || model.type === 'RUL_CNN'
            );

            // 将模型动态添加到下拉框
            rulModels.forEach(model => {
                const option = document.createElement('option');
                
                // 'value' 应该是模型路径，因为 /api/predict_rul 接口需要 model_path
                option.value = model.path; 
                
                // 'textContent' 是用户看到的文本
                const accuracy = (model.accuracy * 100).toFixed(2);
                option.textContent = `${model.name} (${model.type}, Acc: ${accuracy}%)`;
                
                selectElement.appendChild(option);
            });
            
        } else {
            selectElement.options[0].textContent = '无可用模型';
            console.error('加载模型数据格式错误:', data.message);
        }

    } catch (error) {
        console.error('加载RUL模型时发生网络错误:', error);
    }
}

// 加载数据集列表
async function loadRulDatasets() {
        const selectElement = document.getElementById('rul-dataset');

    while (selectElement.options.length > 1) {
        selectElement.remove(1);
    }

    try {
        const response = await fetch(`${API_BASE}/api/load_rul_datasets`);
        
        if (!response.ok) {
            console.error('获取数据列表失败:', response.statusText);
            // 可以在此处向用户显示错误
            selectElement.options[0].textContent = '加载数据失败!';
            return;
        }

        const data = await response.json();

        if (data.success && data.datasets) {
            // 将数据动态添加到下拉框
            data.datasets.forEach(dataset => {
                const option = document.createElement('option');
                
                option.value = dataset; 
                
                option.textContent = dataset;
                 
                selectElement.appendChild(option);
            });
            
        } else {
            selectElement.options[0].textContent = '无可用数据';
            console.error('加载数据格式错误:', data.message);
        }

    } catch (error) {
        console.error('加载RUL数据时发生网络错误:', error);
    }
}

// 开始RUL预测
async function startRULPrediction() {
    const modelPath = document.getElementById('rul-model').value;
    const dataPath = document.getElementById('rul-dataset').value;

    if (!modelPath) {
        alert('请选择预测模型');
        return;
    }
    
    if (!dataPath) {
        alert('请选择数据文件或上传文件');
        return;
    }
    
    const requestData = {
        model_path: modelPath || 'rule-based',
        data_path: dataPath
    };
    
    try {
        const response = await fetch(`${API_BASE}/api/predict_rul`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            currentRULTask = result.task_id;
            showRULProgress();
            monitorRULProgress();
        } else {
            alert('启动预测失败: ' + result.message);
        }
    } catch (error) {
        alert('启动预测失败: ' + error.message);
    }
}

// 显示RUL进度
function showRULProgress() {
    document.getElementById('rul-progress').style.display = 'block';
    document.getElementById('rul-result').style.display = 'none';
}

// 监控RUL进度
async function monitorRULProgress() {
    if (!currentRULTask) return;
    
    try {
        const response = await fetch(`${API_BASE}/api/rul_status/${currentRULTask}`);
        const result = await response.json();
        
        if (result.success) {
            const task = result.task;
            document.getElementById('rul-status').textContent = task.message;
            
            if (task.status === 'completed') {
                // 预测完成
                document.getElementById('rul-progress').style.display = 'none';
                showRULResult(task.result);
            } else if (task.status === 'failed') {
                // 预测失败
                document.getElementById('rul-progress').style.display = 'none';
                alert('预测失败: ' + task.message);
            } else {
                // 继续监控
                setTimeout(monitorRULProgress, 1000);
            }
        }
    } catch (error) {
        console.error('查询RUL状态失败:', error);
        setTimeout(monitorRULProgress, 1000);
    }
}

// 显示RUL结果
function showRULResult(result) {
    const resultDiv = document.getElementById('rul-result');
    const contentDiv = document.getElementById('rul-result-content');
    
    contentDiv.innerHTML = `
        <div class="rul-card">
            <h4>剩余使用寿命（精确到小时）</h4>
            <div class="value">${result.rul.toFixed(1)}</div>
            <div class="unit">小时（约）</div>
        </div>
        <div class="rul-card" style="background: linear-gradient(135deg, var(--success-color), var(--info-color));">
            <h4>健康指数</h4>
            <div class="value">${result.health_index.toFixed(1)}</div>
            <div class="unit">/ 100</div>
        </div>
        <div class="rul-card" style="background: linear-gradient(135deg, var(--warning-color), var(--danger-color));">
            <h4>设备状态</h4>
            <div class="value">${result.status}</div>
            <div class="unit">置信度: ${(result.confidence * 100).toFixed(1)}%</div>
        </div>
    `;
    
    resultDiv.style.display = 'block';
}

// 加载设备列表
async function loadDevices() {
    try {
        const response = await fetch(`${API_BASE}/api/devices`);
        const result = await response.json();
        
        if (result.success) {
            const container = document.getElementById('devices-list');
            container.innerHTML = '';
            
            result.devices.forEach(device => {
                const card = `
                    <div class="device-card">
                        <div class="device-card-header">
                            <div>
                                <h3>${device.name}</h3>
                                <p class="device-info"><i class="fas fa-tag"></i> ${device.type}</p>
                                <p class="device-info"><i class="fas fa-map-marker-alt"></i> ${device.location}</p>
                            </div>
                            <span class="device-status ${device.status}">${device.status === 'normal' ? '正常' : device.status}</span>
                        </div>
                        <button class="btn btn-danger" onclick="deleteDevice('${device.id}')">
                            <i class="fas fa-trash"></i> 删除
                        </button>
                    </div>
                `;
                container.innerHTML += card;
            });
        }
    } catch (error) {
        console.error('加载设备列表失败:', error);
    }
}

// 显示添加设备对话框
function showAddDeviceDialog() {
    document.getElementById('dialog-overlay').classList.add('active');
    document.getElementById('add-device-dialog').classList.add('active');
}

// 关闭对话框
function closeDialog() {
    // 隐藏遮罩层
    const overlay = document.getElementById('dialog-overlay');
    overlay.classList.remove('active');
    overlay.style.display = 'none';
    
    // 隐藏所有对话框
    document.getElementById('add-device-dialog').classList.remove('active');
    document.getElementById('add-device-dialog').style.display = 'none';
    
    document.getElementById('change-password-dialog').classList.remove('active');
    document.getElementById('change-password-dialog').style.display = 'none';
}

// 添加设备
async function addDevice() {
    const name = document.getElementById('device-name').value;
    const type = document.getElementById('device-type').value;
    const location = document.getElementById('device-location').value;
    
    if (!name || !type || !location) {
        alert('请填写所有字段');
        return;
    }
    
    const requestData = {
        name: name,
        type: type,
        location: location
    };
    
    try {
        const response = await fetch(`${API_BASE}/api/device`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            alert('添加成功');
            closeDialog();
            loadDevices();
            // 清空表单
            document.getElementById('device-name').value = '';
            document.getElementById('device-type').value = '';
            document.getElementById('device-location').value = '';
        } else {
            alert('添加失败: ' + result.message);
        }
    } catch (error) {
        alert('添加失败: ' + error.message);
    }
}

// 删除设备
async function deleteDevice(deviceId) {
    if (!confirm('确定要删除此设备吗？')) return;
    
    try {
        const response = await fetch(`${API_BASE}/api/device/${deviceId}`, {
            method: 'DELETE'
        });
        
        const result = await response.json();
        
        if (result.success) {
            alert('删除成功');
            loadDevices();
        } else {
            alert('删除失败: ' + result.message);
        }
    } catch (error) {
        alert('删除失败: ' + error.message);
    }
}

// 加载历史记录
async function loadHistory() {
    loadDiagnosisHistory();
    loadRULHistory();
}

// 加载诊断历史
async function loadDiagnosisHistory() {
    try {
        const response = await fetch(`${API_BASE}/api/diagnosis_history`);
        const result = await response.json();
        
        if (result.success) {
            const tbody = document.getElementById('diagnosis-history-list');
            tbody.innerHTML = '';
            
            result.diagnoses.reverse().forEach(record => {
                const row = `
                    <tr>
                        <td>${record.time}</td>
                        <td>${record.model_path.split('/').pop()}</td>
                        <td>${record.data_path.split('/').pop()}</td>
                        <td><strong>${record.result}</strong></td>
                    </tr>
                `;
                tbody.innerHTML += row;
            });
        }
    } catch (error) {
        console.error('加载诊断历史失败:', error);
    }
}

// 加载RUL历史
async function loadRULHistory() {
    try {
        const response = await fetch(`${API_BASE}/api/rul_history`);
        const result = await response.json();
        
        if (result.success) {
            const tbody = document.getElementById('rul-history-list');
            tbody.innerHTML = '';
            
            result.predictions.reverse().forEach(record => {
                const row = `
                    <tr>
                        <td>${record.time}</td>
                        <td>${record.model_path.split('/').pop()}</td>
                        <td>${record.data_path.split('/').pop()}</td>
                        <td>${record.rul.toFixed(1)} 小时</td>
                        <td>${record.confidence ? (record.confidence * 100).toFixed(1) + '%' : 'N/A'}</td>
                        <td><strong>${record.status || 'N/A'}</strong></td>
                    </tr>
                `;
                tbody.innerHTML += row;
            });
        }
    } catch (error) {
        console.error('加载RUL历史失败:', error);
    }
}

// 切换历史标签
function showHistoryTab(tab) {
    const tabs = document.querySelectorAll('.tab-btn');
    tabs.forEach(t => t.classList.remove('active'));
    event.target.closest('.tab-btn').classList.add('active');
    
    const contents = document.querySelectorAll('.history-content');
    contents.forEach(c => c.classList.remove('active'));
    document.getElementById(`${tab}-history`).classList.add('active');
}

// 工具函数：格式化文件大小
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// ==================== 用户管理功能 ====================
function toggleUserMenu() {
    const menu = document.getElementById('user-menu');
    menu.classList.toggle('hidden');
}

// 点击页面其他地方时自动关闭菜单
document.addEventListener('click', function(e) {
    const userName = document.getElementById('user-name');
    const userMenu = document.getElementById('user-menu');
    if (!userMenu.contains(e.target) && e.target !== userName) {
        userMenu.classList.add('hidden');
    }
});

// 显示修改密码对话框
function showChangePasswordDialog() {
    document.getElementById('dialog-overlay').style.display = 'block';
    document.getElementById('change-password-dialog').style.display = 'block';
    // 清空输入框
    document.getElementById('old-password').value = '';
    document.getElementById('new-password').value = '';
    document.getElementById('confirm-password').value = '';
    document.getElementById('password-message').style.display = 'none';
}

// 修改密码
async function changePassword() {
    const oldPassword = document.getElementById('old-password').value;
    const newPassword = document.getElementById('new-password').value;
    const confirmPassword = document.getElementById('confirm-password').value;
    const messageDiv = document.getElementById('password-message');
    
    // 验证输入
    if (!oldPassword || !newPassword || !confirmPassword) {
        showPasswordMessage('请填写所有字段', 'error');
        return;
    }
    
    if (newPassword.length < 6) {
        showPasswordMessage('新密码长度不能少于6位', 'error');
        return;
    }
    
    if (newPassword !== confirmPassword) {
        showPasswordMessage('两次输入的新密码不一致', 'error');
        return;
    }
    
    try {
        const response = await fetch('/api/change_password', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            credentials: 'include',
            body: JSON.stringify({
                old_password: oldPassword,
                new_password: newPassword
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showPasswordMessage('密码修改成功！', 'success');
            // 2秒后关闭对话框
            setTimeout(() => {
                closeDialog();
            }, 2000);
        } else {
            showPasswordMessage(data.message || '修改失败', 'error');
        }
    } catch (error) {
        console.error('修改密码失败:', error);
        showPasswordMessage('网络错误，请重试', 'error');
    }
}

// 显示密码修改消息
function showPasswordMessage(text, type) {
    const messageDiv = document.getElementById('password-message');
    messageDiv.textContent = text;
    messageDiv.className = `message ${type}`;
    messageDiv.style.display = 'block';
}

// 登出
async function logout() {
    if (!confirm('确定要退出登录吗？')) {
        return;
    }
    
    try {
        const response = await fetch('/api/logout', {
            method: 'POST',
            credentials: 'include'
        });
        
        const data = await response.json();
        
        if (data.success) {
            // 跳转到登录页
            window.location.href = '/';
        } else {
            alert('登出失败，请重试');
        }
    } catch (error) {
        console.error('登出失败:', error);
        alert('网络错误，请重试');
    }
}
