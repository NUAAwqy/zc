# bearingPlatform_hua
西储大学轴承数据集故障诊断的仿真平台

## ⚠️ PyTorch版本说明
**本项目已从 Keras/TensorFlow 1.x 迁移到 PyTorch！**

📖 **详细运行说明请查看**: [运行说明_PyTorch版本.md](./运行说明_PyTorch版本.md)

⚡ **遇到TensorFlow/Keras错误?**: [快速修复指南.md](./快速修复指南.md)

### 主要改进
- ✅ 使用 **PyTorch** 替代 Keras/TensorFlow
- ✅ 更好的 GPU 支持
- ✅ 模型保存为 `.pth` 格式（而非 `.h5`）
- ✅ 改进的训练流程和性能

### 快速开始

#### GUI版本（本地运行）
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 安装PyTorch（访问 https://pytorch.org/ 选择适合的版本）
pip install torch torchvision torchaudio

# 3. 运行GUI程序
python main.py
```

#### 无GUI版本（服务器运行）⭐
```bash
# 训练模型
python train_headless.py --model 1D_CNN --data_path /path/to/data

# 诊断数据
python diagnose_headless.py --model ./models/1D_CNN.pth --data /path/to/test.mat

# 详细说明见：服务器运行指南.md
```

---

## 1.简介
因为学习需要，因而简单学习了一下PySide2的使用，并粗暴的开发一款简单的故障诊断仿真平台（真的是简单粗暴的一个平台￣□￣｜｜)
），该平台使用[西储大学轴承数据集](https://www.cnblogs.com/gshang/p/10712809.html)实现了对轴承的故障诊断。平台主要功能： 

* 选择不同算法训练模型
* 使用保存的模型进行故障诊断

## 2.环境
* Windows 10 / Linux / macOS
* Python 3.6+ (推荐 3.8-3.10)
* PyTorch 1.10+
* Anaconda (推荐)
## 3. 文件说明
UI: 存放的软件平台页面布局文件

data_preprocess.py: 数据预处理

diagnosis.py: 故障诊断相关函数

feature_extraction.py: 特征提取函数

figure_canvas.py: GraphicsView控件中利用matplotlib绘图的相关函数

main.py: 主程序

message_signal.py: 自定义信号

preprocess_train_result.py: 处理模性训练结果的相关函数

training_model.py 模型训练的相关函数
## 4. 故障分类算法
算法可以对**0马力，采样频率为48KHZ**的轴承的9类故障以及正常状态进行分类，这9类故障分别为：
* 滚动体故障：0.1778mm
* 滚动体故障：0.3556mm
* 滚动体故障：0.5334mm
* 内圈故障：0.1778mm
* 内圈故障：0.3556mm
* 内圈故障：0.5334mm
* 外圈故障（6点方向）：0.1778mm
* 外圈故障（6点方向）：0.3556mm
* 外圈故障（6点方向）：0.5334mm

平台中一共使用了4种不同的算法来进行故障诊断，这4种算法分别为：
* 1D_CNN
* LSTM
* GRU
* 随机森林

对于故障诊断的算法以及数据的处理，参考了[Jiali Zhang](https://github.com/zhangjiali1201/keras_bearing_fault_diagnosis)的代码。

*对于整体的算法可能并不是很完美，欢迎大家一起讨论改善*

## 5.效果图

<img src="img/diagnosis_page.jpg" alt="故障诊断页面" style="zoom: 67%;" />

<img src="img/train_model_page.jpg" alt="训练模型页面" style="zoom: 67%;" />

## 6.说明
1. 对于数据可视化的图片显示，参照[Pyside2中嵌入Matplotlib的绘图](https://blog.csdn.net/qq_28053421/article/details/113828372?spm=1001.2014.3001.5501)，使用了  GraphicsView控件嵌入Matplotlib的绘图，
而在训练结果展示的图片显示中，由于绘图使用了sklearn中封装的函数，所以目前是将其先存到本地，然后再读取显示。
2. 为保证在使用模型进行 本地诊断/实时诊断 时的准确性，需要对诊断数据利用与模型训练时相同的数据标准化尺度进行标准化处理，
因此在保存训练模型时，会同时保存一个配置文件（JSON文件），以记录标准化的相关信息。
同时，为保证配置文件和模型的匹配性，会同时记录模型文件的md5值，以便在加载模型时校验。
3. **PyTorch版本改进**：
   - 神经网络模型（CNN/LSTM/GRU）使用PyTorch实现
   - 模型保存为`.pth`格式，配置文件为`.json`格式
   - 支持GPU自动检测和加速
   - 训练过程实时显示loss和accuracy

## 7. 模型文件格式
- **PyTorch模型**: `.pth` 或 `.pt`
- **随机森林模型**: `.m` (joblib格式)
- **配置文件**: `.json` (必须与模型同名，存放在同一目录)

## 8. 服务器/无GUI环境运行

如果您在服务器（AutoDL、云服务器等）上遇到GUI错误，请使用无GUI版本：

📖 **详细指南**: [服务器运行指南.md](./服务器运行指南.md)

### 快速命令

```bash
# 训练模型（无GUI）
python train_headless.py --model 1D_CNN --data_path /path/to/data --save_dir ./models

# 诊断单个文件
python diagnose_headless.py --model ./models/1D_CNN.pth --data sample.mat

# 批量诊断
python diagnose_headless.py --model ./models/1D_CNN.pth --data ./test_data --batch --output results.csv
```

### 新增脚本说明

- **train_headless.py**: 无GUI训练脚本，支持命令行参数
- **diagnose_headless.py**: 无GUI诊断脚本，支持单个/批量诊断

## 9. TO DO List
1. 添加模型参数设置功能
2. 支持更多深度学习模型（Transformer、ResNet等）
3. 添加模型可视化功能
4. 优化训练速度和内存使用