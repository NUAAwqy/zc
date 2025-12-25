"""


@Desc  : 工具函数 - PyTorch版本
"""
import hashlib


def generate_md5(file_path):
    """
    生成文件的MD5校验值
    
    参数:
        file_path: 文件路径
    
    返回:
        MD5字符串
    """
    with open(file_path, 'rb') as f:
        content = f.read()

    md5hash = hashlib.md5(content)
    md5 = md5hash.hexdigest()
    return md5
