#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MySQL用户认证模块
用于用户名和密码的验证与管理
"""

import pymysql
import os

# MySQL数据库配置
MYSQL_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '1234',
    'database': 'bearing_diagnosis',
    'charset': 'utf8mb4'
}


def get_db_connection():
    """获取数据库连接"""
    return pymysql.connect(**MYSQL_CONFIG)


def init_database():
    """初始化数据库和用户表"""
    try:
        # 首先连接到MySQL服务器（不指定数据库）
        conn = pymysql.connect(
            host=MYSQL_CONFIG['host'],
            port=MYSQL_CONFIG['port'],
            user=MYSQL_CONFIG['user'],
            password=MYSQL_CONFIG['password'],
            charset=MYSQL_CONFIG['charset']
        )
        cursor = conn.cursor()
        
        # 创建数据库（如果不存在）
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_CONFIG['database']} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        cursor.close()
        conn.close()
        
        # 连接到指定数据库
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 创建用户表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL
            )
        """)
        
        # 检查是否有用户，如果没有则创建默认管理员账户
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        if user_count == 0:
            # 创建默认管理员账户: admin / admin123
            cursor.execute(
                "INSERT INTO users (username, password) VALUES (%s, %s)",
                ('admin', 'admin123')
            )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("数据库初始化成功！")
        return True
        
    except Exception as e:
        print(f"数据库初始化失败: {str(e)}")
        return False


def verify_user(username, password):
    """
    验证用户名和密码
    
    Args:
        username: 用户名
        password: 密码
        
    Returns:
        bool: 验证成功返回True，否则返回False
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 查询用户
        cursor.execute(
            "SELECT password FROM users WHERE username = %s",
            (username,)
        )
        result = cursor.fetchone()
        print(f"[login debug] username={username}, db_password={result[0] if result else None}")
        
        if result:
            stored_password = result[0]
            if stored_password == password:
                cursor.close()
                conn.close()
                return True
        
        cursor.close()
        conn.close()
        return False
        
    except Exception as e:
        print(f"验证用户失败: {str(e)}")
        return False


def add_user(username, password):
    """
    添加新用户
    
    Args:
        username: 用户名
        password: 密码
        
    Returns:
        tuple: (成功标志, 消息)
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 检查用户是否已存在
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return False, "用户名已存在"
        
        # 添加用户（直接存储明文密码）
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (%s, %s)",
            (username, password)
        )
        conn.commit()
        cursor.close()
        conn.close()
        
        return True, "用户添加成功"
        
    except Exception as e:
        return False, f"添加用户失败: {str(e)}"


def change_password(username, old_password, new_password):
    """
    修改用户密码
    
    Args:
        username: 用户名
        old_password: 旧密码
        new_password: 新密码
        
    Returns:
        tuple: (成功标志, 消息)
    """
    try:
        # 先验证旧密码
        if not verify_user(username, old_password):
            return False, "旧密码不正确"
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 更新密码
        cursor.execute(
            "UPDATE users SET password = %s WHERE username = %s",
            (new_password, username)
        )
        conn.commit()
        cursor.close()
        conn.close()
        
        return True, "密码修改成功"
        
    except Exception as e:
        return False, f"修改密码失败: {str(e)}"


def delete_user(username):
    """
    删除用户
    
    Args:
        username: 用户名
        
    Returns:
        tuple: (成功标志, 消息)
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 检查是否是最后一个用户
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        if user_count <= 1:
            cursor.close()
            conn.close()
            return False, "不能删除最后一个用户"
        
        # 删除用户
        cursor.execute("DELETE FROM users WHERE username = %s", (username,))
        conn.commit()
        
        if cursor.rowcount > 0:
            cursor.close()
            conn.close()
            return True, "用户删除成功"
        else:
            cursor.close()
            conn.close()
            return False, "用户不存在"
        
    except Exception as e:
        return False, f"删除用户失败: {str(e)}"


def list_users():
    """
    列出所有用户
    
    Returns:
        list: 用户列表
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        cursor.execute("""
            SELECT id, username 
            FROM users 
            ORDER BY id DESC
        """)
        users = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return users
        
    except Exception as e:
        print(f"获取用户列表失败: {str(e)}")
        return []
