from mysql_auth import init_database

if __name__ == '__main__':
    result = init_database()
    print(f"数据库初始化结果: {result}")