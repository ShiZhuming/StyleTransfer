import configparser
file = 'config.ini'

# 创建配置文件对象
con = configparser.ConfigParser()

# 读取文件
con.read(file, encoding='utf-8')

# 模型参数
modelpath = dict(con.items('path'))

# 备案信息
record = dict(con.items('record'))

# if __name__ == '__main__':
#     print(modelpath['decoder'],record)