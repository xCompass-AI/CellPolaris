import subprocess
import logging
import concurrent.futures

# 配置日志记录
logging.basicConfig(filename='jingzi_log_13.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


params=[['./data/generated_grn/RS5o6_generated_grn.txt', './data/RS5o6.txt', './data_sample/RS5o6_sample.csv', 'RS5o6'], ['./data/generated_grn/RS1o2_generated_grn.txt', './data/RS1o2.txt', './data_sample/RS1o2_sample.csv', 'RS1o2'], ['./data/generated_grn/RS3o4_generated_grn.txt', './data/RS3o4.txt', './data_sample/RS3o4_sample.csv', 'RS3o4'], ['./data/generated_grn/RS7o8_generated_grn.txt', './data/RS7o8.txt', './data_sample/RS7o8_sample.csv', 'RS7o8']]

output_folder = './resultS/'
edge_num=3000


# 执行命令行的函数
def execute_command(param):
    # 构建命令行参数
    cmd = ["python", "model_PGM.py", param[0], param[1], param[2],param[3],output_folder,str(edge_num)]
    # 定义日志文件名
    log_file = f"log_{param[3]}_jingzi13.txt"
    # 打开日志文件以追加模式写入
    with open(log_file, "a") as f:
        # 记录程序运行的信息
        logging.info(f"Running command: {' '.join(cmd)}")
        # 执行命令行，并将标准输出和标准错误输出重定向到日志文件
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, universal_newlines=True)
        # 检查命令行的错误输出
        if result.returncode != 0:
            logging.error(f"Command error: {result.stderr}")

# 创建进程池
executor = concurrent.futures.ProcessPoolExecutor()

# 提交命令给进程池进行并行执行
futures = [executor.submit(execute_command, param) for param in params]

# 等待所有命令执行完成
concurrent.futures.wait(futures)

# 关闭进程池
executor.shutdown()
