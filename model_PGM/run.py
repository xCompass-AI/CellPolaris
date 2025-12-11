import subprocess
import logging
import concurrent.futures


logging.basicConfig(filename='jingzi_log_13.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


params=[['./data/generated_grn/RS5o6_generated_grn.txt', './data/RS5o6.txt', './data_sample/RS5o6_sample.csv', 'RS5o6'], ['./data/generated_grn/RS1o2_generated_grn.txt', './data/RS1o2.txt', './data_sample/RS1o2_sample.csv', 'RS1o2'], ['./data/generated_grn/RS3o4_generated_grn.txt', './data/RS3o4.txt', './data_sample/RS3o4_sample.csv', 'RS3o4'], ['./data/generated_grn/RS7o8_generated_grn.txt', './data/RS7o8.txt', './data_sample/RS7o8_sample.csv', 'RS7o8']]

output_folder = './resultS/'
edge_num=3000


def execute_command(param):  
    cmd = ["python", "model_PGM.py", param[0], param[1], param[2],param[3],output_folder,str(edge_num)]   
    log_file = f"log_{param[3]}_jingzi13.txt"
    with open(log_file, "a") as f:
        logging.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, universal_newlines=True)
        if result.returncode != 0:
            logging.error(f"Command error: {result.stderr}")


executor = concurrent.futures.ProcessPoolExecutor()

futures = [executor.submit(execute_command, param) for param in params]

concurrent.futures.wait(futures)

executor.shutdown()
