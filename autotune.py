import subprocess
from itertools import product
import re

# 定义超参数范围
PARAMETER_SPACE = {
    "BM": [32, 64, 128],
    "BN": [32, 64, 128],
    "BK": [4, 8, 16],
    "TM": [4, 8, 16],
    "TN": [4, 8, 16],
}

best_time = float("inf")
best_parameters = {}


# 自定义的合理性检查逻辑
def check_parameters(parameters):
    BM, BN, BK, TM, TN = (
        parameters["BM"],
        parameters["BN"],
        parameters["BK"],
        parameters["TM"],
        parameters["TN"],
    )
    if BM != BN:
        return False
    if TM != TN:
        return False

    if BM % TM != 0:
        return False
    if BN % TN != 0:
        return False

    if BN < TN:
        return False
    if BM < TM:
        return False

    BLOCKSIZE = (BM / TM) * (BN / TN)
    if BLOCKSIZE > 1024 or BLOCKSIZE < 16:
        return False

    return True


def compile_and_run():
    compile_command = "nvcc -o gemm temp.cu"
    subprocess.run(compile_command, shell=True, check=True)

    run_command = "./gemm"  # 请替换为您的 CUDA 程序执行命令
    try:
        result = subprocess.run(
            run_command,
            capture_output=True,
            text=True,
            shell=True,
            check=True,
            timeout=2,
        )
    except subprocess.TimeoutExpired:
        # 如果超时，则杀死正在运行的程序并返回一个“无穷大”的时间
        subprocess.run(f"pkill -f '{run_command}'", shell=True)
        return float("inf")
    except subprocess.CalledProcessError:
        return float("inf")

    if result.returncode != 0:
        return float("inf")
    # 从程序输出中提取执行时间
    return float(result.stdout.split("ms")[0])


for parameters in product(*[PARAMETER_SPACE[param] for param in PARAMETER_SPACE]):
    parameters = {key: value for key, value in zip(PARAMETER_SPACE.keys(), parameters)}
    if not check_parameters(parameters):
        continue

    with open("kernel.cu", "r") as file:
        source_code = file.read()

    for key, value in parameters.items():
        source_code = re.sub(
            rf"#define {key} \d+", f"#define {key} {value}", source_code
        )

    # 将修改后的CUDA源代码写入临时文件
    with open("temp.cu", "w") as file:
        file.write(source_code)

    # 运行CUDA程序并捕获执行时间
    execution_time = compile_and_run()
    # 打印当前超参数组合的执行时间
    print(f"Parameters: {parameters} \t Execution Time: {execution_time} ms")

    # 更新最佳超参数和最佳执行时间
    if execution_time < best_time:
        best_time = execution_time
        best_parameters = parameters

print("Best Parameters:", best_parameters)
print("Best Execution Time:", best_time, "ms")
