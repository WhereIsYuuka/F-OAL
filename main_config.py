# 导入必要的库和模块
import argparse  # 用于解析命令行参数
from utils.io import load_yaml  # 从本地工具导入YAML文件加载器
from types import SimpleNamespace  # 用于创建简单命名空间对象
from utils.utils import boolean_string  # 从本地工具导入布尔字符串转换器
import time  # 用于时间相关操作
import torch  # PyTorch深度学习框架
import random  # 随机数生成模块
import numpy as np  # 数值计算库
from experiment.run import multiple_run  # 从实验模块导入多次运行函数


def main(args):
    """主函数，执行实验的主要流程"""

    # 1. 加载配置文件参数
    # 从YAML文件加载通用参数
    genereal_params = load_yaml(args.general)
    # 从YAML文件加载数据相关参数
    data_params = load_yaml(args.data)
    # 从YAML文件加载智能体相关参数
    agent_params = load_yaml(args.agent)

    # 2. 更新运行时参数
    # 设置是否显示详细信息
    genereal_params['verbose'] = args.verbose
    # 检测是否可用CUDA并设置
    genereal_params['cuda'] = torch.cuda.is_available()

    # 3. 合并所有参数到简单命名空间对象
    # 将三类参数合并为一个简单命名空间对象
    final_params = SimpleNamespace(**genereal_params, **data_params, **agent_params)

    # 记录开始时间
    time_start = time.time()
    # 打印最终参数配置
    print(final_params)

    # 4. 设置随机种子保证实验可重复性
    # 设置numpy随机种子
    np.random.seed(final_params.seed)
    # 设置python随机种子
    random.seed(final_params.seed)
    # 设置PyTorch随机种子
    torch.manual_seed(final_params.seed)

    # 如果使用CUDA，设置相关随机种子和配置
    if final_params.cuda:
        # 设置CUDA随机种子
        torch.cuda.manual_seed(final_params.seed)
        # 确保CUDA操作确定性
        torch.backends.cudnn.deterministic = True
        # 关闭CUDA基准优化
        torch.backends.cudnn.benchmark = False

    # 5. 运行实验
    # 调用多次运行函数执行实验
    multiple_run(final_params)


if __name__ == "__main__":
    # 命令行参数解析器
    parser = argparse.ArgumentParser('CVPR Continual Learning Challenge')

    # 添加命令行参数定义
    # 通用配置文件路径，默认为'config/general.yml'
    parser.add_argument('--general', dest='general', default='config/general.yml')
    # 数据配置文件路径，默认为'config/data/cifar100/cifar100_nc.yml'
    parser.add_argument('--data', dest='data', default='config/data/cifar100/cifar100_nc.yml')
    # 智能体配置文件路径，默认为'config/agent/er.yml'
    parser.add_argument('--agent', dest='agent', default='config/agent/er.yml')
    # 是否显示详细信息，默认为True
    parser.add_argument('--verbose', type=boolean_string, default=True,
                        help='是否打印信息（True/False）')

    # 解析命令行参数
    args = parser.parse_args()
    # 调用主函数
    main(args)