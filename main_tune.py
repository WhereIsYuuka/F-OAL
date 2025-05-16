# 导入必要的库和模块
import argparse  # 用于解析命令行参数
from utils.io import load_yaml  # 从本地工具导入YAML配置文件加载器
from types import SimpleNamespace  # 用于创建简单命名空间对象
from utils.utils import boolean_string  # 从本地工具导入布尔字符串转换器
import time  # 用于时间相关操作
import torch  # PyTorch深度学习框架
import random  # 随机数生成模块
import numpy as np  # 数值计算库
from experiment.run import multiple_run_tune_separate  # 导入带调参的多次运行实验函数
from utils.setup_elements import default_trick  # 导入默认技巧配置


def main(args):
    """主函数，执行实验的主要流程"""

    # 1. 加载各类配置文件参数
    genereal_params = load_yaml(args.general)  # 加载通用配置参数
    data_params = load_yaml(args.data)  # 加载数据相关参数
    default_params = load_yaml(args.default)  # 加载默认模型参数
    tune_params = load_yaml(args.tune)  # 加载调参参数

    # 2. 更新运行时参数配置
    genereal_params['verbose'] = args.verbose  # 设置是否显示详细信息
    genereal_params['cuda'] = torch.cuda.is_available()  # 检测并设置CUDA可用性
    genereal_params['train_val'] = args.train_val  # 设置是否使用验证批次训练

    # 3. 处理特殊技巧参数
    if args.trick:  # 如果指定了特定技巧
        default_trick[args.trick] = True  # 激活对应的技巧
    genereal_params['trick'] = default_trick  # 将技巧配置加入参数

    # 4. 合并基础参数到命名空间对象
    final_default_params = SimpleNamespace(**genereal_params, **data_params, **default_params)

    # 记录开始时间并打印参数
    time_start = time.time()
    print(final_default_params)  # 打印基础参数配置
    print()  # 空行分隔

    # 5. 设置随机种子保证实验可重复性
    np.random.seed(final_default_params.seed)  # 设置numpy随机种子
    random.seed(final_default_params.seed)  # 设置python随机种子
    torch.manual_seed(final_default_params.seed)  # 设置PyTorch随机种子

    # 如果使用CUDA，设置相关配置
    if final_default_params.cuda:
        torch.cuda.manual_seed(final_default_params.seed)  # 设置CUDA随机种子
        torch.backends.cudnn.deterministic = True  # 确保CUDA操作确定性
        torch.backends.cudnn.benchmark = False  # 关闭CUDA基准优化

    # 6. 运行实验（带参数调优的分离运行）
    multiple_run_tune_separate(
        final_default_params,  # 基础参数配置
        tune_params,  # 调参参数配置
        args.save_path  # 结果保存路径
    )


if __name__ == "__main__":
    # 命令行参数解析器
    parser = argparse.ArgumentParser('Continual Learning')

    # 添加配置文件路径参数
    parser.add_argument('--general', dest='general',
                        default='config/general_1.yml',
                        help='通用配置文件路径')
    parser.add_argument('--data', dest='data',
                        default='config/data/cifar100/cifar100_nc.yml',
                        help='数据配置文件路径')
    parser.add_argument('--default', dest='default',
                        default='config/agent/er/er_1k.yml',
                        help='默认模型配置文件路径')
    parser.add_argument('--tune', dest='tune',
                        default='config/agent/er/er_tune.yml',
                        help='调参配置文件路径')

    # 添加运行控制参数
    parser.add_argument('--save-path', dest='save_path',
                        default=None,
                        help='结果保存路径')
    parser.add_argument('--verbose', type=boolean_string,
                        default=False,
                        help='是否打印详细信息(True/False)')
    parser.add_argument('--train_val', type=boolean_string,
                        default=False,
                        help='是否使用验证批次训练(True/False)')
    parser.add_argument('--trick', type=str,
                        default=None,
                        help='指定要使用的特殊技巧名称')

    # 解析命令行参数
    args = parser.parse_args()
    # 调用主函数
    main(args)