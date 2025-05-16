# 导入必要的库
import argparse  # 用于解析命令行参数
import random  # 用于随机数生成
import numpy as np  # 用于数值计算
import torch  # PyTorch深度学习框架
from experiment.run import multiple_run  # 从本地模块导入多次运行实验的函数
from utils.utils import boolean_string  # 从本地工具模块导入布尔字符串转换函数
import time  # 用于时间相关操作


def main(args):
    """主函数，执行实验的主要逻辑"""
    print(args)  # 打印输入参数

    # 设置随机种子以确保实验可重复性
    # args.seed = int(time.time())  # 可以使用当前时间作为随机种子（注释状态）
    np.random.seed(args.seed)  # 设置numpy的随机种子
    random.seed(args.seed)  # 设置python随机模块的种子
    torch.manual_seed(args.seed)  # 设置PyTorch的随机种子

    # 如果使用CUDA（GPU加速），设置相关的随机种子和配置
    if args.cuda:
        torch.cuda.manual_seed(args.seed)  # 设置CUDA的随机种子
        torch.backends.cudnn.deterministic = True  # 确保CUDA卷积操作确定性
        torch.backends.cudnn.benchmark = False  # 关闭CUDA基准优化

    # 将各种技巧参数整理到字典中
    args.trick = {
        'labels_trick': args.labels_trick,  # 标签技巧
        'separated_softmax': args.separated_softmax,  # 分离的softmax
        'kd_trick': args.kd_trick,  # 知识蒸馏技巧
        'kd_trick_star': args.kd_trick_star,  # 改进版知识蒸馏技巧
        'review_trick': args.review_trick,  # 复习技巧
        'ncm_trick': args.ncm_trick  # 最近类均值分类器技巧
    }

    # 调用多次运行函数执行实验
    multiple_run(args, store=args.store, save_path=args.save_path)


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="Online Continual Learning PyTorch")

    ######################## 通用参数 #########################
    parser.add_argument('--num_runs', dest='num_runs', default=1, type=int,
                        help='运行次数 (default: %(default)s)')
    parser.add_argument('--seed', dest='seed', default=0, type=int,
                        help='随机种子')

    ######################## 杂项参数 #########################
    parser.add_argument('--val_size', dest='val_size', default=0.1, type=float,
                        help='验证集比例 (default: %(default)s)')
    parser.add_argument('--num_val', dest='num_val', default=3, type=int,
                        help='用于验证的批次数量 (default: %(default)s)')
    parser.add_argument('--num_runs_val', dest='num_runs_val', default=3, type=int,
                        help='验证的运行次数 (default: %(default)s)')
    parser.add_argument('--error_analysis', dest='error_analysis', default=False, type=boolean_string,
                        help='是否进行错误分析 (default: %(default)s)')
    parser.add_argument('--verbose', type=boolean_string, default=True,
                        help='是否打印信息 (default: %(default)s)')
    parser.add_argument('--store', type=boolean_string, default=False,
                        help='是否存储结果 (default: %(default)s)')
    parser.add_argument('--save-path', dest='save_path', default=None,
                        help='结果保存路径')

    ######################## 智能体参数 #########################
    parser.add_argument('--agent', dest='agent', default='ER',
                        choices=['ER', 'EWC', 'AGEM', 'CNDPM', 'LWF', 'ICARL', 'GDUMB', 'ASER', 'SCR', 'FOAL', 'PCR',
                                 'ER_DVC'],
                        help='选择哪种智能体 (default: %(default)s)')
    parser.add_argument('--update', dest='update', default='random', choices=['random', 'GSS', 'ASER'],
                        help='更新方法 (default: %(default)s)')
    parser.add_argument('--retrieve', dest='retrieve', default='random',
                        choices=['MIR', 'random', 'ASER', 'match', 'mem_match', 'MGI'],
                        help='检索方法 (default: %(default)s)')

    ######################## 优化器参数 #########################
    parser.add_argument('--optimizer', dest='optimizer', default='SGD', choices=['SGD', 'Adam'],
                        help='优化器选择 (default: %(default)s)')
    parser.add_argument('--learning_rate', dest='learning_rate', default=0.1, type=float,
                        help='学习率 (default: %(default)s)')
    parser.add_argument('--epoch', dest='epoch', default=1, type=int,
                        help='每个任务的训练周期数 (default: %(default)s)')
    parser.add_argument('--batch', dest='batch', default=10, type=int,
                        help='批次大小 (default: %(default)s)')
    parser.add_argument('--test_batch', dest='test_batch', default=10, type=int,
                        help='测试批次大小 (default: %(default)s)')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0,
                        help='权重衰减系数')

    ######################## 数据参数 #########################
    parser.add_argument('--num_tasks', dest='num_tasks', default=10, type=int,
                        help='任务数量 (default: %(default)s)')
    parser.add_argument('--fix_order', dest='fix_order', default=False, type=boolean_string,
                        help='在NC场景中，是否固定类别顺序 (default: %(default)s)')
    parser.add_argument('--plot_sample', dest='plot_sample', default=False, type=boolean_string,
                        help='在NI场景中，是否绘制样本图像 (default: %(default)s)')
    parser.add_argument('--data', dest='data', default="cifar10",
                        help='数据集路径 (default: %(default)s)')
    parser.add_argument('--cl_type', dest='cl_type', default="nc", choices=['nc', 'ni'],
                        help='持续学习类型: 新类别"nc"或新实例"ni" (default: %(default)s)')
    parser.add_argument('--ns_factor', dest='ns_factor', nargs='+',
                        default=(0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6), type=float,
                        help='非平稳数据的变化因子 (default: %(default)s)')
    parser.add_argument('--ns_type', dest='ns_type', default='noise', type=str, choices=['noise', 'occlusion', 'blur'],
                        help='非平稳类型 (default: %(default)s)')
    parser.add_argument('--ns_task', dest='ns_task', nargs='+', default=(1, 1, 2, 2, 2, 2), type=int,
                        help='NI非平稳任务组成 (default: %(default)s)')
    parser.add_argument('--online', dest='online', default=True, type=boolean_string,
                        help='如果为False，将执行离线训练 (default: %(default)s)')

    ######################## ER参数 #########################
    parser.add_argument('--mem_size', dest='mem_size', default=10000, type=int,
                        help='记忆缓冲区大小 (default: %(default)s)')
    parser.add_argument('--eps_mem_batch', dest='eps_mem_batch', default=10, type=int,
                        help='每批次的记忆片段数 (default: %(default)s)')

    ####################### ER_DVC参数 #########################
    parser.add_argument('--dl_weight', dest='dl_weight', default=4.0, type=float,
                        help='dl损失的权重')

    ######################## EWC参数 ##########################
    parser.add_argument('--lambda', dest='lambda_', default=100, type=float,
                        help='EWC正则化系数')
    parser.add_argument('--alpha', dest='alpha', default=0.9, type=float,
                        help='EWC++ Fisher计算的指数移动平均衰减系数')
    parser.add_argument('--fisher_update_after', dest='fisher_update_after', type=int, default=50,
                        help="训练迭代多少次后更新Fisher矩阵")

    ######################## MIR参数 #########################
    parser.add_argument('--subsample', dest='subsample', default=50, type=int,
                        help='执行MIR的子样本数量 (default: %(default)s)')

    ######################## GSS参数 #########################
    parser.add_argument('--gss_mem_strength', dest='gss_mem_strength', default=10, type=int,
                        help='从记忆中随机采样用于估计得分的批次数量')
    parser.add_argument('--gss_batch_size', dest='gss_batch_size', default=10, type=int,
                        help='用于估计得分的随机采样批次大小')

    ######################## ASER参数 ########################
    parser.add_argument('--k', dest='k', default=5, type=int,
                        help='执行ASER的最近邻数量(K) (default: %(default)s)')
    parser.add_argument('--aser_type', dest='aser_type', default="asvm", type=str,
                        choices=['neg_sv', 'asv', 'asvm'],
                        help='ASER类型: "neg_sv"-仅使用负SV, "asv"-使用对抗SV和合作SV的极值, "asvm"-使用对抗SV和合作SV的平均值')
    parser.add_argument('--n_smp_cls', dest='n_smp_cls', default=2.0, type=float,
                        help='随机采样每类的最大样本数 (default: %(default)s)')

    ######################## CNDPM参数 #########################
    parser.add_argument('--stm_capacity', dest='stm_capacity', default=1000, type=int,
                        help='短期记忆大小')
    parser.add_argument('--classifier_chill', dest='classifier_chill', default=0.01, type=float,
                        help='NDPM classifier_chill参数')
    parser.add_argument('--log_alpha', dest='log_alpha', default=-300, type=float,
                        help='先验log alpha')

    ######################## GDumb参数 #########################
    parser.add_argument('--minlr', dest='minlr', default=0.0005, type=float,
                        help='最小学习率')
    parser.add_argument('--clip', dest='clip', default=10., type=float,
                        help='梯度裁剪的值')
    parser.add_argument('--mem_epoch', dest='mem_epoch', default=70, type=int,
                        help='记忆训练的周期数')

    ######################## FOAL参数 #########################
    parser.add_argument('--projection', dest='projection', default=1000, type=int,
                        help='FOAL的线性投影大小')

    ####################### 技巧参数 #########################
    parser.add_argument('--labels_trick', dest='labels_trick', default=False, type=boolean_string,
                        help='是否使用标签技巧')
    parser.add_argument('--separated_softmax', dest='separated_softmax', default=False, type=boolean_string,
                        help='是否使用分离的softmax')
    parser.add_argument('--kd_trick', dest='kd_trick', default=False, type=boolean_string,
                        help='是否使用知识蒸馏交叉熵技巧')
    parser.add_argument('--kd_trick_star', dest='kd_trick_star', default=False, type=boolean_string,
                        help='是否使用改进版知识蒸馏技巧')
    parser.add_argument('--review_trick', dest='review_trick', default=False, type=boolean_string,
                        help='是否使用复习技巧')
    parser.add_argument('--ncm_trick', dest='ncm_trick', default=False, type=boolean_string,
                        help='是否使用最近类均值分类器')
    parser.add_argument('--mem_iters', dest='mem_iters', default=1, type=int,
                        help='记忆迭代次数')

    #################### 早停参数 ######################
    parser.add_argument('--min_delta', dest='min_delta', default=0., type=float,
                        help='认为有改进的最小分数增加量')
    parser.add_argument('--patience', dest='patience', default=0, type=int,
                        help='在停止训练前等待无改进的事件次数')
    parser.add_argument('--cumulative_delta', dest='cumulative_delta', default=False, type=boolean_string,
                        help='如果为True，`min_delta`定义自上次`patience`重置以来的增加量，否则定义上次事件后的增加量')

    #################### 监督对比学习参数 ######################
    parser.add_argument('--temp', type=float, default=0.07,
                        help='损失函数的温度参数')
    parser.add_argument('--buffer_tracker', type=boolean_string, default=False,
                        help='是否使用字典跟踪缓冲区')
    parser.add_argument('--warmup', type=int, default=4,
                        help='检索前缓冲区的预热次数')
    parser.add_argument('--head', type=str, default='mlp',
                        help='投影头类型')

    # 解析命令行参数
    args = parser.parse_args()
    # 检查是否可用CUDA
    args.cuda = torch.cuda.is_available()
    # 调用主函数
    main(args)