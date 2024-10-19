from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from torch.utils import data
from utils.utils import maybe_cuda, AverageMeter
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import copy


class Lwf(ContinualLearner):
    def __init__(self, model, opt, params):
        super(Lwf, self).__init__(model, opt, params)
        self.k = 0
        self.done = []

    def train_learner(self, x_train, y_train):
        self.before_train(x_train, y_train)

        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)

        # set up model
        self.model = self.model.train()
        tasks = [
            [22, 20, 25, 4],
            [10, 15, 28, 11],
            [18, 29, 27, 35],
            [37, 2, 39, 30],
            [34, 16, 36, 8],
            [13, 5, 17, 14],
            [33, 7, 32, 1],
            [26, 12, 31, 24],
            [6, 23, 21, 19],
            [9, 38, 3, 0]
        ]

        # setup tracker
        losses_batch = AverageMeter()
        acc_batch = AverageMeter()

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)

                logits = self.forward(batch_x)
                loss_old = self.kd_manager.get_kd_loss(logits, batch_x)
                loss_new = self.criterion(logits, batch_y)
                loss = 1/(self.task_seen + 1) * loss_new + (1 - 1/(self.task_seen + 1)) * loss_old
                _, pred_label = torch.max(logits, 1)
                correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)
                # update tracker
                acc_batch.update(correct_cnt, batch_y.size(0))
                losses_batch.update(loss, batch_y.size(0))
                # backward
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # 根据当前任务着色并绘制柱状图


                if i % 100 == 1 and self.verbose:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                        'running train acc: {:.3f}'
                            .format(i, losses_batch.avg(), acc_batch.avg())
                    )
            weights = self.model.fc.weight.data
            weights = weights.cpu()
            row_norms = torch.norm(weights, p=2, dim=1)

            # 根据当前任务着色并绘制柱状图
            colors = []

            for i in range(weights.size(0)):
                if i in tasks[self.k]:
                    colors.append('red')  # 当前任务的类别'
                    self.done.append(i)
                elif i in self.done:
                    colors.append('blue')  # 已遍历过的任务的类别
                else:
                    colors.append('yellow')  # 未遍历的任务的类别

            plt.rc('axes', labelsize=20)  # Axes labels size
            plt.bar(range(weights.size(0)), row_norms.numpy(), color=colors)
            plt.xlabel(f'Classes of DTD at Task {self.k + 1}', labelpad=20)
            plt.ylabel('L2 Norm of Weights')
            red_patch = mpatches.Patch(color='red', label='Current Task')
            blue_patch = mpatches.Patch(color='blue', label='Completed Tasks')
            yellow_patch = mpatches.Patch(color='yellow', label='Future Tasks')
            plt.subplots_adjust(bottom=0.2)
            plt.legend(handles=[red_patch, blue_patch, yellow_patch], loc='upper center',
                       bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)
            plt.savefig(f'recencybiasLWF{self.k + 1}.pdf')
            plt.close()

            # 任务计数器递增，并确保不超出范围
            self.k += 1
        self.after_train()
