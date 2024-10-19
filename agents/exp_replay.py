import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from utils.utils import maybe_cuda, AverageMeter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class ExperienceReplay(ContinualLearner):
    def __init__(self, model, opt, params):
        super(ExperienceReplay, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters
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

        # setup tracker
        losses_batch = AverageMeter()
        losses_mem = AverageMeter()
        acc_batch = AverageMeter()
        acc_mem = AverageMeter()
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

        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                for j in range(self.mem_iters):
                    logits = self.model.forward(batch_x)
                    loss = self.criterion(logits, batch_y)
                    if self.params.trick['kd_trick']:
                        loss = 1 / (self.task_seen + 1) * loss + (1 - 1 / (self.task_seen + 1)) * \
                                   self.kd_manager.get_kd_loss(logits, batch_x)
                    if self.params.trick['kd_trick_star']:
                        loss = 1/((self.task_seen + 1) ** 0.5) * loss + \
                               (1 - 1/((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(logits, batch_x)
                    _, pred_label = torch.max(logits, 1)
                    correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)
                    # update tracker
                    acc_batch.update(correct_cnt, batch_y.size(0))
                    losses_batch.update(loss, batch_y.size(0))
                    # backward
                    self.opt.zero_grad()
                    loss.backward()

                    # mem update
                    mem_x, mem_y = self.buffer.retrieve(x=batch_x, y=batch_y)
                    if mem_x.size(0) > 0:
                        mem_x = maybe_cuda(mem_x, self.cuda)
                        mem_y = maybe_cuda(mem_y, self.cuda)
                        mem_logits = self.model.forward(mem_x)
                        loss_mem = self.criterion(mem_logits, mem_y)
                        if self.params.trick['kd_trick']:
                            loss_mem = 1 / (self.task_seen + 1) * loss_mem + (1 - 1 / (self.task_seen + 1)) * \
                                       self.kd_manager.get_kd_loss(mem_logits, mem_x)
                        if self.params.trick['kd_trick_star']:
                            loss_mem = 1 / ((self.task_seen + 1) ** 0.5) * loss_mem + \
                                   (1 - 1 / ((self.task_seen + 1) ** 0.5)) * self.kd_manager.get_kd_loss(mem_logits,
                                                                                                         mem_x)
                        # update tracker
                        losses_mem.update(loss_mem, mem_y.size(0))
                        _, pred_label = torch.max(mem_logits, 1)
                        correct_cnt = (pred_label == mem_y).sum().item() / mem_y.size(0)
                        acc_mem.update(correct_cnt, mem_y.size(0))

                        loss_mem.backward()

                    if self.params.update == 'ASER' or self.params.retrieve == 'ASER':
                        # opt update
                        self.opt.zero_grad()
                        combined_batch = torch.cat((mem_x, batch_x))
                        combined_labels = torch.cat((mem_y, batch_y))
                        combined_logits = self.model.forward(combined_batch)
                        loss_combined = self.criterion(combined_logits, combined_labels)
                        loss_combined.backward()
                        self.opt.step()
                    else:
                        self.opt.step()

                # update mem
                self.buffer.update(batch_x, batch_y)

                if i % 100 == 1 and self.verbose:
                    print(
                        '==>>> it: {}, avg. loss: {:.6f}, '
                        'running train acc: {:.3f}'
                            .format(i, losses_batch.avg(), acc_batch.avg())
                    )
                    print(
                        '==>>> it: {}, mem avg. loss: {:.6f}, '
                        'running mem acc: {:.3f}'
                            .format(i, losses_mem.avg(), acc_mem.avg())
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
            plt.savefig(f'recencybiasER {self.k + 1}.pdf')
            plt.close()

            # 任务计数器递增，并确保不超出范围
            self.k += 1

        self.after_train()