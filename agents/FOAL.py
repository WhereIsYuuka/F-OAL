from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match
from torch.utils import data
from utils.utils import maybe_cuda, AverageMeter
import torch.nn.functional as F
import torch
import torch.nn.init as init


class FOAL(ContinualLearner):
    def __init__(self, model, opt, params):
        super(FOAL, self).__init__(model, opt, params)
        self.R = ((torch.eye(params.projection)).float()).cuda()
        self.W = (init.zeros_(self.model.fc.weight.t())).double().cuda()
        self.R = self.R.double()


    def train_learner(self, x_train, y_train):
        self.before_train(x_train, y_train)
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0, drop_last=True)
        losses_batch = AverageMeter()
        acc_batch = AverageMeter()
        self.model.eval()

        with torch.no_grad():
            for ep in range(self.epoch):
                for i, batch_data in enumerate(train_loader):
                    # batch update
                    batch_x, batch_y = batch_data

                    batch_x = maybe_cuda(batch_x, self.cuda)
                    batch_y = maybe_cuda(batch_y, self.cuda)
                    new_activation = self.model.expansion(batch_x)
                    new_activation = new_activation.double()

                    label_onehot = F.one_hot(batch_y,self.model.numclass)
                    self.R = self.R - self.R @ new_activation.t() @ torch.pinverse(
                        torch.eye(new_activation.size(0)).cuda(non_blocking=True) +
                        new_activation @ self.R @ new_activation.t()) @ new_activation @ self.R
                    self.W = self.W + self.R @ new_activation.t() @ (label_onehot - new_activation @ self.W)
                    self.model.fc.weight = torch.nn.parameter.Parameter(torch.t(self.W.float()))
                    logits = self.model(batch_x)
                    _, pred_label = torch.max(logits, 1)
                    correct_cnt = (pred_label == batch_y).sum().item() / batch_y.size(0)

                    # update tracker
                    acc_batch.update(correct_cnt, batch_y.size(0))
                    if i % 100 == 1 and self.verbose:
                        print(
                            '==>>> it: {}, avg. loss: {:.6f}, '
                            'running train acc: {:.3f}'
                                .format(i, losses_batch.avg(), acc_batch.avg())
                        )
            self.after_train()

