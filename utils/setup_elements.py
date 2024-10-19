import torch
from models.resnet import SupConResNet
from models.pretrained import ModifiedViT, Encoder,VIT_DVC
from torchvision import transforms



default_trick = {'labels_trick': False, 'kd_trick': False, 'separated_softmax': False,
                 'review_trick': False, 'ncm_trick': False, 'kd_trick_star': False}


input_size_match = {
    'cifar100': [3, 224, 224],
    'cifar10': [3, 32, 32],
    'core50': [3, 224, 224],
    'mini_imagenet': [3, 84, 84],
    'openloris': [3, 50, 50],
    'flower102': [3, 224, 224],
    'food101':[3,224,224],
    'DTD': [3,224,224],
    'aircraft': [3,224,224],
    'CelebA':[3,224,224],
    'Country211':[3,224,224],
    'StanfordCars':[3,224,224]
}


n_classes = {
    'cifar100': 100,
    'cifar10': 10,
    'core50': 50,
    'mini_imagenet': 100,
    'openloris': 69,
    'flower102': 102,
    'DTD': 40,
    'aircraft':100,
    'CelebA':10177,
    'Country211':211,
    'StanfordCars':196
}


transforms_match = {
    'core50': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        ]),
    'cifar100': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        ]),
    'cifar10': transforms.Compose([
        transforms.ToTensor(),
        ]),
    'mini_imagenet': transforms.Compose([
        transforms.ToTensor()]),
    'openloris': transforms.Compose([
            transforms.ToTensor()]),
    'flower102': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        ]),
    'eurosat': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        ]),
    'DTD': transforms.Compose([
        # transforms.Resize((224,224)),
        transforms.ToTensor(),
        ]),
    'aircraft':transforms.Compose([
        transforms.ToTensor(),
        ]),
    'CelebA':transforms.Compose([
        transforms.ToTensor(),
        ]),
    'Country211': transforms.Compose([
        transforms.ToTensor(),
        ]),
    'StanfordCars': transforms.Compose([
        transforms.ToTensor(),
        ]),

}
transforms_aug = {
    'cifar100': transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4),
        transforms.Resize((224,224)),
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))
        ]),
    'DTD': transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4),
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        ]),
    'core50': transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4),
        transforms.Resize((224,224)),
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2615))
        ]),
    'aircraft': transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4),
        transforms.Resize((224,224)),
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        # transforms.Normalize((0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821))
        ])
}

def setup_architecture(params):
    nclass = n_classes[params.data]
    if params.agent in ['SCR', 'SCP']:
        if params.data == 'mini_imagenet':
            return SupConResNet(640, head=params.head)
        return SupConResNet(head=params.head)
    if params.agent == 'FOAL':
        return Encoder(nclass,params.projection)
    if params.agent == 'ER_DVC':
        return VIT_DVC(nclass)
    if params.agent == 'CNDPM':
        from models.ndpm.ndpm import Ndpm
        return Ndpm(params)
    return ModifiedViT(nclass)




def setup_opt(optimizer, model, lr, wd):
    if optimizer == 'SGD':
        optim = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                weight_decay=wd)
    elif optimizer == 'Adam':
        optim = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=wd)
    else:
        raise Exception('wrong optimizer name')
    return optim
