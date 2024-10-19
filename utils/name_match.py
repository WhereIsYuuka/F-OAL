from agents.gdumb import Gdumb
from continuum.dataset_scripts.cifar100 import CIFAR100
from continuum.dataset_scripts.cifar10 import CIFAR10
from continuum.dataset_scripts.core50 import CORE50
from continuum.dataset_scripts.mini_imagenet import Mini_ImageNet
from continuum.dataset_scripts.openloris import OpenLORIS
from continuum.dataset_scripts.CelebA import CelebA
from continuum.dataset_scripts.food101 import FOOD101
from continuum.dataset_scripts.DTD import DTD
from continuum.dataset_scripts.Aircraft import FGVCAIRCRAFT
from continuum.dataset_scripts.Country211 import Country211
from agents.exp_replay import ExperienceReplay
from agents.agem import AGEM
from agents.ewc_pp import EWC_pp
from agents.cndpm import Cndpm
from agents.lwf import Lwf
from agents.icarl import Icarl
from agents.FOAL import FOAL
from agents.scr import SupContrastReplay
from agents.pcr import ProxyContrastiveReplay
from agents.exp_replay_dvc import ExperienceReplay_DVC
from utils.buffer.random_retrieve import Random_retrieve
from utils.buffer.reservoir_update import Reservoir_update
from utils.buffer.mir_retrieve import MIR_retrieve
from utils.buffer.gss_greedy_update import GSSGreedyUpdate
from utils.buffer.aser_retrieve import ASER_retrieve
from utils.buffer.aser_update import ASER_update
from utils.buffer.sc_retrieve import Match_retrieve
from utils.buffer.mem_match import MemMatch_retrieve
from utils.buffer.mgi_retrieve import MGI_retrieve

data_objects = {
    'cifar100': CIFAR100,
    'cifar10': CIFAR10,
    'core50': CORE50,
    'mini_imagenet': Mini_ImageNet,
    'openloris': OpenLORIS,
    'CelebA':CelebA,
    'food101':FOOD101,
    'DTD':DTD,
    'aircraft':FGVCAIRCRAFT,
    'Country211':Country211,
}

agents = {
    'ER': ExperienceReplay,
    'EWC': EWC_pp,
    'AGEM': AGEM,
    'CNDPM': Cndpm,
    'LWF': Lwf,
    'ICARL': Icarl,
    'GDUMB': Gdumb,
    'SCR': SupContrastReplay,
    'FOAL': FOAL,
    'PCR': ProxyContrastiveReplay,
    'ER_DVC':ExperienceReplay_DVC
}

retrieve_methods = {
    'MIR': MIR_retrieve,
    'random': Random_retrieve,
    'ASER': ASER_retrieve,
    'match': Match_retrieve,
    'mem_match': MemMatch_retrieve,
    'MGI':MGI_retrieve

}

update_methods = {
    'random': Reservoir_update,
    'GSS': GSSGreedyUpdate,
    'ASER': ASER_update
}

