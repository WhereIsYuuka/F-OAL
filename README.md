
# FOAL

This repository contains the implementation of the F-OAL: Forward-only Online Analytic Learning with
Fast Training and Low Memory Footprint in Class Incremental Learning(https://arxiv.org/abs/2403.15751).


## Introduction

Online Class Incremental Learning (OCIL) aims to train models incrementally, where data arrive in mini-batches, and previous data are not accessible. A major challenge in OCIL is Catastrophic Forgetting, i.e., the loss of previously learned knowledge. Among existing baselines, replay-based methods show competitive results but requires extra memory for storing exemplars, while exemplar-free (i.e., data need not be stored for replay in production) methods are resource-friendly but often lack accuracy. In this paper, we propose an exemplar-free  approach—Forward-only Online Analytic Learning (F-OAL). Unlike traditional methods, F-OAL does not rely on back-propagation and is forward-only, significantly reducing memory usage and computational time. Cooperating with a pre-trained frozen encoder with Feature Fusion, F-OAL only needs to update a linear classifier by recursive least square. This approach simultaneously achieves high accuracy and low resource consumption. Extensive experiments on benchmark datasets demonstrate F-OAL's robust performance in OCIL scenarios.



## About Analytic Continual Learning
Analytic Continual Learning (ACL) is a new branch for Continual Learning (CL). ACL redefines CL problem into a recursive least square form, which is solved in one epoch with high accuracy. F-OAL expand ACL to OCIL setting. Other ACL papers and codes can be found at (https://github.com/ZHUANGHP/Analytic-continual-learning)

## Reproduce the results in the paper
Using following command can reproduce some of results:
```
python general_main.py --data cifar100 --cl_type nc --agent FOAL
python general_main.py --data cifar100 --cl_type nc --agent LWF --learning_rate 0.001
python general_main.py --data cifar100 --cl_type nc --agent EWC --fisher_update_after 50 --alpha 0.9 --lambda 100 --learning_rate 0.001
python general_main.py --data cifar100 --cl_type nc --agent ER --retrieve random --update random --mem_size 5000 --learning_rate 0.001
python general_main.py --data cifar100 --cl_type nc --agent ICARL --retrieve random --update random --mem_size 5000 --learning_rate 0.001
python general_main.py --data cifar100 --cl_type nc --agent ER --update ASER --retrieve ASER --mem_size 5000 --aser_type asvm --n_smp_cls 1.5 --k 3 --learning_rate 0.001
python general_main.py --data cifar100 --cl_type nc --agent SCR --retrieve random --update random --mem_size 5000 --head mlp --temp 0.07 --eps_mem_batch 10 --learning_rate 0.001
python general_main.py --data  cifar100 --cl_type nc --agent PCR  --retrieve random --update random --mem_size 5000 --learning_rate 0.001
python general_main.py--data  cifar100 --cl_type nc --agent ER_DVC  --retrieve MGI --update random --mem_size 5000 --dl_weight 4.0 --learning_rate 0.001
```

