
# FOAL

This repository contains the implementation of the F-OAL: Forward-only Online Analytic Learning with
Fast Training and Low Memory Footprint in Class Incremental Learning(https://arxiv.org/abs/2403.15751).


## Introduction

Online Class Incremental Learning (OCIL) aims to train models incrementally, where data arrive in mini-batches, and previous data are not accessible. A major challenge in OCIL is Catastrophic Forgetting, i.e., the loss of previously learned knowledge. Among existing baselines, replay-based methods show competitive results but requires extra memory for storing exemplars, while exemplar-free (i.e., data need not be stored for replay in production) methods are resource-friendly but often lack accuracy. In this paper, we propose an exemplar-free  approach—Forward-only Online Analytic Learning (F-OAL). Unlike traditional methods, F-OAL does not rely on back-propagation and is forward-only, significantly reducing memory usage and computational time. Cooperating with a pre-trained frozen encoder with Feature Fusion, F-OAL only needs to update a linear classifier by recursive least square. This approach simultaneously achieves high accuracy and low resource consumption. Extensive experiments on benchmark datasets demonstrate F-OAL's robust performance in OCIL scenarios.


