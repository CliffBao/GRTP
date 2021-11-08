# Lifelong Vehicle Trajectory Prediction Framework Based on Generative Replay

## Abstract 

Accurate trajectory prediction of vehicles is essential for reliable and safe autonomous driving. To maintain consistent performance as a vehicle driving around different cities, it is crucial to adapt to changing traffic circumstances and achieve lifelong trajectory prediction model. To learn a lifelong prediction model,  catastrophic forgetting is a main problem to be addressed. In this paper, a divergence measurement method is proposed first to evaluate spatiotemporal  dependency difference among varied driving circumstances. Conditional probability density estimation is realized through the Mixture Density Networks, which results in Gaussian Mixture Models for different circumstances. Conditional Kullback-Leibler divergence between two GMMs is calculated through pointwise Monte-Carlo sampling, which indicates difference of two circumstances. Then based on generative replay, a novel lifelong vehicle trajectory prediction framework is developed. The proposed framework consists of a conditional generation model and a vehicle trajectory prediction model. The conditional generation model is a generative adversarial network conditioned on position configuration of vehicles, which is called Recurrent Regression GAN. After learning and merging trajectory distribution of vehicles across different cities, the generation model replays trajectories with prior samplings as inputs. The vehicle trajectory prediction model is trained by the replayed trajectories and achieves consistent prediction performance on visited cities. A lifelong experiment setup is established on four open datasets US101, I801, HighD, and Interaction dataset including five tasks. Spatiotemporal dependency divergence is calculated for different tasks. Even though these divergence, the proposed framework exhibits lifelong learning ability and achieves comparable performance to the best in theory.

## Introduction

This is source code repository for *Lifelong Vehicle Trajectory Prediction Framework Based on Generative Replay*.

## Runtime environment 

Python 3.6

Pytorch

## File description

### CKLD

This folder is used for conditional probability density function divergence calculation. 

### clstm_fixed

Lifelong prediction for method FM (fixed model).

### clstm_finetuning

Lifelong prediction for method FT (fine tuning).

### clstm_fullavailable

Lifelong prediction for method JT (joint training).

### clstm_lifelong

Lifelong prediction for method GRTP-D and GRTP-T. Five folders are included corresponding to lifelong task chain. In each task, a solver is contained that predicts vehicle trajectories, which is the same as that in FM, FT and JT.