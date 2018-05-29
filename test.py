# -*- coding: utf-8 -*-
import os
import torch
import numpy
import random
import argparse
from TorchNN import *
from driver.Config import Configurable
from driver.MyIO import read_pkl
from driver.Vocab import PAD, VocabSrc, VocabTgt
from driver.Train import predict

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["OMP_NUM_THREADS"] = "1"


if __name__ == '__main__':
    # random
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    random.seed(666)
    numpy.random.seed(666)

    # gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: ", torch.backends.cudnn.enabled)

    # parameters
    parse = argparse.ArgumentParser('Attention Target Classifier')
    parse.add_argument('--config_file', type=str, default='default.ini')
    parse.add_argument('--thread', type=int, default=1)
    parse.add_argument('--use_cuda', action='store_true', default=False)
    parse.add_argument('--model', type=str, default='model.742')
    args, extra_args = parse.parse_known_args()

    config = Configurable(args.config_file, extra_args)
    torch.set_num_threads(args.thread)
    config.use_cuda = False
    if gpu and args.use_cuda:
        config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    # load vocab and model
    feature_list = read_pkl(config.load_feature_voc)
    label_list = read_pkl(config.load_label_voc)
    feature_vec = VocabSrc(feature_list)
    label_vec = VocabTgt(label_list)

    # model
    if config.which_model == 'Vanilla':
        model = Vanilla(config, feature_vec.size, config.embed_dim,
                        PAD, label_vec.size)
    elif config.which_model == 'Contextualized':
        model = Contextualized(config, feature_vec.size, config.embed_dim,
                               PAD, label_vec.size)
    elif config.which_model == 'ContextualizedGates':
        model = ContextualizedGates(config, feature_vec.size, config.embed_dim,
                                    PAD, label_vec.size)
    else:
        print('please choose right model')
        exit()
    model_path = os.path.join(config.load_model_path, args.model)
    model.load_state_dict(torch.load(model_path,map_location=lambda storage, loc: storage))
    if config.use_cuda:
        torch.backends.cudnn.enabled = True
        model = model.cuda()

    # test
    # 顺便计算F1 score等等数据

