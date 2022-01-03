from __future__ import division
import os
import sys
import time
import glob
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from torch.autograd import Variable

from tensorboardX import SummaryWriter

import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

from config_train import config

from model_search import FBNet as Network
from model_infer import FBNet_Infer

from thop import profile
from thop.count_hooks import count_convNd

import genotypes

import operations
from quantize import QConv2d

from calibrate_bn import bn_update

operations.DWS_CHWISE_QUANT = config.dws_chwise_quant

custom_ops = {QConv2d: count_convNd}

import argparse

parser = argparse.ArgumentParser(description='Search on CIFAR-100')
parser.add_argument('--dataset_path', type=str, default=None,
                    help='path to cifar-100')
args = parser.parse_args()

def main(pretrain=True):
    if args.dataset_path is not None:
        config.dataset_path = args.dataset_path
        
    config.save = 'ckpt/{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    logger = SummaryWriter(config.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info("args = %s", str(config))
    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if os.path.exists(os.path.join(config.load_path, 'arch.pt')):
        state = torch.load(os.path.join(config.load_path, 'arch.pt'))
        alpha = state['alpha']
    else:
        print('No arch.pt')
        sys.exit()
        # alpha = torch.zeros(sum(config.num_layer_list), len(genotypes.PRIMITIVES))
        # alpha[:,0] = 10

    # Model #######################################
    model = FBNet_Infer(alpha, config=config)

    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32),), custom_ops=custom_ops)
    logging.info("params = %fMB, FLOPs = %fGB", params / 1e6, flops / 1e9)

    model = torch.nn.DataParallel(model).cuda()

    # if type(config.pretrain) == str:
    #     state_dict = torch.load(config.pretrain)

    #     for key in state_dict.copy():
    #         if 'bn.0' in key:
    #             new_key_list = []

    #             for i in range(1, len(config.num_bits_list)):
    #                 new_key = []
    #                 new_key.extend(key.split('.')[:-2])
    #                 new_key.append(str(i))
    #                 new_key.append(key.split('.')[-1])
    #                 new_key = '.'.join(new_key)

    #                 state_dict[new_key] = state_dict[key]

    #     model.load_state_dict(state_dict, strict=False)

    if type(config.pretrain) == str:
        state_dict = torch.load(config.pretrain)
        model.load_state_dict(state_dict)


    if config.opt == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=config.betas)
    elif config.opt == 'Sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay)
    else:
        logging.info("Wrong Optimizer Type.")
        sys.exit()

    # lr policy ##############################
    total_iteration = config.nepochs * config.niters_per_epoch
    
    if config.lr_schedule == 'linear':
        lr_policy = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(config.nepochs, 0, config.decay_epoch).step)
    elif config.lr_schedule == 'exponential':
        lr_policy = torch.optim.lr_scheduler.ExponentialLR(optimizer, config.lr_decay)
    elif config.lr_schedule == 'multistep':
        lr_policy = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
    elif config.lr_schedule == 'cosine':
        lr_policy = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(config.nepochs), eta_min=config.learning_rate_min)
    else:
        logging.info("Wrong Learning Rate Schedule Type.")
        sys.exit()


    # data loader ############################

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    if config.dataset == 'cifar10':
        train_data = dset.CIFAR10(root=config.dataset_path, train=True, download=True, transform=transform_train)
        test_data = dset.CIFAR10(root=config.dataset_path, train=False, download=True, transform=transform_test)
    elif config.dataset == 'cifar100':
        train_data = dset.CIFAR100(root=config.dataset_path, train=True, download=True, transform=transform_train)
        test_data = dset.CIFAR100(root=config.dataset_path, train=False, download=True, transform=transform_test)
    else:
        print('Wrong dataset.')
        sys.exit()

    train_loader_model = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size,
        pin_memory=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=config.batch_size,
                                          shuffle=False,
                                          num_workers=4)

    if config.finetune_bn:
        train_bn(train_loader_model, model, logger, config.num_bits_list, config.distill_weight, config.niters_per_epoch, 
            ft_bn_epoch=config.ft_bn_epoch, ft_bn_lr=config.ft_bn_lr, ft_bn_momentum=config.ft_bn_momentum)

    if config.eval_only:
        # if config.finetune_bn:
            # train_bn(train_loader_model, model, optimizer, lr_policy, logger, epoch, num_bits_list=config.num_bits_list,
            #     bit_schedule=config.bit_schedule, loss_scale=config.loss_scale, distill_weight=config.distill_weight)
        logging.info('Eval - Acc under different bits: ' + str(infer(0, model, train_loader_model, test_loader, logger, config.num_bits_list, update_bn=config.update_bn, show_distrib=config.show_distrib)))
        sys.exit(0)

    tbar = tqdm(range(config.nepochs), ncols=80)
    for epoch in tbar:
        logging.info(config.save)
        logging.info("lr: " + str(optimizer.param_groups[0]['lr']))

        # training
        tbar.set_description("[Epoch %d/%d][train...]" % (epoch + 1, config.nepochs))

        if config.num_bits_list_schedule:
            num_bits_list = update_num_bits_list(config.num_bits_list, config.num_bits_list_schedule, config.schedule_freq, epoch)
        else:
            num_bits_list = config.num_bits_list

        train(train_loader_model, model, optimizer, lr_policy, logger, epoch, num_bits_list=num_bits_list, bit_schedule=config.bit_schedule, 
             loss_scale=config.loss_scale, distill_weight=config.distill_weight, cascad=config.cascad, update_bn_freq=config.update_bn_freq)

        torch.cuda.empty_cache()
        lr_policy.step()

        # validation
        if epoch and not (epoch+1) % config.eval_epoch:
            tbar.set_description("[Epoch %d/%d][validation...]" % (epoch + 1, config.nepochs))
            
            with torch.no_grad():

                acc_bits = infer(epoch, model, train_loader_model, test_loader, logger, config.num_bits_list, update_bn=False, show_distrib=False)

                for i, num_bits in enumerate(config.num_bits_list):
                    logger.add_scalar('acc/val_bits_%d' % num_bits, acc_bits[i], epoch)
                    
                logging.info("Epoch: " + str(epoch) +" Acc under different bits: " + str(acc_bits))
                
                logger.add_scalar('flops/val', flops, epoch)
                logging.info("Epoch %d: flops %.3f"%(epoch, flops))

            save(model, os.path.join(config.save, 'weights_%d.pt'%epoch))

    save(model, os.path.join(config.save, 'weights.pt'))

    logging.info('Final Eval - Acc under different bits: ' + str(infer(0, model, train_loader_model, test_loader, logger, config.num_bits_list, update_bn=config.update_bn, show_distrib=config.show_distrib)))



def train(train_loader_model, model, optimizer, lr_policy, logger, epoch, num_bits_list, bit_schedule, loss_scale, distill_weight, cascad, update_bn_freq):
    model.train()

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format, ncols=80)
    dataloader_model = iter(train_loader_model)

    if len(num_bits_list) == 1:
        bit_schedule = 'high2low'

    for step in pbar:
        lr = optimizer.param_groups[0]['lr']

        optimizer.zero_grad()

        input, target = dataloader_model.next()

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        loss_value = [-1 for _ in num_bits_list]

        if bit_schedule == 'sandwich':
            num_bits_list_sort = sorted(num_bits_list)
            max_bits = num_bits_list_sort[-1]
            min_bits = num_bits_list_sort[0]
            random_bits_1 = np.random.choice(num_bits_list_sort[1:-1])
            random_bits_2 = np.random.choice(num_bits_list_sort[1:-1])

            for num_bits in [max_bits, min_bits, random_bits_1, random_bits_2]:
                logit = model(input, num_bits)
                loss = model.module._criterion(logit, target)

                loss = loss * loss_scale[num_bits_list.index(num_bits)]

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

                loss_value[num_bits_list.index(num_bits)] = loss.item()

        elif bit_schedule == 'low2high':
            for num_bits in sorted(num_bits_list):
                logit = model(input, num_bits)
                loss = model.module._criterion(logit, target)

                loss = loss * loss_scale[num_bits_list.index(num_bits)]

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

                loss_value[num_bits_list.index(num_bits)] = loss.item()

        elif bit_schedule == 'high2low':
            for num_bits in sorted(num_bits_list, reverse=True):
                logit = model(input, num_bits)
                loss = model.module._criterion(logit, target)

                loss = loss * loss_scale[num_bits_list.index(num_bits)]

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

                loss_value[num_bits_list.index(num_bits)] = loss.item()

        elif bit_schedule == 'avg_loss':
            if distill_weight > 0:
                if cascad:
                    teacher_list = []
                    for num_bits in num_bits_list[::-1]:
                        logit = model(input, num_bits)
                        loss = model.module._criterion(logit, target)

                        if len(teacher_list) > 0:
                            for logit_teacher in teacher_list:
                                loss += distill_weight * nn.MSELoss()(logit, logit_teacher)
                        
                        teacher_list.append(logit.detach())

                        loss = loss * loss_scale[num_bits_list.index(num_bits)]

                        loss.backward()

                        loss_value[num_bits_list.index(num_bits)] = loss.item()

                        del logit
                        del loss

                else:
                    logit = model(input, num_bits_list[-1])
                    loss = model.module._criterion(logit, target)
                    loss = loss * loss_scale[-1]
                    loss.backward()
                    loss_value[-1] = loss.item()

                    logit_teacher = logit.detach()

                    del logit
                    del loss

                    for num_bits in num_bits_list[:-1]:
                        logit = model(input, num_bits)
                        loss = model.module._criterion(logit, target) + distill_weight * nn.MSELoss()(logit, logit_teacher)

                        loss = loss * loss_scale[num_bits_list.index(num_bits)]

                        loss.backward()

                        loss_value[num_bits_list.index(num_bits)] = loss.item()

                        del logit
                        del loss

            else:
                for num_bits in num_bits_list:
                    logit = model(input, num_bits)
                    loss = model.module._criterion(logit, target)

                    loss = loss * loss_scale[num_bits_list.index(num_bits)]

                    loss.backward()

                    loss_value[num_bits_list.index(num_bits)] = loss.item()

                    del logit
                    del loss

            # nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        elif bit_schedule == 'max_loss':
            if distill_weight > 0:
                if cascad:
                    loss_list = []
                    teacher_list = []

                    for i, num_bits in enumerate(num_bits_list):
                        logit = model(input, num_bits)
                        loss = model.module._criterion(logit, target)

                        loss_list.append(loss.item())
                        teacher_list.append(logit.detach())

                        del logit
                        del loss

                    num_bits_max = num_bits_list[np.array(loss_list).argmax()]

                    logit = model(input, num_bits_max)
                    loss = model.module._criterion(logit, target)

                    for logit_teacher in teacher_list[num_bits_list.index(num_bits_max)+1:]:
                        loss += distill_weight * nn.MSELoss()(logit, logit_teacher)

                    loss = loss * loss_scale[num_bits_list.index(num_bits_max)]

                    loss.backward()

                else:
                    loss_list = []

                    for i, num_bits in enumerate(num_bits_list[:-1]):
                        logit = model(input, num_bits)
                        loss = model.module._criterion(logit, target)

                        loss_list.append(loss.item())

                        del logit
                        del loss

                    logit = model(input, num_bits_list[-1])
                    loss = model.module._criterion(logit, target)
                    loss_list.append(loss.item())

                    logit_teacher = logit.detach()

                    del logit
                    del loss

                    num_bits_max = num_bits_list[np.array(loss_list).argmax()]

                    logit = model(input, num_bits_max)

                    if num_bits_max == num_bits_list[-1]:
                        loss = model.module._criterion(logit, target)
                    else:
                        loss = model.module._criterion(logit, target) + distill_weight * nn.MSELoss()(logit, logit_teacher)

                    loss = loss * loss_scale[num_bits_list.index(num_bits_max)]

                    loss.backward()

            else:
                loss_list = []

                for i, num_bits in enumerate(num_bits_list):
                    logit = model(input, num_bits)
                    loss = model.module._criterion(logit, target)

                    loss_list.append(loss.item())

                    del logit
                    del loss

                num_bits_max = num_bits_list[np.array(loss_list).argmax()]

                logit = model(input, num_bits_max)
                loss = model.module._criterion(logit, target)

                loss = loss * loss_scale[num_bits_list.index(num_bits_max)]

                loss.backward()
            
            # nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            # loss_value[num_bits_list.index(num_bits_max)] = loss.item()
            loss_value = loss_list

        else:
            logging.info('Wrong Bit Schedule.')
            sys.exit()

        for i, num_bits in enumerate(num_bits_list):
            if loss_value[i] != -1:
                logger.add_scalar('loss/num_bits_%d' % num_bits, loss_value[i], epoch*len(pbar)+step)

        pbar.set_description("[Step %d/%d]" % (step + 1, len(train_loader_model)))

    torch.cuda.empty_cache()


def train_bn(train_loader_model, model, logger, num_bits_list, distill_weight, niters_per_epoch, ft_bn_epoch=10, ft_bn_lr=1e-3, ft_bn_momentum=0.9):
    model.train()

    dataloader_model = iter(train_loader_model)

    param_list = [[] for _ in num_bits_list]

    for name, param in model.named_parameters():
        for index in range(len(num_bits_list)):
            if 'bn.'+str(index) in name and ('weight' in name or 'bias' in name):
                param_list[index].append(param)

    optimizer_list = []

    for index in range(len(num_bits_list)):
        optimizer_list.append(
            torch.optim.SGD(
            param_list[index],
            lr=ft_bn_lr,
            momentum=ft_bn_momentum
            )
        )

    for _epoch in range(ft_bn_epoch):
        # print('Epoch:', _epoch)

        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(niters_per_epoch), file=sys.stdout, bar_format=bar_format, ncols=80)
        dataloader = iter(train_loader_model)

        for step in pbar:
            input, target = dataloader.next()

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            loss_value = [-1 for _ in num_bits_list]

            if distill_weight > 0:
                logit = model(input, num_bits_list[-1])
                loss = model.module._criterion(logit, target)
                loss.backward()

                optimizer_list[-1].step()
                optimizer_list[-1].zero_grad()

                loss_value[-1] = loss.item()

                logit_teacher = logit.detach()

                del logit
                del loss

                for index, num_bits in enumerate(num_bits_list[:-1]):
                    logit = model(input, num_bits)
                    loss = model.module._criterion(logit, target) + distill_weight * nn.MSELoss()(logit, logit_teacher)
                    loss.backward()

                    # nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                    optimizer_list[index].step()
                    optimizer_list[index].zero_grad()

                    loss_value[index] = loss.item()

                    del logit
                    del loss

            else:
                for index, num_bits in enumerate(num_bits_list):
                    logit = model(input, num_bits)
                    loss = model.module._criterion(logit, target)
                    loss.backward()

                    # nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                    optimizer_list[index].step()
                    optimizer_list[index].zero_grad()

                    loss_value[index] = loss.item()

                    del logit
                    del loss

            for i, num_bits in enumerate(num_bits_list):
                if loss_value[i] != -1:
                    logger.add_scalar('loss_ft_bn/num_bits_%d' % num_bits, loss_value[i], _epoch*len(pbar)+step)

            pbar.set_description("[Epoch %d/%d Step %d/%d]" % (_epoch+1, ft_bn_epoch, step + 1, len(train_loader_model)))

    torch.cuda.empty_cache()


def infer(epoch, model, train_loader, test_loader, logger, num_bits_list, update_bn=False, show_distrib=False):
    model.eval()
    acc_bits = []

    if show_distrib:
        self_entropy_list = [[] for _ in range(len(num_bits_list))]

    for num_bits in num_bits_list:
        prec1_list = []

        if update_bn:
            bn_update(train_loader, model, num_bits=num_bits)
            save(model, os.path.join(config.save, 'weights_ftbn.pt'))
            model.eval()

        for i, (input, target) in enumerate(test_loader):
            input_var = Variable(input, volatile=True).cuda()
            target_var = Variable(target, volatile=True).cuda()

            output = model(input_var, num_bits)

            if show_distrib:
                output_softmax = torch.nn.functional.softmax(output, dim=-1)
                self_entropy = -torch.mean(torch.sum(output_softmax * torch.log(output_softmax), dim=-1)).item()
                self_entropy_list[num_bits_list.index(num_bits)].append(self_entropy)

            prec1, = accuracy(output.data, target_var, topk=(1,))
            prec1_list.append(prec1)

        acc = sum(prec1_list)/len(prec1_list)
        acc_bits.append(acc)

    if show_distrib:
        for i in range(len(self_entropy_list)):
            self_entropy_list[i]  = sum(self_entropy_list[i])/len(self_entropy_list[i])
        print('Self-Entropy under different bits: ' + str(self_entropy_list))

    return acc_bits


def update_num_bits_list(num_bits_list_orig, num_bits_list_schedule, schedule_freq, epoch):
    assert num_bits_list_schedule in ['low2high', 'high2low']

    if num_bits_list_schedule == 'low2high':
        num_bits_list = num_bits_list_orig[:int(epoch // schedule_freq + 1)]

    elif num_bits_list_schedule == 'high2low':
        num_bits_list = num_bits_list_orig[-int(epoch // schedule_freq + 1):]

    return num_bits_list



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save(model, model_path):
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main() 
