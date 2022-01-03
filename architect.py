import sys
import os
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchcontrib
import numpy as np
from pdb import set_trace as bp
from thop import profile
from operations import *
from genotypes import PRIMITIVES


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args):
        # self.network_momentum = args.momentum
        # self.network_weight_decay = args.weight_decay
        self.model = model
        self._args = args

        self.optimizer = torch.optim.Adam(list(self.model.module._arch_params.values()), lr=args.arch_learning_rate, betas=(0.5, 0.999))#, weight_decay=args.arch_weight_decay)
        
        self.flops_weight = args.flops_weight

        self.distill_weight = args.distill_weight

        self.cascad_arch = args.cascad_arch

        self.criteria_arch = args.criteria_arch

        print("architect initialized!")


    def step(self, input_train, target_train, input_valid, target_valid, num_bits_list, bit_schedule, loss_scale, temp=1):
        if len(num_bits_list) == 1:
            bit_schedule = 'high2low'

        self.optimizer.zero_grad()

        loss, loss_flops = self._backward_step_flops(input_valid, target_valid, num_bits_list, bit_schedule, loss_scale, temp=temp)

        # loss.backward()
        # self.optimizer.step()

        return loss


    def _backward_step_flops(self, input_valid, target_valid, num_bits_list, bit_schedule, loss_scale, temp=1):
        loss_value = [-1 for _ in num_bits_list]

        if self.criteria_arch is not None:
            if self.criteria_arch == 'min':
                num_bits = min(num_bits_list)
            elif self.criteria_arch == 'max':
                num_bits = max(num_bits_list)
            else:
                num_bits = np.random.choice(num_bits_list)
            
            logit = self.model(input_valid, num_bits, temp=temp)
            loss = self.model.module._criterion(logit, target_valid)

            loss = loss * loss_scale[num_bits_list.index(num_bits)]

            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_value[num_bits_list.index(num_bits)] = loss.item()

        elif bit_schedule == 'sandwich':
            num_bits_list_sort = sorted(num_bits_list)
            max_bits = num_bits_list_sort[-1]
            min_bits = num_bits_list_sort[0]
            random_bits_1 = np.random.choice(num_bits_list_sort[1:-1])
            random_bits_2 = np.random.choice(num_bits_list_sort[1:-1])

            for num_bits in [max_bits, min_bits, random_bits_1, random_bits_2]:
                logit = self.model(input_valid, num_bits, temp=temp)
                loss = self.model.module._criterion(logit, target_valid)

                loss = loss * loss_scale[num_bits_list.index(num_bits)]

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                loss_value[num_bits_list.index(num_bits)] = loss.item()

        elif bit_schedule == 'low2high':
            for num_bits in sorted(num_bits_list):
                logit = self.model(input_valid, num_bits, temp=temp)
                loss = self.model.module._criterion(logit, target_valid)

                loss = loss * loss_scale[num_bits_list.index(num_bits)]

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                loss_value[num_bits_list.index(num_bits)] = loss.item()

        elif bit_schedule == 'high2low':
            for num_bits in sorted(num_bits_list, reverse=True):
                logit = self.model(input_valid, num_bits, temp=temp)
                loss = self.model.module._criterion(logit, target_valid)

                loss = loss * loss_scale[num_bits_list.index(num_bits)]

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                loss_value[num_bits_list.index(num_bits)] = loss.item()


        elif bit_schedule == 'avg_loss':
            if self.distill_weight > 0:
                if self.cascad_arch:
                    teacher_list = []
                    for num_bits in num_bits_list[::-1]:
                        logit = self.model(input_valid, num_bits, temp=temp)
                        loss = self.model.module._criterion(logit, target_valid)

                        if len(teacher_list) > 0:
                            for logit_teacher in teacher_list:
                                loss += self.distill_weight * nn.MSELoss()(logit, logit_teacher)
                        
                        teacher_list.append(logit.detach())

                        loss = loss * loss_scale[num_bits_list.index(num_bits)]

                        loss.backward()

                        loss_value[num_bits_list.index(num_bits)] = loss.item()

                        del logit
                        del loss

                else:
                    logit = self.model(input_valid, num_bits_list[-1], temp=temp)
                    loss = self.model.module._criterion(logit, target_valid)
                    loss = loss * loss_scale[-1]
                    loss.backward()
                    loss_value[-1] = loss.item()

                    logit_teacher = logit.detach()

                    del logit
                    del loss

                    for num_bits in num_bits_list[:-1]:
                        logit = self.model(input_valid, num_bits, temp=temp)
                        loss = self.model.module._criterion(logit, target_valid) + self.distill_weight * nn.MSELoss()(logit, logit_teacher)

                        loss = loss * loss_scale[num_bits_list.index(num_bits)]

                        loss.backward()

                        loss_value[num_bits_list.index(num_bits)] = loss.item()

                        del logit
                        del loss

            else:
                for num_bits in num_bits_list:
                    logit = self.model(input_valid, num_bits, temp=temp)
                    loss = self.model.module._criterion(logit, target_valid)

                    loss = loss * loss_scale[num_bits_list.index(num_bits)]

                    loss.backward()

                    loss_value[num_bits_list.index(num_bits)] = loss.item()

                    del logit
                    del loss

            self.optimizer.step()
            self.optimizer.zero_grad()


        elif bit_schedule == 'max_loss':
            if self.distill_weight > 0:
                loss_list = []

                for i, num_bits in enumerate(num_bits_list[:-1]):
                    logit = self.model(input_valid, num_bits, temp=temp)
                    loss = self.model.module._criterion(logit, target_valid)

                    loss_list.append(loss.item())

                    del logit
                    del loss

                logit = self.model(input_valid, num_bits_list[-1], temp=temp)
                loss = self.model.module._criterion(logit, target_valid)
                loss_list.append(loss.item())

                logit_teacher = logit.detach()

                del logit
                del loss

                num_bits_max = num_bits_list[np.array(loss_list).argmax()]

                logit = self.model(input_valid, num_bits_max, temp=temp)

                if num_bits_max == num_bits_list[-1]:
                    loss = self.model.module._criterion(logit, target_valid)
                else:
                    loss = self.model.module._criterion(logit, target_valid) + self.distill_weight * nn.MSELoss()(logit, logit_teacher)

                loss = loss * loss_scale[num_bits_list.index(num_bits_max)]

                loss.backward()

            else:
                loss_list = []

                for i, num_bits in enumerate(num_bits_list):
                    logit = self.model(input_valid, num_bits, temp=temp)
                    loss = self.model.module._criterion(logit, target_valid)

                    loss_list.append(loss.item())

                    del logit
                    del loss

                num_bits_max = num_bits_list[np.array(loss_list).argmax()]

                logit = self.model(input_valid, num_bits_max, temp=temp)
                loss = self.model.module._criterion(logit, target_valid)

                loss = loss * loss_scale[num_bits_list.index(num_bits_max)]

                loss.backward()
                
            self.optimizer.step()
            self.optimizer.zero_grad()

            # loss_value[num_bits_list.index(num_bits_max)] = loss.item()
            loss_value = loss_list

        else:
            print('Wrong Bit Schedule.')
            sys.exit()


        if self.flops_weight > 0:
            # flops = self.model.module.forward_flops((16, 32, 32))
            flops = self.model.module.forward_flops((3, 32, 32), temp=temp)
                
            self.flops_supernet = flops
            loss_flops = self.flops_weight * flops

            loss_flops.backward()
        else:
            loss_flops = 0
            self.flops_supernet = 0
        
        self.optimizer.step()
        self.optimizer.zero_grad()

        # print(flops, loss_flops, loss)
        return loss_value, loss_flops


