import torch
import numpy as np

def train_collate(batch):
    batch_size = len(batch)#2*3*3
    domain_num = len(batch[0])
    # print('domain',batch_size,domain_num)
    inputs = []
    bboxes = []
    labels = []
    for b in range(batch_size):
        for d_num in range(domain_num):
            inputs.append(batch[b][d_num][0])
            bboxes.append(batch[b][d_num][1])
            labels.append(batch[b][d_num][2])
    # inputs = torch.stack([batch[b][0] for b in range(batch_size)], 0)
    # bboxes = [batch[b][1] for b in range(batch_size)]
    # labels = [batch[b][2] for b in range(batch_size)]
    # print(type(inputs[0]))
    inputs = torch.stack(inputs, 0)

    return [inputs, bboxes, labels]


def eval_collate(batch):
    batch_size = len(batch)
    inputs = torch.stack([batch[b][0] for b in range(batch_size)], 0)
    bboxes = [batch[b][1] for b in range(batch_size)]
    labels = [batch[b][2] for b in range(batch_size)]

    return [inputs, bboxes, labels]


def test_collate(batch):
    batch_size = len(batch)
    for b in range(batch_size): 
        inputs = torch.stack([batch[b][0]for b in range(batch_size)], 0)
        images = [batch[b][1] for b in range(batch_size)]

    return [inputs, images]
