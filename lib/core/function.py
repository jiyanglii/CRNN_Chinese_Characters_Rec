from  __future__ import  absolute_import
import time

import numpy as np

import lib.utils.utils as utils
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(config, train_loader, dataset, converter, model, criterion, optimizer, device, epoch, writer_dict=None, output_dict=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    for i, (inp, idx) in enumerate(train_loader):
        # measure data time
        data_time.update(time.time() - end)

        labels = utils.get_batch_label(dataset, idx)
        # print(idx)
        # print(labels)
        inp = inp.to(device)
        # print(inp)

        # inference
        preds = model(inp).cpu()
        # preds = model(inp)
        preds = preds.to(torch.float64)
        preds = preds.to(device)
        # compute loss
        batch_size = inp.size(0)
        text, length = converter.encode(labels)                    # length = 一个batch中的总字符长度, text = 一个batch中的字符所对应的下标
        # print(text)
        # print(length)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)  # timestep * batchsize  preds.size: 41 x 16 x 2018
        loss = criterion(preds, text, preds_size, length)  # text size is sum(length)
        # print(preds.size(2))
        # print(text.size(0))
        # print(sum(length))
        # print(preds_size)    # 41
        # print("length")
        # print(length)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inp.size(0))

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=inp.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            print(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.avg, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()


def validate(config, val_loader, dataset, converter, model, criterion, device, epoch, writer_dict, output_dict):

    losses = AverageMeter()
    model.eval()

    n_correct = 0
    wer_list = []
    # print(val_loader)
    with torch.no_grad():
        for i, (inp, idx) in enumerate(val_loader):

            labels = utils.get_batch_label(dataset, idx)
            # print(labels)
            inp = inp.to(device)
            # print(inp)
            # print(idx)

            # inference
            preds = model(inp).cpu()

            # compute loss
            batch_size = inp.size(0)
            text, length = converter.encode(labels)
            # print(text)
            # print(length)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            # print(preds_size)
            loss = criterion(preds, text, preds_size, length)

            losses.update(loss.item(), inp.size(0))

            _, preds = preds.max(2)
            # print(_)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            # print(preds.data)    # [0, 0, 0, ...]
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            # print(sim_preds)    # '' '' '' ...

            for pred, target in zip(sim_preds, labels):
                maxlen = max(len(pred), len(target))
                wer_list.append(1 - wer(pred, target)/maxlen)
                pred = ''.join(pred)
                target = ''.join(target)
                if pred == target:
                    n_correct += 1




            if (i + 1) % config.PRINT_FREQ == 0:
                print('Epoch: [{0}][{1}/{2}]'.format(epoch, i, len(val_loader)))

            if i == config.TEST.NUM_TEST_BATCH:
                break

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:config.TEST.NUM_TEST_DISP]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    num_test_sample = config.TEST.NUM_TEST_BATCH * config.TEST.BATCH_SIZE_PER_GPU
    if num_test_sample > len(dataset):
        num_test_sample = len(dataset)

    print("[#correct:{} / #total:{}]".format(n_correct, num_test_sample))
    accuracy = n_correct / float(num_test_sample)
    print('Test loss: {:.4f}, accuray: {:.4f}'.format(losses.avg, accuracy))
    print('Accuray based on WER: {:.4f}'.format(np.mean(wer_list)))

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_acc', accuracy, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return accuracy


def wer(s1, s2):

    d = np.zeros([len(s1) + 1, len(s2) + 1])
    d[:, 0] = np.arange(len(s1)+1)
    d[0, :] = np.arange(len(s2)+1)

    for j in range(1, len(s2)+1):
        for i in range(1, len(s1)+1):
            if s1[i-1] == s2[j-1]:
                d[i, j] = d[i-1, j-1]
            else:
                d[i, j] = min(d[i-1, j]+1, d[i, j-1]+1, d[i-1, j-1]+1)

    return d[-1, -1]