import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from dataset import DollarDataset
from model import IncomeNet


class AverageMeter(object):
    def __init__(self):
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

def train_epoch(model, dataset, optimizer, criterion=nn.MSELoss()):
    model.train()
    
    step = 0
    avg_mse = AverageMeter()
    iter_size = 1

    for data in dataset:
        inputs, targets = data
        inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())

        outputs = model(inputs)
        
        loss = criterion(outputs, targets) / iter_size
        loss.backward()

        if step % iter_size == 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_mse.update(loss.data[0], inputs.data.size(0))

        if step % 100 == 0:
            print("Step:{}\t Loss:{:1.3f}\tAverage MSE:{:1.3f}".format(
                step, loss.data[0] * iter_size, avg_mse.avg))

        step += 1

    return avg_mse.avg


def val_epoch(model, dataset, criterion=nn.MSELoss()):
    model.eval()

    avg_mse = AverageMeter()

    for data in dataset:
        inputs, targets = data
        inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        avg_mse.update(loss.data[0], inputs.data.size(0))

    print("Valid Average MSE loss:{:1.3f}".format(avg_mse.avg))

    return avg_mse.avg


def adjust_lr(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
    print("Adjust Learning Rate By {:.2f}".format(decay_rate))


def main():
    global args

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--root_dir', default='./data', type=str, help='dataset root dir')
    parser.add_argument('--train_json', default='train_set.json', type=str, help='train json path')
    parser.add_argument('--valid_json', default='valid_set.json', type=str, help='valid json path')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate,default is 1e-3')
    parser.add_argument('--checkpoint_path', default=None, type=str, help='ckpt path')
    parser.add_argument('--train_dir', default='./checkpoints', type=str, help='ckpt path')
    parser.add_argument('--decay_rate', default=0.5, type=float, help='learning rate decay')
    parser.add_argument('--decay_milestones', default='10,15', type=str, help='lr decay policy')
    parser.add_argument('--num_epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size default is 32')
    parser.add_argument('--valid_batch_size', default=32, type=int, help='Validation batch size default is 32')
    parser.add_argument('--global_step', default=0, type=int, help='Global step of the model')
    parser.add_argument('--img_size', default=300, type=int, help='image size')
    parser.add_argument('--device', default='0', type=str, help='assign a GPU device')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    decay_milestones = [int(m) for m in args.decay_milestones.split(',')]
    ending_lr = 1e-5  # The minimal learning rate during training
    
    model = IncomeNet(pretrained=args.checkpoint_path)
    if torch.cuda.is_available():
        model = model.cuda()

    dataset = DollarDataset(base_path=args.db_root_dir,
                            train_json=args.train_json,
                            valid_json=args.valid_json,
                            batch_size=args.batch_size,
                            valid_batch_size=args.valid_batch_size,
                            img_size=args.img_size)

    if args.checkpoint_path is None:
        print('Phase 1: Training final layer')
        optimizer = optim.Adam([model.base.fc], lr=0.001, weight_decay=1e-4)
        train_loss, train_acc, train_acc_thresh = train_epoch(model, dataset.get_train_loader(), optimizer)
        print("Training loss:{:1.3f}\taccuracy:{:1.3f}\taccuracy_thresh:{:1.3f}".format(train_loss, train_acc, train_acc_thresh))
        print()
        print('Phase 2: Traning the whole network')


    optimizer = optim.Adam([model.base.fc, model.base.layer4], lr=args.learning_rate, weight_decay=1e-4)
    global_step = args.global_step
    cur_lr = args.learning_rate
    best_loss = 100000
    t0 = time.time()
    for epoch in range(args.num_epochs):
        cur_lr = optimizer.state_dict()['param_groups'][0]['lr']

        if epoch in decay_milestones:
            if cur_lr > ending_lr:
                adjust_lr(optimizer, args.decay_rate)
        

        dataset_train = dataset.get_train_loader()
        dataset_valid = dataset.get_valid_loader()

        train_loss = train_epoch(model, dataset_train, optimizer)
        print("Training loss:{:1.3f}".format(train_loss))
        
        valid_loss = val_epoch(model, dataset_valid) 
        
        save_path = os.path.join(args.checkpoint_path, 'model_wts_%d.pth' % (global_step))
        if valid_loss < best_loss:
            best_loss = valid_loss + 0.5
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            torch.save(model, save_path)
            print("Model saved in %s" % save_path)

        global_step += len(dataset_train) // args.batch_size

        duration = time.time() - t0
        t0 = time.time()
        print("Epoch {}:\tin {} min {:1.2f} sec".format(epoch, duration // 60, duration % 60))

if __name__ == '__main__':
    main()
