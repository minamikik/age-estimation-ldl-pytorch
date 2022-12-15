import argparse
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pretrainedmodels
import pretrainedmodels.utils
from age_model import get_model
from age_dataset import FaceDataset
from age_defaults import _C as cfg
from datetime import datetime
from matplotlib import pyplot as plt
import loss as L
from types import MethodType

def get_args():
    model_names = sorted(name for name in pretrainedmodels.__dict__
                         if not name.startswith("__")
                         and name.islower()
                         and callable(pretrainedmodels.__dict__[name]))
    parser = argparse.ArgumentParser(description=f"available models: {model_names}",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str,
                        required=True, help="Data root directory")
    parser.add_argument("--dataset", type=str,
                        required=True, help="Dataset name")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint if any")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoint", help="Checkpoint directory")
    parser.add_argument('--multi_gpu', action="store_true",
                        help="Use multi GPUs (data parallel)")
    parser.add_argument('--expand', type=float, default=0, help="expand the crop area by a factor, typically between 0 and 1")
    parser.add_argument('--aug', action="store_true",
                        help="Apply data augmentation")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line, e.g. MODEL.ARCH vgg16_bn")
    args = parser.parse_args()
    return args


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


##############################################################
# Substitute original methods

def features(self, x):
    x = self.layer0(x)
    x = self.layer1(x)
    x = self.layer2(x)
    y = x # y.shape = (batch,512,28,28)
    x = self.layer3(x)
    x = self.layer4(x)
    return x, y
    
def logits(self, x ,y):
    y = self.gen_eth_avgpool(y)
    y = y.squeeze(-1)
    y = y.squeeze(-1) # y.shape = (batch,512)
    # gender
    g = self.gen_fc(y)
    # ethnicity
    e = self.eth_fc(y)
    # age estimation
    x = self.avg_pool(x)
    if self.dropout is not None:
        x = self.dropout(x)
    x = x.view(x.size(0), -1)
    fusion = torch.cat((x, g, e), 1)
    a = self.last_linear(fusion)
    return a, g, e

def forward(self, x):
    x, y = self.features(x)
    return self.logits(x, y)
##############################################################


def multitask(model): # Add gender and race prediction to the original model
    model.gen_eth_avgpool = nn.AdaptiveAvgPool2d(1)
    model.gen_fc = nn.Linear(512, 2)
    model.eth_fc = nn.Linear(512, 5)
    model.last_linear = nn.Linear(2055, 101)
    model.features = MethodType(features, model)
    model.forward = MethodType(forward, model)
    model.logits = MethodType(logits, model)

def train(train_loader, model, criterion, optimizer, epoch, device, train_count):
    model.train()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()
    rank = torch.Tensor([i for i in range(101)]).to(device)
    correct_gender = 0
    correct_race = 0

    with tqdm(train_loader) as _tqdm:
        for x, y, lbl, g, r in _tqdm:
            x = x.to(device)
            y = y.to(device)
            lbl = lbl.to(device)
            g = g.to(device)
            r = r.to(device)

            # compute output
            outputs, gen, race = model(x)
            outputs = F.softmax(outputs, dim = 1)
            ages = torch.sum(outputs*rank, dim=1)

            # calc loss
            # loss = criterion(outputs, y)
            loss1 = L.kl_loss(outputs, lbl)
            loss2 = L.L1_loss(ages, y)
            loss3 = criterion(gen, g)
            loss4 = criterion(race, r)
            loss = loss1 + loss2 + loss3 + loss4
            cur_loss = loss.item()

            # calc accuracy
            correct_num = (abs(ages - y) < 1).sum().item()
            gen = F.softmax(gen, dim = 1)
            race = F.softmax(race, 1)
            correct_gender += torch.sum(torch.argmax(gen, 1) == g).item()
            correct_race += torch.sum(torch.argmax(race, 1) == r).item()

            # measure accuracy and record loss
            sample_num = x.size(0)
            loss_monitor.update(cur_loss, sample_num)
            accuracy_monitor.update(correct_num, sample_num)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _tqdm.set_postfix(OrderedDict(stage="train", epoch=epoch, loss=loss_monitor.avg),
                              acc=accuracy_monitor.avg, correct=correct_num, sample_num=sample_num)
    print("Gender accu:", correct_gender/train_count)
    print("Race accu:", correct_race/train_count)

    return loss_monitor.avg, accuracy_monitor.avg


def validate(validate_loader, model, criterion, epoch, device, val_count, get_ca=False):
    model.eval()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()
    preds = [] 
    gt = []
    rank = torch.Tensor([i for i in range(101)]).to(device)
    ca = None
    correct_gender = 0
    correct_race = 0
    if get_ca:
        ca = {3:0, 5:0, 7:0}
    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for i, (x, y, lbl, g, r) in enumerate(_tqdm):
                x = x.to(device)
                y = y.to(device)
                lbl = lbl.to(device)
                g = g.to(device)
                r = r.to(device)

                # compute output
                outputs, gen, race = model(x)
                outputs = F.softmax(outputs, dim = 1)
                ages = torch.sum(outputs*rank, dim=1)  # age expectation
                preds.append(ages.cpu().numpy())  # append predicted age
                gt.append(y.cpu().numpy())  # append real age

                for ind, age in enumerate(ages): 
                    if ca is not None:
                        if abs(y[ind].item() - age) < 3:
                            ca[3] += 1
                        if abs(y[ind].item() - age) < 5:
                            ca[5] += 1
                        if abs(y[ind].item() - age) < 7:
                            ca[7] += 1
                        

                # valid for validation, not used for test
                if criterion is not None:
                    # calc loss
                    loss1 = L.kl_loss(outputs, lbl)
                    loss2 = L.L1_loss(ages, y)
                    loss3 = criterion(gen, g)
                    loss4 = criterion(race, r)
                    loss = loss1 + loss2 + loss3 + loss4
                    cur_loss = loss.item()

                    # calc accuracy
                    correct_num = (abs(ages - y) < 1).sum().item()
                    
                    gen = F.softmax(gen, dim = 1)
                    correct_gender += torch.sum(torch.argmax(gen, 1) == g).item()
                    race = F.softmax(race, 1)
                    correct_race += torch.sum(torch.argmax(race, 1) == r).item()

                    # measure accuracy and record loss
                    sample_num = x.size(0)
                    loss_monitor.update(cur_loss, sample_num)
                    accuracy_monitor.update(correct_num, sample_num)
                    _tqdm.set_postfix(OrderedDict(stage="val", epoch=epoch, loss=loss_monitor.avg),
                                      acc=accuracy_monitor.avg, correct=correct_num, sample_num=sample_num)
 
    preds = np.concatenate(preds, axis=0)
    gt = np.concatenate(gt, axis=0)
    mae = np.abs(preds - gt).mean()
    print("Gender accu:", correct_gender/val_count)
    print("Race accu:", correct_race/val_count)
    if ca is not None:
        for i in ca.keys():
            ca[i] = ca[i] / val_count
        print("\n")
        print("CA3: {:.2f} CA5: {:.2f} CA7: {:2f}".format(ca[3] * 100, ca[5]*100, ca[7]*100))
    
    return loss_monitor.avg, accuracy_monitor.avg, mae, ca


def main():
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    start_epoch = 0
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # create model
    print("=> creating model '{}'".format(cfg.MODEL.ARCH))
    model = get_model(model_name=cfg.MODEL.ARCH)
    multitask(model)

    if cfg.TRAIN.OPT == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # optionally resume from a checkpoint
    resume_path = args.resume

    if resume_path:
        if Path(resume_path).is_file():
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location="cpu")
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(resume_path))

    if args.multi_gpu:
        model = nn.DataParallel(model)

    if device == "cuda":
        cudnn.benchmark = True
    
    get_ca = True if "megaage" in args.dataset.lower() else True # display cummulative acuracy 
    value_ca = True if "megaage" in args.dataset.lower() else False # use CA to update saved model
    if get_ca:
        print("Cummulative Accuracy will be calculated for", args.dataset)
    if value_ca:
        print("Cummulative Accuracy will be compared to update saved model")

    criterion = nn.CrossEntropyLoss().to(device)
    train_dataset = FaceDataset(args.data_dir, "train", args.dataset, img_size=cfg.MODEL.IMG_SIZE, augment=args.aug,
                                age_stddev=cfg.TRAIN.AGE_STDDEV, label=True, expand= args.expand, gender=True)
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.TRAIN.WORKERS, drop_last=False)

    val_dataset = FaceDataset(args.data_dir, "valid", args.dataset,
                              img_size=cfg.MODEL.IMG_SIZE, augment=False, label=True, expand= args.expand, gender=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.TRAIN.WORKERS, drop_last=False)

    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.LR_DECAY_RATE,
                       last_epoch=start_epoch - 1)
    best_val_mae = 10000.0
    train_writer = None
    global_ca = {3: 0.0, 5: 0.0, 7: 0.0}
    train_count = len(train_dataset)
    val_count = len(val_dataset)

    all_train_loss = []
    all_train_accu = []
    all_val_loss = []
    all_val_accu = []

    # range(start_epoch, cfg.TRAIN.EPOCHS):
    for epoch in range(cfg.TRAIN.EPOCHS):
        # train
        train_loss, train_acc = train(
            train_loader, model, criterion, optimizer, epoch, device, train_count)

        # validate
        val_loss, val_acc, val_mae, new_ca= validate(
            val_loader, model, criterion, epoch, device, val_count, get_ca)

        all_train_loss.append(float(train_loss))
        all_train_accu.append(float(train_acc))
        all_val_loss.append(float(val_loss))
        all_val_accu.append(float(val_mae))

        # checkpoint
        if ((not value_ca) and (val_mae < best_val_mae)) or ((get_ca and value_ca) and (new_ca[3] > global_ca[3])):
            print(
                f"=> [epoch {epoch:03d}] best val mae was improved from {best_val_mae:.3f} to {val_mae:.3f}")
            model_state_dict = model.module.state_dict(
            ) if args.multi_gpu else model.state_dict()
            torch.save(
                {
                    'epoch': epoch + 1,
                    'arch': cfg.MODEL.ARCH,
                    'state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict()
                },
                str(checkpoint_dir.joinpath("epoch{:03d}_{}_{:.5f}_{:.4f}_{}_{}_ldl.pth".format(
                    epoch, args.dataset, val_loss, val_mae, datetime.now().strftime("%Y%m%d"), cfg.MODEL.ARCH)))
            )
            best_val_mae = val_mae
            best_checkpoint = str(checkpoint_dir.joinpath("epoch{:03d}_{}_{:.5f}_{:.4f}_{}_{}_ldl.pth".format(epoch, args.dataset, val_loss, val_mae, datetime.now().strftime("%Y%m%d"), cfg.MODEL.ARCH)))
            if get_ca:
                global_ca = new_ca
        else:
            print(
                f"=> [epoch {epoch:03d}] best val mae was not improved from {best_val_mae:.3f} ({val_mae:.3f})")

        # adjust learning rate
        scheduler.step()

    print("=> training finished")
    print(f"additional opts: {args.opts}")
    print(f"best val mae: {best_val_mae:.3f}")
    if get_ca:
        print("CA3: {:.2f} CA5: {:.2f} CA7: {:.2f}".format(global_ca[3] * 100, global_ca[5]*100, global_ca[7]*100))
    print("best mae saved model:", best_checkpoint)

    x = np.arange(cfg.TRAIN.EPOCHS)
    plt.xlabel("Epoch")

    plt.ylabel("Train Loss")
    plt.plot(x, all_train_loss)
    plt.savefig("savefig/{}_{}_{}_train_loss.png".format(args.dataset,
                                                         cfg.MODEL.ARCH, datetime.now().strftime("%Y%m%d")))
    plt.clf()

    plt.ylabel("Train Accuracy")
    plt.plot(x, all_train_accu)
    plt.savefig("savefig/{}_{}_{}_train_accu.png".format(args.dataset,
                                                         cfg.MODEL.ARCH, datetime.now().strftime("%Y%m%d")))
    plt.clf()

    plt.ylabel("Validation Loss")
    plt.plot(x, all_val_loss)
    plt.savefig("savefig/{}_{}_{}_val_loss.png".format(args.dataset,
                                                       cfg.MODEL.ARCH, datetime.now().strftime("%Y%m%d")))
    plt.clf()

    plt.ylabel("Validation Accuracy")
    plt.plot(x, all_val_accu)
    plt.savefig("savefig/{}_{}_{}_val_mae.png".format(args.dataset,
                                                      cfg.MODEL.ARCH, datetime.now().strftime("%Y%m%d")))


if __name__ == '__main__':
    main()