from torch.autograd.variable import Variable
from data_loader import image_Loader
from DRML_model import DRML_net
from utils import *
import torch
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim
from lr_scheduler import step_lr_scheduler
import numpy as np
import sys

##################PARAMETERS################################
optim_dict = {'SGD': optim.SGD, 'Adam': optim.Adam}
config_gen_path = "/home/tiankang/JAA_Net3/"
config_imgdir = "/Storage/Data/BP4D+_v0.2/2D+3D/aligned/"
#config_csv_name = "true_marked_30subset.csv"
config_csv_name = "true_marked.csv"
config_write_path = "/home/tiankang/AU_Detections/DRML/outs/"
n_epochs = 251
config_lr_decay_rate = 0.9
config_class_num = 12
config_train_batch = 16
config_test_batch = 16
config_test_every_epoch = 10
config_start_epoch = 80
config_optimizer_type = "SGD"
config_gamma = 0.3
config_init_train_epochs = 12
config_stepsize = 3
config_init_lr = 0.001
############################################################

use_gpu = torch.cuda.is_available()

master_file = pd.read_csv(config_gen_path+"outs/"+config_csv_name)
unique_names = master_file['Subject'].unique()
unique_idxx = np.arange(len(unique_names))
np.random.seed(0)
train_idx = np.random.choice(unique_idxx, int(
    (2/3)*len(unique_names)))  # 3 fold CV
train_master_file = master_file[master_file['Subject'].isin(
    unique_names[train_idx])]
train_master_file.reset_index(drop=True, inplace=True)
test_master_file = master_file[master_file['Subject'].isin(
    unique_names[-train_idx])]
test_master_file.reset_index(drop=True, inplace=True)

au_weight = calculate_AU_weight(master_file)
au_weight = torch.from_numpy(au_weight.astype('float'))

if use_gpu:
    au_weight = au_weight.cuda()

print("AU Weight:", au_weight)
dsets_train = image_Loader(csv_file=train_master_file, img_dir=config_imgdir)
dsets_test = image_Loader(csv_file=test_master_file, img_dir=config_imgdir)
train_set = DataLoader(
    dsets_train, batch_size=config_train_batch, shuffle=False)
test_set = DataLoader(dsets_test, batch_size=config_test_batch, shuffle=False)

net = DRML_net(config_class_num)
counter = 0

opt = optim_dict[config_optimizer_type](net.parameters(
), lr=config_init_lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
param_lr = []
for param_group in opt.param_groups:
    param_lr.append(param_group['lr'])

if use_gpu:
    net = net.cuda()

old_stdout = sys.stdout
log_file = open("/home/tiankang/AU_Detections/DRML/message.log", "w")
sys.stdout = log_file

AU_pred = None
AU_actual = None

for epoch_idx in range(n_epochs):

    if epoch_idx > config_start_epoch and epoch_idx % config_test_every_epoch == 0:
        print('taking snapshot ...')
        torch.save(net.state_dict(), config_write_path +
                   'DRMLNetParams_' + str(epoch_idx) + '.pth')

    if epoch_idx > config_start_epoch and epoch_idx % config_test_every_epoch == 0:
        print("testing:")
        net.train(False)
        f1score, accuracies, matrixA, matrixB = AU_detection_evalv2(
            test_set, net, use_gpu=use_gpu)
        print("F1 Score:", f1score, "accuracies: ", accuracies)
        if AU_actual is None:
            AU_actual = np.expand_dims(matrixA, 0)
        else:
            AU_actual = np.concatenate(
                [AU_actual, np.expand_dims(matrixA, 0)], 0)

        if AU_pred is None:
            AU_pred = np.expand_dims(matrixB, 0)
        else:
            AU_pred = np.concatenate([AU_pred, np.expand_dims(matrixB, 0)], 0)

    for batch_index, (img, label) in enumerate(train_set):
        if counter > 0 and batch_index % 4 == 0:
            print('the number of training iterations is %d' % (counter))
            print('[epoch = %d][iter = %d][loss = %f][loss_au_dice = %f][loss_au_softmax = %f]' % (epoch_idx, batch_index,
                                                                                                   loss.data.cpu().numpy(), loss_au_dice.data.cpu().numpy(), loss_au_softmax.data.cpu().numpy()))
        img = Variable(img)
        label = Variable(label)

        if use_gpu:
            img = img.cuda()
            label = label.cuda()

        if epoch_idx > config_init_train_epochs:
            opt = step_lr_scheduler(
                param_lr, opt, epoch_idx, config_gamma, config_stepsize, config_init_lr)

        opt.zero_grad()
        pred = net(img)
        loss_au_softmax = au_softmax_loss(pred, label, weight=au_weight)
        loss_au_dice = au_dice_loss(pred, label, weight=au_weight)
        loss = loss_au_dice+loss_au_softmax
        loss.backward()
        opt.step()
        counter += 1

sys.stdout = old_stdout
log_file.close()
