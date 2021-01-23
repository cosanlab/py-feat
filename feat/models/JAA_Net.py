import os
import torch.optim as optim
import torch.utils.data as util_data
import pandas as pd
import numpy as np
import network
import pre_process as prep
from util import *
from data_list import ImageList
import random
from PIL import Image

def make_dataset(master_file, land, biocular, row_idx):

    aus = master_file[["1","2","4","6","7","10","12","14","15","17","23","24"]]
    aus = aus.to_numpy(dtype='int')
    #len_ = master_file.shape[0]
    images = []
    for idx in row_idx:
        #img_filename = img_dir + "aligned\\" + master_file['Subject'][idx] + "\\" + master_file['Task'][idx] + "\\" + str(master_file['Number'][idx]) + ".jpg"
        img_filename = master_file['path'][idx]
        land_k = land[idx,:]
        biocular_k = biocular[idx]
        au_k = aus[idx,:]
        images.append((img_filename,land_k,biocular_k,au_k))

    return images

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # Image value: [0,1]
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)

class ImageList(object):

    def __init__(self, crop_size, img_dir, master_file_path, aligned_land_path, biocular_path, row_idx, 
                phase='train', transform=None, target_transform=None, loader=default_loader):
        
        master_file = pd.read_csv(master_file_path)
        
        with open(aligned_land_path, 'rb') as fp:
            land = pickle.load(fp)
        with open(biocular_path,'rb') as ffp:
            biocular = pickle.load(ffp)

        #biocular = np.loadtxt(biocular_path)
        imgs = make_dataset(img_dir, master_file, land, biocular, row_idx)

        if len(imgs) == 0:
            raise (RuntimeError('Found 0 images in subfolders of: ' + path + '\n'))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.crop_size = crop_size
        self.phase = phase

    def __getitem__(self, index):

        path, land, biocular, au = self.imgs[index]
        img = self.loader(path) # Loader ok
        if self.phase == 'train':
            w, h = img.size
            offset_y = random.randint(0, h - self.crop_size)
            offset_x = random.randint(0, w - self.crop_size)

            flip = random.randint(0, 1)

            if self.transform is not None:
                img = self.transform(img, flip, offset_x, offset_y)
            if self.target_transform is not None:
                land = self.target_transform(land, flip, offset_x, offset_y)
        # for testing
        else:
            w, h = img.size
            offset_y = (h - self.crop_size)/2
            offset_x = (w - self.crop_size)/2
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                land = self.target_transform(land, 0, offset_x, offset_y)

        return img, land, biocular, au

    def __len__(self):
        return len(self.imgs)

optim_dict = {'SGD': optim.SGD, 'Adam': optim.Adam}



def main():
    ## set loss criterion
    use_gpu = torch.cuda.is_available()
    master_file = pd.read_csv(config_gen_path+"true_marked.csv")
    ## prepare data
    dsets = {}
    dset_loaders = {}
    test_idx = np.arange(master_file.shape[0])
    dsets['test'] = ImageList(crop_size=config_crop_size, master_file_path=config_gen_path+"true_marked.csv", aligned_land_path=config_gen_path+'aligned_landmarks.obj',
                               biocular_path=config_gen_path+'biocular.pkl', phase='test', row_idx=test_idx, transform = prep.image_test(crop_size=config_crop_size),
                               target_transform=prep.land_transform(img_size=config_crop_size,
                                                                            flip_reflect=np.loadtxt(
                                                                                config_flip_reflect,delimiter=' ',dtype=np.int)))

    dset_loaders['test'] = util_data.DataLoader(dsets['test'], batch_size=config_eval_batch_size,
                                                shuffle=False, num_workers=config_num_workers)

    ## set network modules
    region_learning = network.network_dict["HMRegionLearning"](input_dim=3, unit_dim = config_unit_dim)
    align_net = network.network_dict["AlignNet"](crop_size=config_crop_size, map_size=config_map_size,
                                                           au_num=config_au_num, land_num=config_land_num,
                                                           input_dim=config_unit_dim*8, fill_coeff=config_fill_coeff)
    local_attention_refine = network.network_dict["LocalAttentionRefine"](au_num=config_au_num, unit_dim=config_unit_dim)
    local_au_net = network.network_dict["LocalAUNetv2"](au_num=config_au_num, input_dim=config_unit_dim*8,
                                                                                     unit_dim=config_unit_dim)
    global_au_feat = network.network_dict["HLFeatExtractor"](input_dim=config_unit_dim*8,
                                                                                     unit_dim=config_unit_dim)
    au_net = network.network_dict["AUNet"](au_num=config_au_num, input_dim = 12000, unit_dim = config_unit_dim)

    if use_gpu:
        region_learning = region_learning.cuda()
        align_net = align_net.cuda()
        local_attention_refine = local_attention_refine.cuda()
        local_au_net = local_au_net.cuda()
        global_au_feat = global_au_feat.cuda()
        au_net = au_net.cuda()

    if not os.path.exists(config_write_path_prefix + config_run_name):
        os.makedirs(config_write_path_prefix + config_run_name)
    if not os.path.exists(config_write_res_prefix + config_run_name):
        os.makedirs(config_write_res_prefix + config_run_name)

    if config_start_epoch <= 0:
        raise (RuntimeError('start_epoch should be larger than 0\n'))

    res_file = open(
        config_write_res_prefix + config_run_name + '/'  + 'offline_AU_pred_' + str(config_start_epoch) + '.txt', 'w')
    region_learning.train(False)
    align_net.train(False)
    local_attention_refine.train(False)
    local_au_net.train(False)
    global_au_feat.train(False)
    au_net.train(False)

    for epoch in range(config_start_epoch, config_n_epochs + 1):
        region_learning.load_state_dict(torch.load(
            config_write_path_prefix + 'region_learning' + '.pth'))
        align_net.load_state_dict(torch.load(
            config_write_path_prefix + 'align_net' + '.pth'))
        local_attention_refine.load_state_dict(torch.load(
            config_write_path_prefix + 'local_attention_refine' + '.pth'))
        local_au_net.load_state_dict(torch.load(
            config_write_path_prefix + 'local_au_net' + '.pth'))
        global_au_feat.load_state_dict(torch.load(
            config_write_path_prefix + 'global_au_feat' + '.pth'))
        au_net.load_state_dict(torch.load(
            config_write_path_prefix + 'au_net' + '.pth'))

        if config_pred_AU:
            local_f1score_arr, local_acc_arr, f1score_arr, acc_arr, mean_error, failure_rate = AU_detection_evalv2(
                dset_loaders['test'], region_learning, align_net, local_attention_refine,
                local_au_net, global_au_feat, au_net, use_gpu=use_gpu)
            print('epoch =%d, local f1 score mean=%f, local accuracy mean=%f, '
                  'f1 score mean=%f, accuracy mean=%f, mean error=%f, failure rate=%f' % (
                  epoch, local_f1score_arr.mean(),
                  local_acc_arr.mean(), f1score_arr.mean(),
                  acc_arr.mean(), mean_error, failure_rate))
            print('%d\t%f\t%f\t%f\t%f\t%f\t%f' % (epoch, local_f1score_arr.mean(),
                                                  local_acc_arr.mean(), f1score_arr.mean(),
                                                  acc_arr.mean(), mean_error, failure_rate), file=res_file)
            
            np.savetxt('F:/PyTorch-JAANet3/'+str(epoch)+'_offline_f1score'+'.txt',f1score_arr,delimiter=',') 

        if config_vis_attention:
            if not os.path.exists(config_write_res_prefix + config_run_name + '/vis_map/' + str(epoch)):
                os.makedirs(config_write_res_prefix + config_run_name + '/vis_map/' + str(epoch))
            if not os.path.exists(config_write_res_prefix + config_run_name + '/overlay_vis_map/' + str(epoch)):
                os.makedirs(config_write_res_prefix + config_run_name + '/overlay_vis_map/' + str(epoch))

            vis_attention(dset_loaders['test'], region_learning, align_net, local_attention_refine,
                              config_write_res_prefix, config_run_name, epoch, use_gpu=use_gpu)

    res_file.close()