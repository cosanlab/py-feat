from feat.emo_detectors.ferNet.ferNet_model import fer_net
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
from feat.utils import get_resource_path, face_rect_to_coords
import os 
from torchvision import transforms

class ferNetModule(nn.Module):
    def __init__(self) -> None: 
        self.pretrained_path = os.path.join(get_resource_path(), 'best_ferModel.pth')
    
    def detect_emo(self, imgs, detected_face, img_w = 48, img_h = 48):
        
        img_pil = Image.fromarray(imgs)
        grayscale_image = ImageOps.grayscale(img_pil)
        grayscale_cropped_face = grayscale_image.crop(face_rect_to_coords(detected_face[0]))
        grayscale_cropped_resized_face = grayscale_cropped_face.resize((img_w, img_h))
        grayscale_cropped_resized_reshaped_face = np.array(grayscale_cropped_resized_face).reshape(img_w, img_h)

        #n_chs = np.min(grayscale_cropped_resized_reshaped_face.shape)
        szht = np.max(grayscale_cropped_resized_reshaped_face.shape)

        use_gpu = torch.cuda.is_available()
        net0 = fer_net(in_chs=1,num_classes=7,img_size=szht)
        if use_gpu:
            net0 = net0.cuda()
        
        if use_gpu:
            net0.load_state_dict(torch.load(self.pretrained_path))
        else:
            net0.load_state_dict(torch.load(self.pretrained_path,map_location={'cuda:0': 'cpu'}))
        net0.eval()
        imgs_net = Image.fromarray(grayscale_cropped_resized_reshaped_face)
        imgs_net = transforms.ToTensor()(imgs_net).unsqueeze_(0)
        if use_gpu:
            imgs_net = imgs_net.cuda()
        pred_emo = net0(imgs_net)
        pred_emo_softmax = nn.functional.softmax(pred_emo).cpu().float().data.numpy()

        return pred_emo_softmax


