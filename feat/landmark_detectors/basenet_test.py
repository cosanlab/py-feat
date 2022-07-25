# Backbone networks used for face landmark detection
# Cunjian Chen (cunjian@msu.edu)

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


# USE global depthwise convolution layer. Compatible with MobileNetV2 (224×224), MobileNetV2_ExternalData (224×224)
class MobileNet_GDConv(nn.Module):
    def __init__(self, num_classes):
        super(MobileNet_GDConv, self).__init__()
        self.pretrain_net = models.mobilenet_v2(pretrained=False)
        self.base_net = nn.Sequential(*list(self.pretrain_net.children())[:-1])
        self.linear7 = ConvBlock(1280, 1280, (7, 7), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(1280, num_classes, 1, 1, 0, linear=True)

    def forward(self, x):
        x = self.base_net(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        return x


# class MobileNet:
#         def __init__(
#         self,
#         num_classes=136,
#         device="auto",

#     ):
#         """Creates an img2pose model. Constrained model is optimized for face detection/ pose estimation for
#         front-facing faces ( [-90, 90] degree range) only. Unconstrained model can detect faces and poses at any angle,
#         but shows slightly dampened performance on face pose estimation.

#         Args:
#             device (str): device to execute code. can be ['auto', 'cpu', 'cuda', 'mps']
#             contrained (bool): whether to run constrained (default) or unconstrained mode

#         Returns:
#             Img2Pose object

#         """

#         self.device = set_torch_device(device)
#         self.model = MobileNet_GDConv(num_classes)

#         model_file =  "mobilenet_224_model_best_gdconv_external.pth.tar"
#         self.load_model(os.path.join(get_resource_path(), model_file))
#         self.model.evaluate()


#     def load_model(self, model_path):
#         """Loads model weights for the mobilenet model
#         Args:
#             model_path (str): file path to saved model weights

#         Returns:
#             None
#         """

#         self.model = torch.nn.DataParallel( self.model)

#         checkpoint = torch.load(model_path, map_location=self.device)

#             self.landmark_detector = self.landmark_detector(136)
#             self.landmark_detector = torch.nn.DataParallel(
#                 self.landmark_detector
#             )
#             checkpoint = torch.load(model_path, map_location=self.device)
#             self.landmark_detector.load_state_dict(checkpoint["state_dict"])


#         self.model.fpn_model.load_state_dict(checkpoint["fpn_model"])


#     def __call__(self, img):
#         """Runs scale_and_predict on each image in the passed image list

#         Args:
#             img_ (np.ndarray): (B,H,W,C), B is batch number, H is image height, W is width and C is channel.

#         Returns:
#             tuple: (faces, poses) - 3D lists (B, F, bbox) or (B, F, face pose) where B is batch/ image number and
#                                     F is face number
#         """

#         img = convert_image_to_tensor(img)
#         img = img.type(torch.float32)
#         img = img.to(self.device)

#         preds = self.scale_and_predict(img)
#         # faces = []
#         # poses = []
#         # for img in img_:
#         #     preds = self.scale_and_predict(img)
#         #     faces.append(preds["boxes"])
#         #     poses.append(preds["poses"])

#         return preds["boxes"], preds["poses"]
