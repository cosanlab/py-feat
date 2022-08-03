# Implements different statistical learning algorithms to classify Emotions
# Please see https://www.cl.cam.ac.uk/~mmam3/pub/FG2015.pdf for more details and reasons
# Currently support: SVM (as in the paper), RandomForest (new implementation).
import numpy as np
from feat.utils.io import get_resource_path
import joblib
import os
import torch.nn as nn
import torch.nn.functional as F
import math


def load_classifier(cf_path):
    clf = joblib.load(cf_path)
    return clf


class EmoSVMClassifier:
    def __init__(self) -> None:
        self.pca_model = load_classifier(
            os.path.join(get_resource_path(), "emo_hog_pca.joblib")
        )
        self.classifier = load_classifier(
            os.path.join(get_resource_path(), "emoSVM38.joblib")
        )
        self.scaler = load_classifier(
            os.path.join(get_resource_path(), "emo_hog_scalar.joblib")
        )

    def detect_emo(self, frame, landmarks):
        """
        Note that here frame is represented by hogs
        """
        # landmarks = np.array(landmarks)
        # landmarks = landmarks.reshape(landmarks.shape[0]*landmarks.shape[1],landmarks.shape[2],landmarks.shape[3])
        # landmarks = landmarks.reshape(-1,landmarks.shape[1]*landmarks.shape[2])
        landmarks = np.concatenate(landmarks)
        landmarks = landmarks.reshape(-1, landmarks.shape[1] * landmarks.shape[2])

        pca_transformed_frame = self.pca_model.transform(
            self.scaler.fit_transform(frame)
        )
        feature_cbd = np.concatenate((pca_transformed_frame, landmarks), 1)
        pred_emo = []
        for keys in self.classifier:
            emo_pred = self.classifier[keys].predict(feature_cbd)
            emo_pred = emo_pred  # probably need to delete this
            pred_emo.append(emo_pred)

        pred_emos = np.array(pred_emo).T
        return pred_emos


class HOGLayer(nn.Module):
    def __init__(
        self,
        nbins=10,
        pixels_per_cell=8,
        cells_per_block=2,
        max_angle=math.pi,
        stride=1,
        padding=1,
        dilation=1,
        transform_sqrt=False,
        block_normalization=None,
        feature_vector=True,
        device="auto",
    ):
        """Pytorch Model to extract HOG features. Designed to be similar to skimage.feature.hog.

        Based on https://gist.github.com/etienne87/b79c6b4aa0ceb2cff554c32a7079fa5a

        Args:
            orientations (int): Number of orientation bins.
            pixels_per_cell (int, int): Size (in pixels) of a cell.
            transform_sqrt (bool): Apply power law compression to normalize the image before processing.
                                    DO NOT use this if the image contains negative values.
            block_normalization (str): Block normalization method:
                                    ``L1``
                                       Normalization using L1-norm.
                                    ``L1-sqrt``
                                       Normalization using L1-norm, followed by square root.
                                    ``L2``
                                       Normalization using L2-norm.
                                    ``L2-Hys``
                                       Normalization using L2-norm, followed by limiting the
                                       maximum values to 0.2 (`Hys` stands for `hysteresis`) and
                                       renormalization using L2-norm. (default)
            feature_vector (bool): Return as a feature vector
            device (str): device to execute code. can be ['auto', 'cpu', 'cuda', 'mps']

        """

        super(HOGLayer, self).__init__()
        self.nbins = nbins
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.max_angle = max_angle
        self.transform_sqrt = transform_sqrt
        self.device = set_torch_device(device)
        self.feature_vector = feature_vector
        if block_normalization is not None:
            self.block_normalization = block_normalization.lower()
        else:
            self.block_normalization = block_normalization

        # Construct a Sobel Filter
        mat = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        mat = torch.cat((mat[None], mat.t()[None]), dim=0)
        self.register_buffer("weight", mat[:, None, :, :])
        self.cell_pooler = nn.AvgPool2d(
            pixels_per_cell,
            stride=pixels_per_cell,
            padding=0,
            ceil_mode=False,
            count_include_pad=True,
        )

    def forward(self, img):
        with torch.no_grad():

            img = img.to(self.device)

            # 1. Global Normalization. The first stage applies an optional global
            # image normalization equalisation that is designed to reduce the influence
            # of illuminationeffects. In practice we use gamma (power law) compression,
            # either computing the square root or the log of each color channel.
            # Image texture strength is typically proportional to the local surface
            # illumination so this compression helps to reduce the effects of local
            # shadowing and illumination variations.
            if self.transform_sqrt:
                img = img.sqrt()

            # 2. Compute Gradients. The second stage computes first order image gradients.
            # These capture contour, silhouette and some texture information,
            # while providing further resistance to illumination variations.
            gxy = F.conv2d(
                img,
                self.weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=1,
            )

            # 3. Binning Mag with linear interpolation. The third stage aims to produce
            # an encoding that is sensitive to local image content while remaining
            # resistant to small changes in pose or appearance. The adopted method pools
            # gradient orientation information locally in the same way as the SIFT
            # [Lowe 2004] feature. The image window is divided into small spatial regions,
            # called "cells". For each cell we accumulate a local 1-D histogram of gradient
            # or edge orientations over all the pixels in the cell. This combined
            # cell-level 1-D histogram forms the basic "orientation histogram" representation.
            # Each orientation histogram divides the gradient angle range into a fixed
            # number of predetermined bins. The gradient magnitudes of the pixels in the
            # cell are used to vote into the orientation histogram.
            mag = gxy.norm(dim=1)
            norm = mag[:, None, :, :]
            phase = torch.atan2(gxy[:, 0, :, :], gxy[:, 1, :, :])
            phase_int = phase / self.max_angle * self.nbins
            phase_int = phase_int[:, None, :, :]

            n, c, h, w = gxy.shape
            out = torch.zeros(
                (n, self.nbins, h, w), dtype=torch.float, device=self.device
            )
            print(out.shape)
            out.scatter_(1, phase_int.floor().long() % self.nbins, norm)
            out.scatter_add_(1, phase_int.ceil().long() % self.nbins, 1 - norm)
            out = self.cell_pooler(out)

            # 4. Compute Normalization. The fourth stage computes normalization,
            # which takes local groups of cells and contrast normalizes their overall
            # responses before passing to next stage. Normalization introduces better
            # invariance to illumination, shadowing, and edge contrast. It is performed
            # by accumulating a measure of local histogram "energy" over local groups
            # of cells that we call "blocks". The result is used to normalize each cell
            # in the block. Typically each individual cell is shared between several
            # blocks, but its normalizations are block dependent and thus different.
            # The cell thus appears several times in the final output vector with
            # different normalizations. This may seem redundant but it improves the
            # performance. We refer to the normalized block descriptors as Histogram
            # of Oriented Gradient (HOG) descriptors.
            #
            # Note: we can probably find a way to vectorize this loop at some point
            if self.block_normalization is not None:
                n_batch, n_channel, s_row, s_col = img.shape
                c_row, c_col = [self.pixels_per_cell] * 2
                b_row, b_col = [self.cells_per_block] * 2
                n_cells_row = int(s_row // c_row)  # number of cells along row-axis
                n_cells_col = int(s_col // c_col)  # number of cells along col-axis
                n_blocks_row = (
                    n_cells_row - b_row
                ) + 1  # number of blocks along row-axis
                n_blocks_col = (
                    n_cells_col - b_col
                ) + 1  # number of blocks along col-axis

                hog_out = torch.zeros(
                    (
                        n_batch,
                        nbins,
                        cells_per_block,
                        cells_per_block,
                        n_blocks_row,
                        n_blocks_col,
                    ),
                    dtype=torch.float,
                    device=self.device,
                )
                for r in range(n_blocks_row):
                    for c in range(n_blocks_col):
                        hog_out[:, :, :, :, r, c] = self._normalize_block(
                            out[:, :, r : r + b_row, c : c + b_col],
                            self.block_normalization,
                        )
                out = hog_out

            if self.feature_vector:
                return out.flatten(start_dim=1)
            else:
                return out

    def _normalize_block(self, block, method, eps=1e-5):
        """helper function to perform normalization"""

        if method == "l1":
            out = torch.divide(block, block.abs().sum() + eps)
        elif method == "l1-sqrt":
            out = torch.divide(block, block.abs().sum() + eps).sqrt()
        elif method == "l2":
            out = torch.divide(block, (torch.square(block).sum() + eps**2).sqrt())
        elif method == "l2-hys":
            out = torch.divide(block, (torch.square(block).sum() + eps**2).sqrt())
            out = torch.minimum(out, torch.ones(out.shape) * 0.2)
            out = torch.divide(out, (torch.square(out).sum() + eps**2).sqrt())
        else:
            raise ValueError(
                'Selected block normalization method is invalid. Use ["l1","l1-sqrt","l2","l2-hys"]'
            )
        return out
