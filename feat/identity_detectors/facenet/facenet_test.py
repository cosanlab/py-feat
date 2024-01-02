from feat.utils import set_torch_device
from feat.identity_detectors.facenet.facenet_model import InceptionResnetV1


class Facenet:
    """Facenet Inception Resnet V1 model with optional loading of pretrained weights.

    Model parameters can be loaded based on pretraining on the VGGFace2 or CASIA-Webface
    datasets. Pretrained state_dicts are automatically downloaded on model instantiation if
    requested and cached in the torch cache. Subsequent instantiations use the cache rather than
    redownloading.

    Keyword Arguments:
        pretrained {str} -- Optional pretraining dataset. Either 'vggface2' or 'casia-webface'.
            (default: {None})
        classify {bool} -- Whether the model should output classification probabilities or feature
            embeddings. (default: {False})
        num_classes {int} -- Number of output classes. If 'pretrained' is set and num_classes not
            equal to that used for the pretrained model, the final linear layer will be randomly
            initialized. (default: {None})
        dropout_prob {float} -- Dropout probability. (default: {0.6})
    """

    def __init__(
        self,
        pretrained="vggface2",
        classify=False,
        num_classes=None,
        dropout_prob=0.6,
        device="auto",
    ):
        super().__init__()

        self.model = InceptionResnetV1(
            pretrained=pretrained,
            classify=classify,
            num_classes=num_classes,
            dropout_prob=dropout_prob,
            device=device,
        )

        self.device = set_torch_device(device)

        self.model.eval()

    def __call__(self, img):
        """
        img is of shape BxCxHxW --
        """

        return self.model.forward(img)
