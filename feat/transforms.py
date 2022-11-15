"""
Custom transforms for torch.Datasets
"""

from torchvision.transforms import Compose, Resize, Pad
import numpy as np


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
                                    matched to output_size. If int, will set largest edge
                                    to output_size if target size is bigger,
                                    or smallest edge if target size is smaller
                                    to keep aspect ratio the same.
        preserve_aspect_ratio (bool): Output size is matched to preserve aspect ratio.
                                    Note that longest edge of output size is preserved,
                                    but actual output may differ from intended output_size.
        padding (bool): Transform image to exact output_size. If tuple,
                        will preserve aspect ratio by adding padding.
                        If int, will set both sides to the same size.

    Returns:
        dict: {'Image':transformed tensor, 'Scale':image scaling for transformation}

    """

    def __init__(self, output_size, preserve_aspect_ratio=True, padding=False):

        if not isinstance(output_size, (int, tuple)):
            raise ValueError(
                f"output_size must be (int, tuple) not {type(output_size)}."
            )

        self.output_size = output_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.padding = padding

    def __call__(self, image):

        height, width = image.shape[-2:]

        if isinstance(self.output_size, int):
            scale = self.output_size / max(height, width)
            new_height, new_width = (scale * np.array([height, width])).astype(int)
        else:
            scale = max(self.output_size) / max(height, width)
            new_height, new_width = np.array(self.output_size).astype(int)

        if self.preserve_aspect_ratio or self.padding:

            # Calculate Scaling Value
            if isinstance(self.output_size, int):
                scale = self.output_size / max(height, width)
            else:
                if (
                    new_height >= height & new_width >= width
                ):  # output size is bigger than image
                    if height > width:
                        scale = new_height / height
                    else:
                        scale = new_width / width
                else:  # output size is smaller than image
                    if (height > new_height) & (width <= new_width):
                        scale = new_height / height
                    elif (width > new_width) & (height <= new_height):
                        scale = new_width / width
                    else:
                        if height > width:
                            scale = new_height / height
                        else:
                            scale = new_width / width

            # Compute new height and width
            if isinstance(self.output_size, int):
                new_height = int(height * scale)
                new_width = int(width * scale)
            else:
                new_height, new_width = (scale * np.array([height, width])).astype(int)

            if self.padding:
                if isinstance(self.output_size, int):
                    output_height, output_width = (self.output_size, self.output_size)
                else:
                    output_height, output_width = self.output_size

                if new_height < output_height:
                    padding_height = output_height - new_height
                    if (padding_height) % 2 == 0:
                        padding_top, padding_bottom = [int(padding_height / 2)] * 2
                    else:
                        padding_top, padding_bottom = (
                            padding_height // 2,
                            1 + (padding_height // 2),
                        )
                else:
                    padding_top, padding_bottom = (0, 0)
                if new_width < output_width:
                    padding_width = output_width - new_width
                    if (padding_width) % 2 == 0:
                        padding_left, padding_right = [int(padding_width / 2)] * 2
                    else:
                        padding_left, padding_right = (
                            padding_width // 2,
                            1 + (padding_width // 2),
                        )
                else:
                    padding_left, padding_right = (0, 0)

        if self.padding:
            padding_dict = {
                "Left": int(padding_left),
                "Top": int(padding_top),
                "Right": int(padding_right),
                "Bottom": int(padding_bottom),
            }
            transform = Compose(
                [
                    Resize((int(new_height), int(new_width))),
                    Pad(
                        (
                            padding_dict["Left"],
                            padding_dict["Top"],
                            padding_dict["Right"],
                            padding_dict["Bottom"],
                        )
                    ),
                ]
            )
        else:
            transform = Compose([Resize((int(new_height), int(new_width)))])
            padding_dict = {"Left": 0, "Top": 0, "Right": 0, "Bottom": 0}
        
        return {"Image": transform(image), "Scale": scale, "Padding": padding_dict}
