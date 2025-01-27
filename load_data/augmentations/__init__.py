# +
from torchvision import transforms

import torch

import torch.nn as nn
import numpy as np

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
# class GaussianBlur(object):
#     """blur a single image on CPU"""

#     def __init__(self, kernel_size):
#         radias = kernel_size // 2
#         kernel_size = radias * 2 + 1
#         self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
#                                 stride=1, padding=0, bias=False, groups=3)
#         self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
#                                 stride=1, padding=0, bias=False, groups=3)
#         self.k = kernel_size
#         self.r = radias

#         self.blur = nn.Sequential(
#             nn.ReflectionPad2d(radias),
#             self.blur_h,
#             self.blur_v
#         )

#         self.pil_to_tensor = transforms.ToTensor()
#         self.tensor_to_pil = transforms.ToPILImage()

#     def __call__(self, img):
#         img = self.pil_to_tensor(img).unsqueeze(0)

#         sigma = np.random.uniform(0.1, 2.0)
#         x = np.arange(-self.r, self.r + 1)
#         x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
#         x = x / x.sum()
#         x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

#         self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
#         self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

#         with torch.no_grad():
#             img = self.blur(img)
#             img = img.squeeze()

#         img = self.tensor_to_pil(img)

#         return img
def get_transform(transform_type='imagenet', image_size=32, args=None):

    if transform_type == 'imagenet':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        interpolation = args.interpolation
        crop_pct = args.crop_pct

        train_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])
    elif transform_type == '224':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        train_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

    elif transform_type == 'mnist':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
#         train_transform = transforms.Compose([
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.ColorJitter(),
# #             transforms.RandomGrayscale(p=0.2),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=torch.tensor(mean),
#                 std=torch.tensor(std))
#         ])
        
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(28, (0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(0.3, 0.3, 0.15, 0.1)], p=0.5
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])
    else:
        raise ValueError('Unknown transform_type: {}'.format(transform_type))

    return (train_transform, test_transform)
