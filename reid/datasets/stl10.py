from torchvision.datasets import STL10
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from skimage.transform import downscale_local_mean

from .mixins import PigMixin

# train 8000
# test 3000
# 96*96
class PigSTL10(PigMixin, STL10):
    pass

class Data:
    def __init__(self, args):
        # pin_memory = False
        # if args.gpu is not None:
        pin_memory = True
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(), transforms.Resize((32, 32)), transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        trainset = PigSTL10(root=args.data_dir, split='train', download=False, transform=transform_train)
        self.loader_train = DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, 
            num_workers=2, pin_memory=pin_memory, drop_last=True
            )

        # trainset = downscale_local_mean(trainset, (1, 1, 3, 3))
        self.train_num = len(trainset)

        testset = PigSTL10(root=args.data_dir, split='test', download=False, transform=transform_test)
        self.loader_test = DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, 
            num_workers=2, pin_memory=pin_memory)

        # Example:
        # >>> policy = CIFAR10Policy()
        # >>> transformed = policy(image)
        #
        # Example as a PyTorch Transform:
        # >>> transform=transforms.Compose([
        # >>>     transforms.Resize(256),
        # >>>     CIFAR10Policy(),
        # >>>     transforms.ToTensor()])