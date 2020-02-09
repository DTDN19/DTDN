from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from .mixins import PigMixin, RGBMixin

# train 60000
# test 10000
# 28 * 28
class PigMNIST(PigMixin, RGBMixin, MNIST):
    pass

class Data:
    def __init__(self, args):
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        trainset = PigMNIST(root=args.data_dir, train=True, download=True, transform=transform)
        self.loader_train = DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=8
            )

        self.train_num = len(trainset)
        testset = PigMNIST(root=args.data_dir, train=False, download=True, transform=transform)
        self.loader_test = DataLoader(
            testset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8)
