from torchvision.datasets import SVHN
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from .mixins import PigMixin

# train 73257
# test 26032
# 32 * 32
class PigSVHN(PigMixin, SVHN):
    pass

class Data:
    def __init__(self, args):
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = PigSVHN(root=args.data_dir, split='train', download=True, transform=transform)
        self.loader_train = DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=8
            )
        
        self.train_num = len(trainset)

        testset = PigSVHN(root=args.data_dir, split='test', download=True, transform=transform)
        self.loader_test = DataLoader(
            testset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8)
