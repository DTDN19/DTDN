from PIL import Image

class PigMixin(object):
    def __getitem__(self, index):
        data = super(PigMixin, self).__getitem__(index)
        return (*data, index)

class RGBMixin(object):
    """For MNIST Only
    """
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy()).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
