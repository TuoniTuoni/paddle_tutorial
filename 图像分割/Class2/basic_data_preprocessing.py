from basic_transforms import *

class TrainAugmentation():
    def __init__(self, image_size, mean_val=0, std_val=1.0):
        #TODO: add self.augment, which contains
        # random scale, pad, random crop, random flip, convert data type, and normalize ops
        self.iamge_szie=image_size
        self.mean_val=mean_val
        self.std_val=std_val
        augment=basic_transforms.Compose([RandomScale(),
                    RandomFlip(),
                    Pad(crop_size,mean_val=[0.485,0.456,0.406]),
                    RandomCrop(crop_size),
                    ConvertDataType(),
                    Normalize(0,1)])

        
    def __call__(self, image, label):
        return self.augment(image, label)
