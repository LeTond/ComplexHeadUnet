import torch
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms


from torch.utils.data import Dataset
from Preprocessing.preprocessing import ReadImages, PreprocessData, EvalPreprocessData
from parameters import *
from parameters import MetaParameters
from Preprocessing.dirs_logs import *


class GetData(MetaParameters):
    def __init__(self, files=None, augmentation=None): 
        
        super(MetaParameters, self).__init__()
        self.files = files
        self.augmentation = augmentation

    def generated_data_list(self):
        list_images, list_masks, list_names = [], [], []

        for file_name in self.files:
            if file_name.endswith('.nii'):

                images = ReadImages(f"{self.ORIGS_DIR}/{file_name}").view_matrix()
                masks = ReadImages(f"{self.MASKS_DIR}/{file_name}").view_matrix()
                
                sub_name = file_name.replace('.nii', '')

                masks[masks>0] = 1

                for slc in range(images.shape[2]):
                    image = images[:, :, slc]
                    mask = masks[:, :, slc]
                    
                    if self.EMPTY is False and (mask>0).any() == False:
                        print(f"Subject {sub_name} slice {slc} was passed because EMPY is FALSE")
                        pass

                    else:
                        try:
                            normalized = PreprocessData(image, mask).preprocessing(self.KERNEL)
                        except:
                            print(f'Data Preprocessing Problem with {sub_name}')
                      
                        list_images.append(normalized[0])
                        list_masks.append(normalized[1])
                        list_names.append(f'{sub_name} Slice {images.shape[2] - slc}')

        try:
            shuff = PreprocessData(list_images, list_masks, list_names).shuff_dataset()
        except:
            pass
        return list_images, list_masks, list_names


class MyDataset(Dataset):
    def __init__(self, num_class, ds_origin, ds_mask, ds_names, kernel_sz, transform = None, images_and_labels = []):
        self.transform = transform
        self.images_and_labels = images_and_labels
        self.images = ds_origin
        self.masks = ds_mask
        self.names = ds_names
        self.kernel_sz = kernel_sz
        self.num_class = num_class

        for i in range(len(self.images)):
            self.images_and_labels.append((i, i, i))

    def preprocessing(self, image, label):
        try:
            label = label / (self.num_class - 1)
            image = TF.to_pil_image(image)
            label = TF.to_pil_image(label)
            image = TF.pil_to_tensor(image)
            label = TF.pil_to_tensor(label)

            tcat = torch.cat((image, label), 0)
            image, label = self.transform(tcat)

        except:
            image = np.array(image.reshape(self.kernel_sz, self.kernel_sz, 1), dtype=np.float32)
            mask = np.array(label.reshape(self.kernel_sz, self.kernel_sz, 1), dtype=np.float32)

            transformed = self.transform(image=image, mask=mask)
            image, label = transformed["image"], transformed["mask"]

        image = np.array(image.reshape(self.kernel_sz, self.kernel_sz, 1), dtype=np.float32)
        label = np.array(label.reshape(self.kernel_sz, self.kernel_sz, 1), dtype=np.float32)
        label = label * (self.num_class - 1)

        return image, label

    def __getitem__(self, item):
        imgs, labs, sub_nms = self.images_and_labels[item]
        image = self.images[imgs][:][:]
        label = self.masks[labs][:][:]
        sub_names = self.names[sub_nms]

        image, label = self.preprocessing(image, label)
        image = image.transpose(2, 0, 1)

        label = np.resize(label, (self.kernel_sz, self.kernel_sz))
        label = np.array(label, dtype=np.int8)
        label = np.eye(self.num_class)[label]
        label = np.array(label, dtype=np.float32)
        label = label.transpose(2, 0, 1)

        return image, label, sub_names

    def __len__(self):
        return len(self.images)