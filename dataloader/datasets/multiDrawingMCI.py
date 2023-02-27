import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms

class MultiDrawingMCIDataset2022(Dataset):
    
    def __init__(self, 
                 dataset_dir,
                 data_info_df, 
                 transform=None, 
                 target_transform=None, 
                 task_list = ['clock','copy','trail']):
        
        self.dataset_dir = dataset_dir
        self.data_info_df = data_info_df #(patient ID, MoCA score)
        self.transform = transform
        self.target_transform = target_transform
        self.task_list = task_list


    def __len__(self):
        return len(self.data_info_df)

    def __getitem__(self, idx):
        
        ## Images
        image_dict = {}
        
        for curr_task in self.task_list: # ['clock','copy','trail']
            
            # read_image returns a Tensor of type unit8
            curr_image = read_image(os.path.join(
            # self.dataset_dir,'images', str(self.data_info_df.iloc[idx, 0]),curr_task+'.png')).to(torch.float32)
            self.dataset_dir,'images', str(self.data_info_df.iloc[idx, 0]),curr_task+'.png'))/255.0
                        
            # Transform the data
            if self.transform:
                curr_image = self.transform(curr_image)
            
            image_dict[curr_task] = curr_image;
          
        ## Labels
        label = self.data_info_df.iloc[idx, 1]
        
        # Transform the label
        if self.target_transform:
            label = self.target_transform(label)
            
        return image_dict, label
    
    
def makeTransformMultiDrawingMCIDataset2022(split_mode='train',
                                            use_pretrained_weight=True):
    
    if use_pretrained_weight:
        # Source https://pytorch.org/vision/0.8/models.html 
        # All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    
    if split_mode in ['test','val']:
        if use_pretrained_weight:
            transform = normalize
        else:
            transform = None
    else:
        # Hard-coded 280 and 256, following the paper
        if use_pretrained_weight:
            transform = transforms.Compose([CustomPad(280),
                                            transforms.RandomCrop(256),
                                            normalize])
        else:
            transform = transforms.Compose([CustomPad(280),transforms.RandomCrop(256)])
        
    return transform    
    
class CustomPad(object):
    
    def __init__(self,target_size):
        self.target_size = target_size
    
    def __call__(self, image):
        num_rows, num_cols = image.shape[-2:]
        # Resize the image such that the resulting max size is equal to target_size if needed
        if max(num_rows, num_cols) > self.target_size:
            scale = self.target_size/max(num_rows, num_cols)
            image = transforms.Resize(size=(int(scale*num_rows),int(scale*num_cols))).forward(image)


        # Pad the image if needed
        num_rows, num_cols = image.shape[-2:]
        if num_rows < self.target_size:
            row_pad_amount = self.target_size-num_rows
            left_pad_amount = int(np.ceil(row_pad_amount/2))
            right_pad_amount = int(np.floor(row_pad_amount/2))
        else:
            left_pad_amount = 0
            right_pad_amount = 0

        if num_cols < self.target_size:
            col_pad_amount = self.target_size-num_cols
            top_pad_amount = int(np.ceil(col_pad_amount/2))
            bottom_pad_amount = int(np.floor(col_pad_amount/2))
        else:
            top_pad_amount = 0
            bottom_pad_amount = 0    

        image = transforms.Pad(padding=(left_pad_amount,top_pad_amount,right_pad_amount,bottom_pad_amount), fill=1).forward(image) # Here, I filled '1' instead of '0' since our background is white

        return image
            
    
