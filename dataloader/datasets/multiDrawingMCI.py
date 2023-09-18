import os, pdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
from torchvision.transforms import Lambda

import matplotlib.pyplot as plt
plt.rcParams["savefig.bbox"] = 'tight'

class MultiDrawingMCIDataset2022(Dataset):
    
    def __init__(self, 
                 dataset_dir,
                 data_info_df, 
                 transform=None, 
                 target_transform=None, 
                 task_list=['clock', 'copy', 'trail'], 
                 label_type='hard'):
        
        self.dataset_dir = dataset_dir
        self.data_info_df = data_info_df # Shape: (IDs, MoCA scores)
        self.transform = transform
        self.target_transform = target_transform
        self.task_list = task_list
        self.label_type = label_type

        
    def __len__(self):
        return len(self.data_info_df)

    
    def __getitem__(self, idx):
        
        ## Images
        image_dict = {}
        
        for curr_task in self.task_list: # ['clock','copy','trail']
            
            # read_image returns a Tensor of type unit8
            curr_image = read_image(os.path.join(self.dataset_dir, 
                                                 'images', 
                                                 str(self.data_info_df.iloc[idx, 0]), 
                                                 curr_task + '.png'))/255.0
                        
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
    
    
    def display_dataset_stats(self, save_dir, mode):
        
        raw_MoCA_scores = self.data_info_df.iloc[:, 1]
        
        plt.figure(figsize=(4, 4))
        if self.target_transform:
            
            labels = [self.target_transform(x) for x in raw_MoCA_scores]
 
            # Make the histogram based on the prob of having MCI
            labels = [curr_label[1] for curr_label in labels]
            plt.hist(labels)
            plt.xlabel(self.label_type + ' labels')

        else:
            plt.hist(raw_MoCA_scores)
            plt.xlabel('MoCA scores')
        
        plt.ylabel('counts')
        plt.title(mode)
        plt.savefig(os.path.join(save_dir, 
                                 'labels_hist_' + mode + '.jpg'), 
                    dpi=150)
        plt.close()


def makeTransformMultiDrawingMCIDataset2022(args, add_info, split_mode='train'):
    
    # Create data transform
    if args.use_pretrained_weight:
        # Source https://pytorch.org/vision/0.8/models.html 
        # "All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]"
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if split_mode in ['test', 'val']:
        if args.use_pretrained_weight:
            transform = normalize
        else:
            transform = None
    else:
        # Hard-coded 280 and 256, following the original paper (Ruengchaijatuporn et al. 2022)
        if args.use_pretrained_weight:
            transform = transforms.Compose([CustomPad(280),
                                            transforms.RandomCrop(256),
                                            normalize])
        else:
            transform = transforms.Compose([CustomPad(280), 
                                            transforms.RandomCrop(256)])
                                                                                  
    # Create target transform
    if args.label_type == 'raw':
        target_transform = None
    elif args.label_type == 'soft':
        # [p(control), p(mci)]
        target_transform = Lambda(lambda x: get_soft_label(x, add_info['healthy_threshold']))
    else:
        # Healthy control = 0, MCI = 1
        target_transform = Lambda(lambda x: get_hard_label(x, add_info['healthy_threshold'])) # 2-output-node version
        # target_transform = Lambda(lambda x: 1 if x < add_info['healthy_threshold'] else 0) # 1-output-node version                                
                                                                                  
    return transform, target_transform
    
    
# Custom padding
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
            
    
def get_soft_label(x, healthy_threshold=25):
    # soft label = 1 - sigmoid(x-24.5). 
    prob_mci = 1 - sigmoid(x - (healthy_threshold - 0.5))
    return np.asarray([1 - prob_mci, prob_mci])


def get_hard_label(x, healthy_threshold=25):
    
    prob_mci = (x < healthy_threshold)*1.0    
    return np.asarray([1 - prob_mci, prob_mci])


def sigmoid(x):
    return 1/(1 + np.exp(-x))