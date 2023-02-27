import pdb, os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision.transforms import Lambda

# Return: a dictionary of dataloaders with the keys: 'train', 'val', 'test'
def getDataloaders(dataset_name, 
                   batch_size, 
                   val_fraction, 
                   test_fraction, 
                   random_seed):
    
    dataset_name = dataset_name.lower()
        
    if dataset_name == 'multiDrawingMCI'.lower():
        
        from dataloader.datasets.multiDrawingMCI import MultiDrawingMCIDataset2022, makeTransformMultiDrawingMCIDataset2022        
        
        # Adjustable parameters
        healthy_threshold = 25 # MoCA score of 25 or above -> healthy
        label_type = 'hardlabel' # 'softlabel', 'hardlabel', 'raw'. Default: 'hardlabel'
        use_pretrained_weight = False # ImageNet. This flag is used to compose the transform
        
        dataset_dir = os.path.join('./data','multiDrawingMCI2022')
        train_fraction = 1-val_fraction-test_fraction
        data_info_df = pd.read_csv('./data/multiDrawingMCI2022/label.csv')
        
        # Define target_transform
        if label_type == 'raw':
            target_transform = None
        elif label_type == 'softlabel':
            # soft label = 1 - sigmoid(x-24.5)
            target_transform = Lambda(lambda x: 1 - 1/(1 + np.exp(-(x-(healthy_threshold-0.5)))))
        else:
            # Healthy control = 0, MCI = 1
            target_transform = Lambda(lambda x: 1 if x<healthy_threshold else 0)
            
        
        # Train, test, split - PyTorch Datasets
        split_info_df = {}
        split_info_df['train'], split_info_df['test'] = train_test_split(data_info_df, 
                                                       test_size=test_fraction, 
                                                       random_state=random_seed, 
                                                       shuffle=True,
                                            stratify=data_info_df.iloc[:,-1]>=healthy_threshold)
        
        split_info_df['train'], split_info_df['val'] = train_test_split(split_info_df['train'], 
                                               test_size=val_fraction/(1-test_fraction), 
                                               random_state=random_seed, 
                                               shuffle=True,
                                    stratify=split_info_df['train'].iloc[:,-1]>=healthy_threshold)
        
        # Train, test, split - PyTorch Dataloaders
        dataloader_dict = {}
        for curr_split_mode in ['train','val','test']:
            
            # Create PyTorch Datasets
            transform = makeTransformMultiDrawingMCIDataset2022(curr_split_mode, use_pretrained_weight)
            
            curr_dataset = MultiDrawingMCIDataset2022(dataset_dir,
                                                      split_info_df[curr_split_mode],
                                                      transform=transform,
                                                      target_transform=target_transform,
                                                      task_list = ['clock','copy','trail'])
                                         
            # Create PyTorch Dataloaders
            if curr_split_mode in ['test','val']:
                dataloader_dict[curr_split_mode]= DataLoader(curr_dataset, batch_size,shuffle=False)
            else:
                dataloader_dict[curr_split_mode]= DataLoader(curr_dataset, batch_size,shuffle=True)

        return dataloader_dict
    else:
        return None