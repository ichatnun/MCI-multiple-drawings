import pdb, os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Return: a dictionary of dataloaders with the keys: 'train', 'val', 'test'
def getDataloaders(args, add_info):
        
    if add_info['dataset_name'].lower() == 'multiDrawingMCI'.lower():
        
        from dataloader.datasets.multiDrawingMCI import MultiDrawingMCIDataset2022, makeTransformMultiDrawingMCIDataset2022        
        
        # Adjustable parameters
        dataset_dir = os.path.join('./data', 'multiDrawingMCI2022')
        train_fraction = 1 - args.val_fraction - args.test_fraction
        data_info_df = pd.read_csv('./data/multiDrawingMCI2022/label.csv') # (IDs, MoCA scores)            
        
        # Train, val, test split
        split_info_df = {}
        split_info_df['train'], split_info_df['test'] = train_test_split(data_info_df, 
                                                       test_size=args.test_fraction, 
                                                       random_state=args.random_seed, 
                                                       shuffle=True,
                                            stratify=data_info_df.iloc[:,-1] >= add_info['healthy_threshold'])
        
        split_info_df['train'], split_info_df['val'] = train_test_split(split_info_df['train'], 
                                               test_size=args.val_fraction/(1 - args.test_fraction), 
                                               random_state=args.random_seed, 
                                               shuffle=True,
                                    stratify=split_info_df['train'].iloc[:,-1] >= add_info['healthy_threshold'])
        
        # Train, val, test Datasets and corresponding Dataloaders
        dataloader_dict = {}
        for curr_split_mode in ['train', 'val', 'test']:
            
            # Create data and target transformations
            transform, target_transform = makeTransformMultiDrawingMCIDataset2022(args, add_info, curr_split_mode)
            
            # Create PyTorch Datasets
            curr_dataset = MultiDrawingMCIDataset2022(dataset_dir,
                                                      split_info_df[curr_split_mode],
                                                      transform,
                                                      target_transform,
                                                      add_info['task_list'],
                                                      args.label_type)
            
            # Save the Datasets stats
            curr_dataset.display_dataset_stats(add_info['results_dir'], 
                                               curr_split_mode)
            
            # Create PyTorch Dataloaders
            if curr_split_mode in ['test', 'val']:
                dataloader_dict[curr_split_mode]= DataLoader(curr_dataset, 
                                                             add_info['batch_size'],
                                                             shuffle=False)
            else:
                dataloader_dict[curr_split_mode]= DataLoader(curr_dataset, 
                                                             add_info['batch_size'],
                                                             shuffle=True)

        return dataloader_dict