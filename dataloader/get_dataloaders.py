import pdb, os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Return: a dictionary of dataloaders with the keys: 'train', 'val', 'test'
def get_dataloaders(use_pretrained_weight, 
                    label_type, 
                    val_fraction, 
                    test_fraction, 
                    random_seed, 
                    batch_size, 
                    num_workers, 
                    dataset_name, 
                    healthy_threshold, 
                    task_list, 
                    results_dir):
        
    if dataset_name.lower() == 'multiDrawingMCI'.lower():
        
        from .datasets.multiDrawingMCI import MultiDrawingMCIDataset2022, make_transform_multi_drawing_mci_dataset2022        
        
        # Adjustable parameters
        dataset_dir = os.path.join(os.getcwd(), 'data', 'multiDrawingMCI2022')
        train_fraction = 1 - val_fraction - test_fraction
        label_path = os.path.join(os.getcwd(), 'data', 'multiDrawingMCI2022', 'label.csv')
        
        # Create the 'data' directory if it does not exist
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            
        # Download images (if needed)
        if not os.path.exists(os.path.join(dataset_dir, 'images')):
            print("*************** Downloading images ***************")
            os.system(f"wget -O images.zip https://github.com/cccnlab/MCI-multiple-drawings/raw/main/images.zip")
            os.system(f"unzip -q images.zip -d {dataset_dir}/")
            os.system(f"rm images.zip")
            print("*************** Done ***************\n")
            
        # Download the labels (if needed)
        if not os.path.exists(label_path):
            print("*************** Downloading the labels ***************")
            os.system(f"wget -P {dataset_dir} https://github.com/cccnlab/MCI-multiple-drawings/raw/main/label.csv")
            print("*************** Done ***************\n")
            
        # Load label into a Pandas dataframe
        data_info_df = pd.read_csv(label_path) # (IDs, MoCA scores) 
        
        # Train, val, test split
        split_info_df = {}
        split_info_df['train'], split_info_df['test'] = train_test_split(data_info_df, 
                                                                         test_size=test_fraction, 
                                                                         random_state=random_seed, 
                                                                         shuffle=True,
                                                                         stratify=data_info_df.iloc[:,-1] >= healthy_threshold)
        
        split_info_df['train'], split_info_df['val'] = train_test_split(split_info_df['train'],
                                                                        test_size=val_fraction/(1 - test_fraction), 
                                                                        random_state=random_seed, 
                                                                        shuffle=True,
                                                                        stratify=split_info_df['train'].iloc[:,-1] >= healthy_threshold)
        
        # Train, val, test Datasets and corresponding Dataloaders
        dataloader_dict = {}
        for curr_split_mode in ['train', 'val', 'test']:
            
            # Create data and target transformations
            transform, target_transform = make_transform_multi_drawing_mci_dataset2022(use_pretrained_weight, 
                                                                                       label_type, 
                                                                                       healthy_threshold,
                                                                                       curr_split_mode)

            
            # Create PyTorch Datasets
            curr_dataset = MultiDrawingMCIDataset2022(dataset_dir,
                                                      split_info_df[curr_split_mode],
                                                      transform,
                                                      target_transform,
                                                      task_list,
                                                      label_type,
                                                      healthy_threshold)
            
            # Get the class distribution of the training data
            if curr_split_mode == 'train':
                label_distribution_train = curr_dataset.get_label_distribution()
            
            # Save the Datasets stats
            curr_dataset.display_dataset_stats(results_dir, 
                                               curr_split_mode)
            
            # Create PyTorch Dataloaders
            if curr_split_mode in ['test', 'val']:
                dataloader_dict[curr_split_mode]= DataLoader(curr_dataset, 
                                                             batch_size,
                                                             num_workers=num_workers,
                                                             shuffle=False, 
                                                             drop_last=False)
            else:
                dataloader_dict[curr_split_mode]= DataLoader(curr_dataset, 
                                                             batch_size,
                                                             num_workers=num_workers,
                                                             shuffle=True,
                                                             drop_last=False)

        return dataloader_dict, label_distribution_train