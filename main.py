import pdb, os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision

import lightning as pl # conda install lightning -c conda-forge
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from models.get_model import get_model
from dataloader.get_dataloaders import get_dataloaders
from utils.utils import make_exp_name, test_dataloader, test_model, save_evaluation


if __name__ == "__main__":
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Torchvision Version: {torchvision.__version__}")

    ## Arguments
    parser = argparse.ArgumentParser()
    
    # Experimental setup
    parser.add_argument('--random_seed', default=777, type=int)
    parser.add_argument('--val_fraction', default=0.15, type=float)
    parser.add_argument('--test_fraction', default=0.15, type=float)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--exp_name', default='', type=str)
    parser.add_argument('--gpu_id', default=0, type=int)
    
    # Model config
    parser.add_argument('--model_name', 
                        default='vgg16', 
                        type=str, 
                        help="Available options: 'vgg16', 'conv-att'")
    parser.add_argument('--config_file', default='', type=str)
    parser.add_argument('--use_pretrained_weight', default=False, action='store_true')
    parser.add_argument('--freeze_backbone', default=False, action='store_true')
    
    # Data and labels
    parser.add_argument('--include_clock', default=False, action='store_true')
    parser.add_argument('--include_copy', default=False, action='store_true')
    parser.add_argument('--include_trail', default=False, action='store_true')
    parser.add_argument('--label_type', 
                        default='soft', 
                        type=str, 
                        help="Options: 'raw', 'hard', 'soft'")
    parser.add_argument('--test_dataloader', 
                        default=False, 
                        action='store_true',
                        help="Set to True to get a sample batch (to test out the dataloader)")
    parser.add_argument('--num_workers', default=8, type=int)
    
    ## Processing the arguments
    args = parser.parse_args()
    
    # Create a dictionary to store additional info
    add_info = {'dataset_name': 'multiDrawingMCI',
                'idx2class_dict': {'0': 'control', '1': 'mci'},
                'healthy_threshold': 25, # MoCA score of >= 25-> healthy
                'batch_size': 64}
    add_info['class_list'] = [add_info['idx2class_dict'][key] for key in add_info['idx2class_dict'].keys()]
    add_info['num_classes'] = len(add_info['idx2class_dict'].keys())
    
    # Create 'results' folder (Ex. results/multidrawingmci/EXP_...)
    add_info['results_dir'] = os.path.join('results', make_exp_name(args.exp_name))
    os.makedirs(add_info['results_dir'], exist_ok=False)
    
    # Check val and test fractions
    if args.val_fraction + args.test_fraction >= 1:
        print('Invalid training fraction')
        sys.exit(1)
        
    # Check if at least one task is specified
    if not (args.include_clock or args.include_copy or args.include_trail):
        print('No valid task specified')
        sys.exit(1)

    # Create task_list
    add_info['task_list'] = []
    if args.include_clock:
        add_info['task_list'].append('clock')
    if args.include_copy:
        add_info['task_list'].append('copy')
    if args.include_trail:
        add_info['task_list'].append('trail')
        
    ## Detect if we have a GPU available
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count() 
        add_info['device'] = torch.device(f"cuda:{args.gpu_id}")
        gpu_list = [args.gpu_id]
        print(f"device: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        add_info['device'] = torch.device("cpu")

    ## Generate dataloders: dataloader_dict.keys -> 'train', 'val', 'test'
    dataloader_dict = get_dataloaders(args, add_info)
        
    ## Try getting a batch from the test Dataloader
    if args.test_dataloader:
        test_dataloader(dataloader_dict['test'], 
                        add_info['results_dir'], 
                        add_info['batch_size'], 
                        add_info['task_list'])
        
    # Define the loss function
    if args.label_type in ['hard', 'soft']:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    elif args.label_type == 'raw':
        add_info['num_classes'] = 1
        loss_fn = torch.nn.MSELoss(reduction='mean')
                
    # Create a ModuleDict model
    model = get_model(args.model_name, 
                      add_info['num_classes'], 
                      add_info['task_list'], 
                      args.use_pretrained_weight, 
                      args.freeze_backbone)

    # Define the LightningModule
    class LitVGG16(pl.LightningModule):
        
        def __init__(self, model, loss_fn):
            super().__init__()
            self.model = model
            self.loss_fn = loss_fn
            self.softmax = torch.nn.Softmax(dim=1)

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits_predicted = self.model(x) # Without softmax
                
            loss = self.loss_fn(logits_predicted, y)
            
            # Log results
            self.log("train_loss", loss)
            
            return loss
        
        def forward(self, x):
            logits_predicted = self.model(x) # Without softmax
            return self.softmax(logits_predicted)
        
        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits_predicted = self.model(x) # Without softmax

            loss = self.loss_fn(logits_predicted, y)

            # Log results
            self.log("val_loss", loss)

            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(model.parameters(), 
                                   lr=0.001, 
                                   momentum=0.9)

    # Create the model
    modelLit = LitVGG16(model, loss_fn)

    # Prepare logger
    logger = CSVLogger(save_dir=add_info['results_dir'])
    
    # Prepare callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=add_info['results_dir'], 
                                          filename='{epoch}-{val_loss:.2f}', 
                                          monitor='val_loss',
                                          mode='min',
                                          save_top_k=1, 
                                          every_n_epochs=1)
    
    # Train the model
    trainer = pl.Trainer(max_epochs=args.num_epochs, 
                         logger=logger,
                         callbacks=[checkpoint_callback],
                         devices=gpu_list, 
                         log_every_n_steps=dataloader_dict['train'].__len__(),
                         accelerator="gpu")

    trainer.fit(model=modelLit, 
                train_dataloaders=dataloader_dict['train'], 
                val_dataloaders=dataloader_dict['val'])

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Load the best model
    trained_model = LitVGG16.load_from_checkpoint(checkpoint_callback.best_model_path, 
                                                  model=model, 
                                                  loss_fn=loss_fn)
    
    # Test the model
    labels_true, probs_predicted = test_model(trained_model, 
                                              dataloader_dict['test'],
                                              add_info['device'])
    
    # Save evaluation metrics    
    save_evaluation(labels_true,
                    probs_predicted,
                    add_info['results_dir'], 
                    add_info['class_list'])
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()