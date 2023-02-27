import pdb, os, sys, argparse, time, copy
import numpy as np
import matplotlib.pyplot as plt

# from __future__ import print_function
# from __future__ import division

import torch
import torch.optim as optim
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import vgg16, VGG16_Weights
from torchvision.utils import make_grid
import torchvision.transforms.functional as F

plt.rcParams["savefig.bbox"] = 'tight'

from dataloader.getDataloaders import getDataloaders

# https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img), cmap='gray', vmin=0, vmax=1)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    # state_dict() returns the weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


if __name__ == "__main__":
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Torchvision Version: {torchvision.__version__}")

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", 
                        required=True,
                        help="name of the dataset (e.g. multiDrawingMCI)")
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--random_seed', default=777, type=int)
    parser.add_argument('--val_fraction', default=0.15, type=float)
    parser.add_argument('--test_fraction', default=0.15, type=float)
    parser.add_argument('--num_epochs', default=1000, type=int)
    
    
    args = parser.parse_args()
    
    if args.val_fraction + args.test_fraction >= 1:
        print('No training data')
        sys.exit(1)
        
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    # Generate dataloders: dataloader_dict.keys -> 'train', 'val', 'test'
    dataloader_dict = getDataloaders(dataset_name=args.dataset,
                              batch_size=args.batch_size,
                              val_fraction=args.val_fraction,
                              test_fraction=args.test_fraction,
                              random_seed=args.random_seed)
    
    # Test the loaders
    if False:
        curr_data_batch, curr_label_batch =  next(iter(dataloader_dict['train']))

        temp_list = []
        for idx in range(args.batch_size):
            temp_list.append(curr_data_batch['clock'][idx])
            temp_list.append(curr_data_batch['copy'][idx])
            temp_list.append(curr_data_batch['trail'][idx])
        grid = make_grid(temp_list)
        show(grid)
        plt.savefig('test_loader.png')
        
        max_val = np.max(curr_data_batch['clock'].numpy())
        min_val = np.min(curr_data_batch['clock'].numpy())
        print(f"The data are in the range ({min_val},{max_val})") #(0.0,1.0)
        
    
    ## Create our model
    num_classes = 2
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    num_final_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_final_features,num_classes)
    
    # Define the loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    pdb.set_trace()
    trained_model, val_acc_history = train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False)
    
#     for args.num_epochs
#     for x,y in dataloader_dict['train']:
#         train_loss[idx] = loss()
        
#         if idx % 100 == 0
#             plt.plot()
#             plt.savefig()
#             f.write()
            
            
    pdb.set_trace()
    



       

