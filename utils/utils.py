import os, pdb
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["savefig.bbox"] = 'tight'

import torch
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
from sklearn.metrics import precision_recall_fscore_support, classification_report


# Make experiment name
def make_exp_name(name=''):
    out_name = 'EXP_'+datetime.now().strftime("%Y-%m-%d-%X")
    if name and len(name) > 0:
        out_name += '_{}'.format(name)
    return out_name


# Show Taken from https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img), cmap='gray', vmin=0, vmax=1)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])        

        
def testDataloader(dataloader, results_dir, batch_size, task_list):
    
    # Get one batch
    curr_data_batch, curr_label_batch =  next(iter(dataloader))
    
    temp_list = []
    for idx in range(batch_size):
        for curr_task in task_list:
            temp_list.append(curr_data_batch[curr_task][idx])
    
    # Create an image grid using all the images in the batch
    grid = make_grid(temp_list)
    
    # Show and save the image grid
    plt.figure(figsize=(8, 8))
    show(grid)
    plt.savefig(os.path.join(results_dir,
                             'test_loader_sample_data_batch.png'), dpi=150)
    
    # Check the values
    for curr_task in task_list:
        max_val = np.max(curr_data_batch[curr_task].numpy())
        min_val = np.min(curr_data_batch[curr_task].numpy())
        print(f"The {curr_task} data are in the range ({min_val}, {max_val})") #(0.0, 1.0)
        
    # Show example labels
    pd.DataFrame(data=curr_label_batch).to_csv(os.path.join(results_dir, 
                                                            'test_loader_sample_label_batch.csv'), 
                                               index=False)
    
def test_model(model, dataloader, device):

    model.eval()
    model.to(device)

    labels_predicted_all = []
    labels_true_all = []
    proba_predicted_all = []

    for inputs, labels in dataloader:

        inputs = inputs.to(device)
        outputs = model(inputs) #(max val, max_indices)
        proba, preds = torch.max(outputs, 1)

        labels_true_all += labels.tolist()
        labels_predicted_all += preds.detach().tolist()
        proba_predicted_all += proba.detach().tolist()

    return labels_true_all, labels_predicted_all, proba_predicted_all
    
    
def save_evaluation(labels_true,
                    labels_predicted, 
                    proba_predicted, 
                    results_dir, 
                    class_list):
    
    # Deal with one-hot/soft labels
    if not isinstance(labels_true[0], int):
        labels_true = [np.argmax(x) for x in labels_true]
    
    pdb.set_trace()
    
    precision, recall, fscore, _ = precision_recall_fscore_support(labels_true, labels_predicted)
    
    accuracy = np.sum(np.array(labels_true)==np.array(labels_predicted))/len(labels_true)
    
    # Save results
    df = pd.DataFrame({'true': labels_true,
                       'predicted': labels_predicted,
                       'prob': proba_predicted})
    df.to_csv(os.path.join(results_dir,
                           'predictions.csv'),index=False)
    
    with open(os.path.join(results_dir, 'eval_metrics.txt'), "w") as f:
        # f.write(f"best val loss = {best_val_loss}\n")
        f.write(f"accuracy = {accuracy}\n")
        f.write(f"precision = {precision}\n")
        f.write(f"recall = {recall}\n")
        f.write(f"fscore = {fscore}\n")
        f.write(classification_report(
            labels_true,
            labels_predicted,
            target_names=class_list))
        
        

# class SoftLabelCrossEntropyLoss(torch.nn.Module):
    
#     def __init__(self, reduction='sum'):
#         super(SoftLabelCrossEntropyLoss, self).__init__()
#         self.log_softmax = LogSoftmax(dim=1)
#         self.reduction = reduction
        
#     def forward(self,prediction,target):
#         pred_log_softmax = self.log_softmax(prediction)
#         loss_each_sample = -(pred_log_softmax*target).sum(axis=1)
#         if self.reduction == 'mean':
#             return loss_each_sample.sum(axis=0)/loss_each_sample.shape[0]
#         else: # default: reduction = 'sum'
#             return loss_each_sample.sum(axis=0)

