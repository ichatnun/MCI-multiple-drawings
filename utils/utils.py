import os, pdb
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["savefig.bbox"] = 'tight'

import torch
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
from sklearn.metrics import precision_recall_fscore_support, classification_report, roc_auc_score


# Make experiment name
def make_exp_name(name=''):
    return name + '_' + datetime.now().strftime("%Y-%m-%d-%X")

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

        
def test_dataloader(dataloader, results_dir, batch_size, task_list):
    
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

    labels_true = []
    probs_predicted = []
    
    for inputs, labels in dataloader: # labels: (batch, num_classes)
        
        for key, value in inputs.items():
            inputs[key] = inputs[key].to(device)
        
        probs = model(inputs) # outputs: (batch, num_classes)
    
        # Move 'probs' to cpu
        probs = probs.detach().cpu()
        
        # Combine the batches
        labels_true.append(labels)
        probs_predicted.append(probs)
        
        # Delete unused variables
        del inputs, probs
        torch.cuda.empty_cache()
    
    # Convert a list of arrays to a single array
    labels_true = torch.vstack(labels_true) # (num_samples, num_classes)
    probs_predicted = torch.vstack(probs_predicted) # (num_samples, num_classes)
    
    model.to("cpu")
    
    return labels_true, probs_predicted
    
    
def save_evaluation(labels_true, # (num_samples, num_classes)
                    probs_predicted, # (num_samples, num_classes)
                    results_dir, 
                    class_list):
    
    # Convert from probabilities to integers
    _, labels_true_int = torch.max(labels_true, 1) # labels_true_int: (batch, )
    
    # Convert from probabilities to integers 
    probs_of_predicted_class, labels_predicted_int = torch.max(probs_predicted, 1) # probs_of_predicted_class, labels_predicted_int: (batch, )
        
    # Compute several evaluation metrics
    eval_metrics_dict = classification_report(labels_true_int,
                                              labels_predicted_int,
                                              target_names=class_list, 
                                              output_dict=True)
    accuracy = eval_metrics_dict['accuracy']
    precision_macro = eval_metrics_dict['macro avg']['precision']
    recall_macro = eval_metrics_dict['macro avg']['recall']
    f1_score_macro = eval_metrics_dict['macro avg']['f1-score']
    precision_weighted = eval_metrics_dict['weighted avg']['precision']
    recall_weighted = eval_metrics_dict['weighted avg']['recall']
    f1_score_weighted = eval_metrics_dict['weighted avg']['f1-score']

    # Compute AUC
    auc = roc_auc_score(soft_to_hard_labels(labels_true), probs_predicted)
    
    ## Save results
    # Save evaluation metrics in a text file
    with open(os.path.join(results_dir, 'eval_metrics.txt'), "w") as f:
        f.write(f"accuracy = {accuracy:.2f}\n")
        f.write(f"precision (macro) = {precision_macro:.2f}\n")
        f.write(f"recall (macro) = {recall_macro:.2f}\n")
        f.write(f"f1-score (macro) = {f1_score_macro:.2f}\n")
        f.write(f"auc = {auc:.2f}\n")
        f.write(f"accuracy = {accuracy:.2f}\n")
        f.write(f"precision (weighted) = {precision_weighted:.2f}\n")
        f.write(f"recall (weighted) = {recall_weighted:.2f}\n")
        f.write(f"f1-score (weighted) = {f1_score_weighted:.2f}\n")
        f.write(classification_report(labels_true_int,
                                      labels_predicted_int,
                                      target_names=class_list))
        
    # Save evaluation metrics in a csv file
    df = pd.DataFrame({'accuracy': accuracy,
                       'f1-score (macro)': f1_score_macro,
                       'auc': auc,
                       'precision (macro)': precision_macro,
                       'recall (macro)':recall_macro,
                       'f1-score (weighted)': f1_score_weighted,
                       'precision (weighted)': precision_weighted,
                       'recall (weighted)':recall_weighted}, index=[0])
    df.to_csv(os.path.join(results_dir,
                           'eval_metrics.csv'), index=False)
    
    # Save predictions
    df = pd.DataFrame({'true': labels_true_int,
                       'predicted': labels_predicted_int,
                       'prob': probs_of_predicted_class})
    df.to_csv(os.path.join(results_dir,
                           'predictions.csv'), index=False)
    
def soft_to_hard_labels(labels_soft):
    labels_hard = torch.zeros_like(labels_soft)
    
    _, max_indices = torch.max(labels_soft, dim=1)
    for idx in range(labels_soft.shape[0]):
        labels_hard[idx, max_indices[idx]] = 1.0
        
    return labels_hard