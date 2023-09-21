import torch
from torchvision.models import vgg16, VGG16_Weights

class VGG16Backbone(torch.nn.Module):

    def __init__(self, use_pretrained_weight, freeze_backbone=False):
        super(VGG16Backbone, self).__init__()

        if use_pretrained_weight:
            self.backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        else:
            self.backbone = vgg16(weights=None)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False        
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        x = self.backbone.features(x)
        x = x.mean([2, 3]) # Global average pooling
        return x #(batch_size, 512)
    
    
class multiInputVGG16(torch.nn.Module):
    def __init__(self, 
                 num_classes, 
                 task_list, 
                 use_pretrained_weight, 
                 freeze_backbone=False):
        super(multiInputVGG16, self).__init__()
        self.task_list = task_list
        self.classif = torch.nn.Linear(512*len(task_list), num_classes, bias=False)
        
        # Create the backbone(s)
        self.model_module_dict = torch.nn.ModuleDict()
        for curr_task in self.task_list:
            self.model_module_dict[curr_task] = VGG16Backbone(use_pretrained_weight,
                                                               freeze_backbone)
    
    def forward(self, x):
        
        # Concatenate the output(s) obtained from the backbone(s)
        for idx_task, curr_task in enumerate(self.task_list):
            x_backbone_processed = self.model_module_dict[curr_task](x[curr_task])
            if idx_task == 0:
                x_concat = x_backbone_processed
            else:
                x_concat = torch.cat((x_concat, x_backbone_processed), axis=-1)
    
        # Pass the concatenated results to the linear classifier (without softmax)
        return self.classif(x_concat)