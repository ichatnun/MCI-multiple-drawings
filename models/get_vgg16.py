import torch
from torch.nn import Sequential, Conv2d, Linear
from torchvision.models import vgg16, VGG16_Weights

class VGG16Backbone(torch.nn.Module):

    def __init__(self, use_pretrained_weight, freeze_backbone=False):
        super(VGG16Backbone, self).__init__()

        self.hidden_dim = 128 # According to the original paper
        if use_pretrained_weight:
            vgg16_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        else:
            vgg16_model = vgg16(weights=None)
            
        self.final_num_channels = vgg16_model.features[-3].out_channels    
        self.backbone = Sequential(*list(vgg16_model.features.children()))

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False        
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        return self.backbone(x) # (batch, 512, 8, 8)
    
    def get_final_num_channels(self):
        return self.final_num_channels
    
    
class multiInputVGG16(torch.nn.Module):
    
    def __init__(self, 
                 num_classes, 
                 task_list, 
                 use_pretrained_weight, 
                 freeze_backbone=False):
        
        super(multiInputVGG16, self).__init__()
        self.task_list = task_list
        
        # Create the backbone(s)
        self.model_module_dict = torch.nn.ModuleDict()
        for curr_task in self.task_list:
            self.model_module_dict[curr_task] = VGG16Backbone(use_pretrained_weight,
                                                               freeze_backbone)
            
        # Create the final layer
        final_num_channels_one_task = self.model_module_dict[self.task_list[0]].get_final_num_channels()    
        self.final = Linear(final_num_channels_one_task*len(task_list),
                            num_classes,
                            bias=True)

    def forward(self, x):
        
        # Concatenate the output(s) obtained from the backbone(s)
        for idx_task, curr_task in enumerate(self.task_list):
            x_backbone_processed = self.model_module_dict[curr_task](x[curr_task]) # (batch, 512, 8, 8)
            x_backbone_processed = x_backbone_processed.mean([2, 3]) # Global average pooling: (batch_size, 512)

            if idx_task == 0:
                x_concat = x_backbone_processed
            else:
                x_concat = torch.cat((x_concat, x_backbone_processed), axis=-1)
    
        # Pass the concatenated results to the linear classifier (without softmax)
        return self.final(x_concat)