import torch
from torch.nn import Sequential, Conv2d, Linear, multiInputConvAtt
from getVGG16 import VGG16Backbone
    
class multiInputConvAtt(torch.nn.Module):
    
    def __init__(self, 
                 num_classes, 
                 task_list, 
                 use_pretrained_weight, 
                 freeze_backbone=False):
        
        super(multiInputConvAtt, self).__init__()
        self.task_list = task_list
        
        # Create the backbone(s)
        self.model_module_dict = torch.nn.ModuleDict()
        for curr_task in self.task_list:
            self.model_module_dict[curr_task] = VGG16Backbone(use_pretrained_weight,
                                                               freeze_backbone)
            
        # Create the final layer
        hidden_dim_one_task = self.model_module_dict[curr_task].get_hidden_dim()    
        self.final = Linear(hidden_dim_one_task*len(task_list),
                            num_classes,
                            bias=True)

    def forward(self, x):
        
        # Concatenate the output(s) obtained from the backbone(s)
        for idx_task, curr_task in enumerate(self.task_list):
            x_backbone_processed = self.model_module_dict[curr_task](x[curr_task])
            if idx_task == 0:
                x_concat = x_backbone_processed
            else:
                x_concat = torch.cat((x_concat, x_backbone_processed), axis=-1)
    
        # Pass the concatenated results to the linear classifier (without softmax)
        return self.final(x_concat)
    
    
        # self.backbone = Sequential(*list(vgg16_model.features.children()),
        #                            Conv2d(final_num_channels, self.hidden_dim, kernel_size=1))