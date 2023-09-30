import torch, pdb
from torch.nn import Sequential, Conv2d, Linear, Parameter, TransformerEncoder, TransformerEncoderLayer
from .get_vgg16 import VGG16Backbone
    
class multiInputConvAtt(torch.nn.Module):
    
    def __init__(self, 
                 num_classes, 
                 task_list, 
                 use_pretrained_weight, 
                 freeze_backbone=False):
        
        super(multiInputConvAtt, self).__init__()
        self.task_list = task_list
        self.hidden_dim_attn = 128 # Hidden dim of multi-head attention
        self.dim_feed_forward = 512 # Hidden dim of the feed forward layers in the encoder
        self.nhead_attn = 1
        
        # Create the backbone(s)
        self.backbone_module_dict = torch.nn.ModuleDict()
        
        for curr_task in self.task_list:
            self.backbone_module_dict[curr_task] = VGG16Backbone(use_pretrained_weight,
                                                                 freeze_backbone)
    
        backbone_num_channels = self.backbone_module_dict[self.task_list[0]].get_final_num_channels()  
        
        # Create a 1x1 convolution that adjusts the number of channels of the vector returned from the backbone
        self.conv1d = Conv2d(backbone_num_channels, self.hidden_dim_attn, kernel_size=1)
          
        # Create the [CLS] token (shared between the tasks): (batch, seq, feature)
        self.CLS_token = Parameter(torch.randn(1, 1, self.hidden_dim_attn))
            
        # Take the encoder of a transformer
        trans_enc_layer = TransformerEncoderLayer(d_model=self.hidden_dim_attn,
                                                  nhead=self.nhead_attn,
                                                  dim_feedforward=self.dim_feed_forward,
                                                  batch_first=True, 
                                                  norm_first=False)

        self.transformer_enc = TransformerEncoder(trans_enc_layer,
                                                  num_layers=3)
          
        # Create the final layer (process the concatenated results)   
        self.final = Linear(self.hidden_dim_attn*len(task_list),
                            num_classes,
                            bias=True)

    def forward(self, x):
        
        # Concatenate the output(s) obtained from the backbone(s)
        for idx_task, curr_task in enumerate(self.task_list):
            
            # Extract the features using the backbone->(batch,512,8,8)
            x_backbone_proc = self.backbone_module_dict[curr_task](x[curr_task])
            
            # Change the feature dimenion using a 1x1 conv layer->(batch,128,8,8)
            x_backbone_proc = self.conv1d(x_backbone_proc)
            
            # Combine the spatial dimensions->(batch,feature,seq)=(batch,128,64)
            x_tokens = torch.flatten(x_backbone_proc, -2, -1)
            
            # Swap the last two dims->(batch,seq,feature)=(batch,64,128)
            x_tokens = torch.transpose(x_tokens, -2, -1)
            
            # Add the [CLS] token->(batch,seq,feature)=(batch,1+64,128)
            x_tokens = torch.cat((self.CLS_token.tile(x_tokens.shape[0], 1, 1), x_tokens), dim=1)
            
            # Perform self attention->(batch,seq,feature)=(batch,1+64,128)
            x_self_attn = self.transformer_enc(x_tokens)
            
            # Get the final representation at the [CLS] location
            x_final_curr_task = x_self_attn[:, 0, :]
            
            # Combine the features from the tasks
            if idx_task == 0:
                x_concat = x_final_curr_task
            else:
                x_concat = torch.cat((x_concat, x_final_curr_task), axis=-1)
    
        # Pass the concatenated results to the linear classifier (without softmax)
        return self.final(x_concat)