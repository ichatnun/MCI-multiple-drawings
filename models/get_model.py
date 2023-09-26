import sys

def get_model(model_name,
              num_classes, 
              task_list, 
              use_pretrained_weight, 
              freeze_backbone):
    
    if model_name == 'vgg16':
        from .get_vgg16 import multiInputVGG16
        return multiInputVGG16(num_classes, 
                               task_list, 
                               use_pretrained_weight, 
                               freeze_backbone)
    elif model_name == 'conv-att':
        from .get_conv_att import multiInputConvAtt
        return multiInputConvAtt(num_classes, 
                                 task_list, 
                                 use_pretrained_weight, 
                                 freeze_backbone)
    else:
        print('Other models are not currently supported')
        sys.exit(1)