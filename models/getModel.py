import sys

def getModel(model_name,
             num_classes, 
             task_list, 
             use_pretrained_weight, 
             freeze_backbone):
    
    if model_name == 'vgg16':
        from .getVGG16 import multiInputVGG16
        return multiInputVGG16(num_classes, 
                               task_list, 
                               use_pretrained_weight, 
                               freeze_backbone)
    elif model_name == 'conv-att':
        print('Currently being implemented')
        sys.exit(1)
    else:
        print('Other models are not currently supported')
        sys.exit(1)