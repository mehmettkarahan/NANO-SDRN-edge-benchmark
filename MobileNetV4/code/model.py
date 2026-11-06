import timm

def create_model(num_classes=3):
    return timm.create_model('mobilenetv4_conv_small',pretrained=False,num_classes=num_classes)
