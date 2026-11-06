import timm

def create_model(num_classes=3):
    return timm.create_model('ghostnetv2_100',pretrained=False,num_classes=num_classes)
