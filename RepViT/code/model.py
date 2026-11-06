import timm

def create_model(num_classes=3):
    return timm.create_model('repvit_m0_9',pretrained=False,num_classes=num_classes)
