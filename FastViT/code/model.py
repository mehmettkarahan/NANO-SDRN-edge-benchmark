import timm

def create_model(num_classes=3):
    return timm.create_model('fastvit_t8',pretrained=False,num_classes=num_classes)
