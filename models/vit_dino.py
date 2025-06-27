import torch.nn as nn
import timm

def get_dino_vit_s16(num_classes: int = 100):
    """
    Loads the DINO ViT-S/16 pretrained model and adapts it for CIFAR-100 classification.
    
    Args:
        num_classes (int): Number of output classes for classification.
    
    Returns:
        torch.nn.Module: Modified DINO ViT-S/16 model.
    """
    model = timm.create_model('vit_small_patch16_224_dino', pretrained=True)

    # model.head is Identity, so use embed_dim as input feature size
    in_features = model.num_features  # or model.embed_dim
    model.head = nn.Linear(in_features, num_classes)

    return model
