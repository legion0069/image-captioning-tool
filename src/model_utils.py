import torch
import torchvision.models as models

def load_resnet18_feature_extractor():
    """
    Loads a pretrained ResNet18 and removes the final classification layer.
    Returns a model that outputs feature vectors.
    """
    model = models.resnet18(pretrained=True)
    
    # Remove the final fully connected layer
    model.fc = torch.nn.Identity()
    
    model.eval()  # Set to evaluation mode
    return model
