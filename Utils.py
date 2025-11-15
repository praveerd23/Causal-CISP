def flatten_cifar(data):
    
    # Flatten CIFAR images tensor of shape (B, C, H, W) to (B, C*H*W)
   
    return data.view(data.size(0), -1)
