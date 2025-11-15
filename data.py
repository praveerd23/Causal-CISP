def load_clients():
    """
    Load CIFAR-10 dataset and partition first NUM_CLIENTS * SAMPLES_PER_CLIENT
    examples equally among NUM_CLIENTS. Returns list of Subset objects. Similarly we can go for EMNIST and MedMNIST
 dataset also.    """
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )

    total_needed = NUM_CLIENTS * SAMPLES_PER_CLIENT
    # ensure we don't exceed dataset length
    total_needed = min(total_needed, len(dataset))

    base_indices = list(range(total_needed))
    client_datasets = []
    for i in range(NUM_CLIENTS):
        start = i * SAMPLES_PER_CLIENT
        end = min((i + 1) * SAMPLES_PER_CLIENT, total_needed)
        if start >= end:
            # no samples left, create empty subset
            client_datasets.append(Subset(dataset, []))
        else:
            indices = base_indices[start:end]
            client_datasets.append(Subset(dataset, indices))

    return client_datasets
