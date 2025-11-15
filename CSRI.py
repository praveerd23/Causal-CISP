def compute_csri_loss(zi, peer_embeddings, lambd=LAMBDA):
    """
    Compute CSRI isolation loss as sum of mean cosine similarities
    between zi and each peer embedding. Safe-slice to equal lengths.
    """
    Liso = 0.0
    for zj in peer_embeddings:
        # ensure we compare equal number of vectors
        min_len = min(zi.size(0), zj.size(0))
        if min_len == 0:
            continue
        zi_slice = zi[:min_len]
        zj_slice = zj[:min_len]
        similarity = F.cosine_similarity(zi_slice, zj_slice, dim=1).mean()
        Liso += similarity
    return lambd * Liso

def csri_training_step(model, data_loader, peer_embeddings, optimizer, device):
    """
    Single epoch training over given data_loader for the model with CSRI loss added.
    data_loader: iterable of (x, y) on CPU — code will send to device.
    """
    model.train()
    total_loss = 0.0
    count = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        zi, output = model(x)
        Lcls = F.cross_entropy(output, y)
        Liso = compute_csri_loss(zi, peer_embeddings)
        loss = Lcls + Liso
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += 1
    return total_loss / max(count, 1)
