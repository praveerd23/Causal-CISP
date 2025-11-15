"""
Main simulation runner for CAUSAL-CISP (modularized).
Run: python -m causal_cisp.run_simulation
"""

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client_datasets = load_clients()

    clients = []
    for i in range(NUM_CLIENTS):
        model = ResNet18(INPUT_DIM, NUM_CLASSES).to(device)
        loader = DataLoader(client_datasets[i], batch_size=BATCH_SIZE, shuffle=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        clients.append({
            'id': i,
            'model': model,
            'data_loader': loader,
            'optimizer': optimizer,
            'trust': TRUST_INIT
        })

    client_logs = {}
    global_acc = 0.80

    print("\n Client Training Begins")
    all_embeddings = []
    # collect a single batch embedding per client for peer embeddings
    for client in clients:
        try:
            x, _ = next(iter(client['data_loader']))
        except StopIteration:
            # empty loader
            all_embeddings.append(torch.empty(0))
            continue
        x = flatten_cifar(x).to(device)
        with torch.no_grad():
            z, _ = client['model'](x)
            all_embeddings.append(z)

    client_updates = []

    for client in clients:
        peer_z = [z for idx, z in enumerate(all_embeddings) if idx != client['id']]
        # Build a loader that yields flattened tensors on the device
        loader = [(flatten_cifar(x).to(device), y.to(device)) for x, y in client['data_loader']]
        # Train with CSRI for 1 epoch (as per EPOCHS constant)
        csri_training_step(client['model'], loader, peer_z, client['optimizer'], device)

        # Simulate a model update vector (placeholder)
        delta = np.random.randn(INPUT_DIM)
        norm = float(np.linalg.norm(delta))

        # encrypt/noise according to trust
        encrypted_update = encrypt_update_with_trust(delta, client['trust'])

        # simulate client accuracy after local update (placeholder)
        acc_t = global_acc - random.uniform(0.001, 0.01)
        delta_acc = acc_t - global_acc

        client_logs[client['id']] = {
            'grad_norm': norm,
            'acc_t': acc_t,
            'acc_t_minus_1': global_acc
        }
        client_updates.append({
            'id': client['id'],
            'delta': delta,
            'norm': norm
        })

        print(f"Client {client['id']} — ΔAcc: {delta_acc:.4f}, GradNorm: {norm:.4f}")

    
    print(f"\n Avg Classification Loss: {avg_cls_loss:.4f}")
    print(f" Avg Representation Isolation Loss: {avg_iso_loss:.4f}")

    adv_clients = ciaa_detect_adversaries(client_logs, TAU)
    print(f"\n Adversarial Clients Detected: {adv_clients} \n")

    clients_info = []
    for client in client_updates:
        delta_acc = client_logs[client['id']]['acc_t'] - client_logs[client['id']]['acc_t_minus_1']
        clients_info.append({
            'client_id': client['id'],
            'delta_acc': delta_acc,
            'update_norm': client['norm'],
            'trust': clients[client['id']]['trust']
        })

    updated_info = update_trust_and_weights(clients_info, ALPHA, BETA, ETA)
    print(" Trust and Aggregation Weights:")
    trust_vals = []
    weights = []
    for info in updated_info:
        trust_vals.append(info['trust'])
        weights.append(info['weight'])
        # update local client trust state
        clients[info['client_id']]['trust'] = info['trust']
        print(f"Client {info['client_id']} — Trust: {info['trust']:.4f}, Weight: {info['weight']:.4f}")

    trust_avg = sum(trust_vals) / len(trust_vals) if trust_vals else 0.0
    weight_var = float(np.var(weights)) if weights else 0.0
    entropy = -sum(w * np.log(w + 1e-8) for w in weights) if weights else 0.0
    print(f"\n Average Trust Score: {trust_avg:.4f}")
    print(f" Aggregation Weight Variance: {weight_var:.6f}")
    print(f" Weight Distribution Entropy: {entropy:.4f}")

if __name__ == "__main__":
    main()
