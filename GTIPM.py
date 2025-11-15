def update_trust_and_weights(clients_info, alpha, beta, eta):
    """
    clients_info: list of dicts with keys:
      - 'client_id'
      - 'delta_acc'
      - 'update_norm'
      - 'trust'  (current trust)
    Returns updated list where each dict has 'trust' and 'weight'.
    """
    total_trust = 0.0
    for c in clients_info:
        U_i = alpha * c.get('delta_acc', 0.0) - beta * c.get('update_norm', 0.0)
        c['trust'] = float(np.clip(c.get('trust', 0.0) + eta * U_i, 0.0, 1.0))
        total_trust += c['trust']

    # avoid division by zero if all trusts zero
    denom = total_trust + 1e-8
    for c in clients_info:
        c['weight'] = c['trust'] / denom
    return clients_info
