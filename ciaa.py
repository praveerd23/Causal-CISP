def ciaa_detect_adversaries(client_logs, threshold_tau):
    """
    Detect adversarial clients using ATE criterion:
    ATE_i = (acc_t - acc_t_minus_1) / (grad_norm + eps)
    If ATE < threshold_tau => adversarial.
    client_logs: dict[client_id] = {'grad_norm': float, 'acc_t': float, 'acc_t_minus_1': float}
    """
    adversarial_clients = set()
    for cid, log in client_logs.items():
        T = float(log.get('grad_norm', 0.0))
        Y = float(log.get('acc_t', 0.0)) - float(log.get('acc_t_minus_1', 0.0))
        ATE = Y / (T + 1e-6)
        if ATE < threshold_tau:
            adversarial_clients.add(cid)
    return adversarial_clients
