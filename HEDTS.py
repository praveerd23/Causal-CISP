def he_encrypt(vec):
    """
    Placeholder for homomorphic encryption wrapper. Returns the vector unchanged.
    """
    return vec

def encrypt_update_with_trust(update_vector, trust_score, sigma_max=SIGMA_MAX):
    """
    Add gaussian noise proportional to (1 - trust_score). Return "encrypted" update.
    update_vector: numpy.ndarray
    trust_score: float in [0,1]
    """
    sigma_i = sigma_max * (1 - float(trust_score))
    noise = np.random.normal(0, sigma_i, size=update_vector.shape)
    noisy_update = update_vector + noise
    return he_encrypt(noisy_update)
