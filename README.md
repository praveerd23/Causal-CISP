**Causal Inference with Secure Protocols (Causal-CISP)** introduces a unified and modular defense framework for Federated Learning (FL) that mitigates adversarial risks while preserving privacy, interpretability, and efficiency. Traditional FL defenses are often fragmented, tailored to specific attack scenarios, and prone to performance degradation under heterogeneous, non-identically distributed (non-IID) client data conditions typical in real world deployments.

Causal-CISP addresses these limitations through three core components:

  1. **Causal Inference for Adversarial Attribution (CIAA):** This mechanism estimates **Individual Treatment Effects (ITE)** by analyzing shifts in relative accuracy between consecutive training rounds, enabling transparent and explainable identification of adversarial client updates. CIAA achieves 95% precision and recall on MedMNIST.
  
  2. **Homomorphic Encryption with Dynamic Trust Scaling (HEDTS):** To balance security and efficiency, HEDTS dynamically adjusts encryption strength based on client trust scores. Low-trust clients undergo stricter encryption, while high-trust clients incur minimal cryptographic overhead, reducing communication costs compared to traditional SMPC.
  
  3. **Game Theoretic Incentive and Penalty Model (GTIPM):** Using Stackelberg game theory, GTIPM incentivizes honest participation and penalizes malicious or lazy behavior. This approach achieves near-optimal trust entropy (1.6093 vs. ideal 1.6094) and highly stable trust scores (std. dev. ≤ 0.001), ensuring equitable client contribution even under non-IID conditions.

              Evaluations on CIFAR-10, EMNIST, and MedMNIST under realistic non-IID scenarios demonstrate that Causal-CISP:
              
              * Maintains high model accuracy (83–89%),
              * Reduces attack success rates to ≤0.08,
              * Preserves privacy (PSNR ≤ 0.20) without false positives.


Components:
- CSRI (Causal Separation Representation Isolation)
- HEDTS (Hybrid Encryption–Dynamic Trust Scaling)
- CIAA (Causal Intervention–Driven Adversary Attribution)
- GTIPM (Global Trust–Influenced Probabilistic Aggregation Model)

