ExpDecayPE v1.2: π-Based Exponential Decay Positional Encoding with Dual Importance Weighting 

Author: Nikolozi Kurtanidze
Version: 1.2 (Research Documentation Release)
Date: 2025

Abstract 

This work introduces ExpDecayPE v1.2, a π-based exponential decay positional encoding extended with dual importance weighting factors (m₁ for user-side control and m₂ for system-side control).

This extension enables precise control of token influence while retaining the original decay behavior. ExpDecayPE is mathematically simple, computationally efficient, and adaptable for both single sequences and multi-message contexts.

The paper provides theoretical analysis, example calculations, and a reference Python implementation, demonstrating its effectiveness for controllable memory and flexible weighting in sequence models.

Version History 

v1.0 — Original formula:
[
PE(p) = π^{-a·b·p} · c · V
]

v1.0–1.1 — Minor edits, clarifications, and improved README

v1.2 — Extended formula with dual importance weighting:
[
PE(p) = π^{-a·b·p} · c · m₁ · m₂ · V
]
Research-paper-style documentation and examples added.

1. Introduction 

Positional encodings are essential for sequence models such as Transformers to capture token order. Existing methods—such as sinusoidal embeddings (Vaswani et al., 2017), rotary embeddings (Su et al., 2021), and learned embeddings—lack:

Explicit control over the fading of older tokens

Dual-importance weighting, allowing multiple agents (user/system) to influence token significance

Smooth integration into multi-message or conversational memory contexts

ExpDecayPE v1.0 introduced a π-based exponential decay:

[
PE(p) = π^{-a·b·p} · c · V
]

ExpDecayPE v1.2 extends this with dual importance weights (m₁ and m₂):

[
PE(p) = π^{-a·b·p} · c · m₁ · m₂ · V
]

This allows independent control of token importance from user and system perspectives while preserving smooth decay.

2. Related Work 

Sinusoidal PE (Vaswani et al., 2017): Fixed embeddings scale with sequence length but do not allow explicit weighting control.

Learned PE: Optimized during training, but no dual-importance control.

Rotary PE (Su et al., 2021): Efficient for long sequences, improves attention interpolation, but lacks explicit fading control.

ALiBi (Press et al., 2022): Linear attention bias for long sequences, but no dual weighting mechanism.

ExpDecayPE v1.2 introduces a π-based decay with tunable user/system weighting, addressing these limitations.

3. Method 3.1 Core Formula 

For token position p and base vector V:

[
PE(p) = π^{-a·b·p} · c · m₁ · m₂ · V
]

Where:

p — token position

a — decay rate

b — total sequence length

c — fading / anti-fading factor

m₁ — user-side importance

m₂ — system-side importance

V — token embedding vector

3.2 Multi-Message Behavior 

Two modes are supported:

Reset-per-message: Each message starts at p = 0. Useful for independent messages.

Continuous mode: p increments across messages, allowing older tokens to fade gradually.

Equal-weighting mode uses:

small a

larger c

tuned m₁ and m₂

optional normalization per message

3.3 Properties 

Smooth exponential decay controlled by a and b

Adjustable fading via c, m₁, m₂

Simple multiplicative structure, computationally cheap

Scales naturally to high-dimensional token embeddings

4. Experiments and Examples 

Sequence: ["I", "love", "Dara"]
Parameters: a=0.5, b=3, c=1, m₁=1, m₂=1, V = [1, 2]

p Decay Value PE(p) 0 1.000 [1.000, 2.000] 1 0.180 [0.180, 0.360] 2 0.032 [0.032, 0.064] 

Observations:

Token influence decreases exponentially with position

User/system weights can amplify or dampen influence

Works consistently across sequences

4.1 Reference Implementation
import math 
import numpy as np 
def exp_decay_pe(p, a, b, c, m1, m2, V): 
decay = math.pi ** (-a * b * p) 
return decay * c * m1 * m2 * np.array(V) 
5. Discussion 

ExpDecayPE v1.2 provides:

Fine-grained control over token influence

Compatibility with single and multi-message sequences

Flexibility for experimental setups, fairness adjustments, or system/user weighting

It remains lightweight, simple, and fully compatible with transformer architectures.

6. Conclusion 

This work introduces ExpDecayPE v1.2, extending the π-based exponential decay positional encoding with dual importance weights m₁ and m₂. This extension improves controllability and flexibility without altering the core decay mechanism.

Future work includes:

Benchmarking on real models

Visualization of decay curves

Integration into conversational AI systems

Preparing a formal submission for academic publication

7. Copyright Copyright (c) 2025 Nikolozi Kurtanidze All rights reserved. ExpDecayPE and the m₁×m₂ dual-importance extension are the author's original creation. 8. References 

Vaswani, A. et al., Attention is All You Need, 2017

Su, J. et al., RoFormer: Enhanced Transformer with Rotary Positional Embedding, 2021

Press, O. et al., ALiBi: Attention with Linear Biases, 2022

