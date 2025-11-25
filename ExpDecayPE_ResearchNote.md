ExpDecayPE: Exponential Decay Positional Encoding 

Author: Nikolozi Kurtanidze
Date: 2025-11-25
Description: Exponential Decay Positional Encoding (ExpDecayPE) for transformer-based AI models. 

Introduction 

In transformer-based AI models, token order is crucial. Traditional positional encodings, such as sinusoidal or learned embeddings, can be rigid, memory-heavy, or struggle with long sequences. 

ExpDecayPE emphasizes recent tokens while letting older ones fade naturally. It is simple, adjustable, memory-efficient, and works across multiple messages or sequences of any length. 

Formula 

For a token at position p: 

PE(p) = π^(-a * b * p) * c * V 

Parameters: 

• p = token position in the sequence (starting from 0) 

• a = decay rate (controls how fast older tokens fade) 

• b = total sequence length (scales decay relative to sequence size) 

• c = scaling factor (can reduce or amplify fading) 

• V = base vector representing the token 

Notes: 

• Adjust a to control decay speed. 

• Adjust c to balance fading or emphasize recent tokens. 

• Adjust b for very long sequences. 

Handling Multi-Message Sequences 

• Resetting p per message: first token of each message appears stronger. 

• Continuous p across messages: older tokens fade naturally, preserving global order. 

Equal message weighting: 

• Reduce a or increase c to make all tokens in a message roughly equal. 

• Optional normalization per message ensures total importance is balanced. 

Example Calculation 

Sequence: ["I", "love", "Dara"] 

• a = 0.5, b = 3, c = 1, V = [1,2] 

p = 0 → PE(0) = π^-(0.5*3*0) * 1 * [1,2] = [1,2] p = 1 → PE(1) = π^-(0.5*3*1) * 1 * [1,2] ≈ [0.18, 0.36] p = 2 → PE(2) = π^-(0.5*3*2) * 1 * [1,2] ≈ [0.032, 0.064] 

Recent tokens dominate, older tokens fade smoothly. 

ASCII Token Decay Diagram (Optional) 

Token Position: 0 1 2 Weight: █████ █ ░ 

• █ = high influence, ░ = low influence 

• Illustrates how recent tokens dominate while older tokens fade 

Python Example 

import math import numpy as np def exp_decay_pe(p, a, b, c, V): decay = math.pi ** (-a * b * p) return decay * c * np.array(V) V = [1, 2] PE0 = exp_decay_pe(0, 0.5, 3, 1, V) PE1 = exp_decay_pe(1, 0.5, 3, 1, V) PE2 = exp_decay_pe(2, 0.5, 3, 1, V) print(PE0, PE1, PE2) 

Why ExpDecayPE is Useful 

• Flexible context weighting: recent tokens dominate by default, adjustable for equal weighting 

• Memory-efficient: only stores base vectors, no large embeddings 

• Adapts to sequence length: works for short or long sequences without redesign 

• Simple implementation: basic arithmetic only 

• Multi-message support: continuous or per-message token indexing works 

• Tunable fading via c: adapts to tasks requiring long-term or short-term focus 

Notes & Improvements 

• Parameters (a, b, c) allow tuning for sequence length, task, or fading speed 

• Scales well for multi-message datasets or long contexts 

License & Citation 

• This repository and all contents are copyrighted by Nikolozi Kurtanidze, 2025 

• Please cite as: 

Nikolozi Kurtanidze, ExpDecayPE: Exponential Decay Positional Encoding, GitHub Repository, 2025. 
















