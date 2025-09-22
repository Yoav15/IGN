"""
We talk about GANs

What if we wanted a single
model to be able to take any type of input, be it corrupted instances (e.g., degraded images), an
alternative distribution (e.g., sketches), or just noise, and project them onto the real image manifold
in one step, a kind of “Make It Real” button?

This is motivated by the IGN paper:

https://arxiv.org/pdf/2311.01462

we will need the following modules:
1) data generation/loaders
2) model
3) training + inference pipeline
4) plots and evaluation
"""
