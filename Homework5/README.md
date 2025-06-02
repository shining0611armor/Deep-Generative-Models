
# 🧠 Homework 5: Deep Generative Models (Energy-Based Models, JEM & JEM++, DVAEs, GANs)

This repository contains Homework 5 for the graduate-level course **Deep Generative Models**, taught by **Dr. B. Nasihatkon** at **K. N. Toosi University of Technology** during **Spring 1404 (2025)**.

The course is aimed at Master’s and PhD students and focuses on state-of-the-art generative modeling techniques using deep learning frameworks, with an emphasis on energy-based modeling, joint generative-classification frameworks, discrete representations, and adversarial learning.

---

## 🔋 Coding Practice 1: Energy-Based Models (EBMs)

**Goal:** Train energy-based models for both unconditional (CIFAR-10) and conditional (MNIST) image generation.

### 📌 Topics Covered:
- Unnormalized energy modeling via a ResNet-style scalar-valued network `E(x)`
- Sampling via **Stochastic Langevin Dynamics (SLD)**
- **Contrastive Divergence** loss for training EBMs
- Comparison between:
  - Random noise initialization
  - Replay buffer with 95% reused and 5% random samples
- Implementation of a **noise schedule**:  
  \[
  \sigma_t = \sigma_{start} \left(\frac{\sigma_{end}}{\sigma_{start}}\right)^{\frac{t}{T}}
  \]
- Class-conditional EBM using `E(x, y)` on MNIST
- Evaluation via convergence behavior and quality of generated images

---

## 🔐 Coding Practice 2: JEM & JEM++

**Goal:** Convert a classifier into a joint energy-based generative model using the **JEM** and **JEM++** approaches.

### 📌 Topics Covered:
- Use of a pretrained classifier (ResNet-18 or custom CNN)
- Energy formulation from classifier logits
- Loss: combination of **cross-entropy** and **maximum likelihood** terms via **SGLD**
- Maintain classification accuracy while enabling generation
- **Modifications in JEM++**:
  - Initialization via **Gaussian Mixture Model (GMM)**
  - **Proximal SGLD** for improved sampling
- Evaluate:
  - Classification accuracy
  - Quality of generated samples
  - Impact of JEM++ modifications

---

## 🔠 Coding Practice 3: Discrete Variational Autoencoders (DVAEs)

**Goal:** Train DVAEs on MNIST/Fashion-MNIST and explore clustering in the latent space.

### 📌 Topics Covered:
- DVAE architecture with vector quantization (VQ)
- Latent code extraction from encoder output
- Apply **KMeans** clustering to quantized tokens
- Evaluate cluster quality using:
  - **Silhouette Score**
  - **Confusion matrix** vs. ground truth
  - **Label distribution across clusters**
  - **t-SNE** visualization
- Compare classification performance:
  - Using **true labels**
  - Using **pseudo-labels** from clustering

---

## 🎨 Coding Practice 4: GANs for Persian Digit Generation

**Goal:** Implement and evaluate GAN and WGAN models to generate binary Persian digits (۰–۹).

### 📌 Topics Covered:
- Binary image preprocessing and resizing (28×28)
- Standard GAN with binary cross-entropy loss
- WGAN implementation with **Wasserstein loss** and weight clipping
- Evaluate and compare:
  - Image quality
  - Training stability
  - Convergence behavior
- Visualizations:
  - Generated samples from both GAN and WGAN
  - Training loss curves

---

## 📘 Course Information

- **Course Title:** Deep Generative Models  
- **Level:** Master’s & PhD  
- **Institution:** K. N. Toosi University of Technology  
- **Instructor:** Dr. B. Nasihatkon  
- **Homework Designer:** Mehran Tamjidi (Head Teaching Assistant)  
- **Framework:** All implementations in **PyTorch**

---

## 📚 References

- Du, Y., & Mordatch, I. (2019). *Implicit generation and modeling with energy based models*. NeurIPS.
- Grathwohl, W., et al. (2019). *Your classifier is secretly an energy based model and you should treat it like one*. arXiv:1912.03263.
- Yang, X., & Ji, S. (2021). *JEM++: Improved techniques for training JEM*. ICCV.
- Arjovsky, M., et al. (2017). *Wasserstein Generative Adversarial Networks*. ICML.

---

## 🛠️ Notes

This project is structured with reproducibility and interpretability in mind. Although some concepts and architectures draw inspiration from official TensorFlow implementations, **all models are implemented from scratch using PyTorch**.

Feel free to explore each subfolder for implementation details, experiment logs, and visualizations.

---

