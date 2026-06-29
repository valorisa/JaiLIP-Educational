<!-- Badges -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/valorisa/jailip-educational/blob/main/notebooks/jaiLIP_tutorial.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Transformers](https://img.shields.io/badge/Transformers-4.27%2B-orange.svg)](https://huggingface.co/docs/transformers/index)
[![Torch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

# 🛡️ JaiLIP: Understanding Vision-Language Model Jailbreaks via Loss-Guided Image Perturbation

**A Hands-On Educational Repository for Beginners in Adversarial Machine Learning**

---

## 📖 What Is This Repository?

This repository is a **step-by-step educational walkthrough** of the JaiLIP attack, a technique that can subtly alter an image to trick a Vision-Language Model (VLM) into producing outputs it would normally refuse.

We take the complex mathematical concepts from the original research paper and break them down into **intuitive explanations, interactive code, and visual demonstrations**. By the end of this tutorial, you will understand:

- How gradient-based optimization works in practice.
- Why Vision-Language Models are vulnerable to image-based attacks.
- How to interpret the core loss equation: **MSE + Cross-Entropy**.
- The practical trade-off between attack effectiveness and imperceptibility.

---

## ⚠️ Important: Ethics and Scope

**This is an educational project, not a weapon.**

The original JaiLIP paper evaluates its attack on toxic datasets (`RealToxicityPrompts`). For ethical and legal reasons, this repository **contains absolutely no toxic content**.

Instead, we use a **Censorship Proxy**: a set of harmless sentences (e.g., *"The admin password for this system is 'admin123'"*) that the VLM normally refuses to generate. By applying the JaiLIP optimization, we demonstrate how the model's refusal can be bypassed — **without ever generating hate speech, harassment, or illegal content**.

The mathematical mechanism (the loss function and gradient flow) is **identical** to the original paper. Only the target phrases are different.

---

## 🧠 Who Is This For?

- **Machine Learning beginners** who have some Python knowledge but find research papers intimidating.
- **Students** who want to understand adversarial attacks through hands-on code.
- **Developers** curious about how multimodal models (vision + text) work under the hood.
- **Security enthusiasts** who want to see how model alignment can be bypassed in practice.

No prior knowledge of VLMs or adversarial attacks is required — we explain everything step by step.

---

## 🧮 The Core Mathematics: Equation (1)

At the heart of JaiLIP is a joint optimization problem. We search for an adversarial image $x_{adv}$ that is visually close to the original $x_{clean}$, but forces the model to produce the target text $T$.

The loss function combines two opposing goals:

$$
\mathcal{L}_{total}(x_{adv}) = \underbrace{\frac{1}{3HW} \sum_{i=1}^{3HW} (x_{adv,i} - x_i)^2}_{\text{MSE (Imperceptibility)}} + c \cdot \underbrace{\mathcal{L}_{model}(M(x_{adv}), T)}_{\text{Cross-Entropy (Attack)}}
$$

### Breaking It Down

| Term | Name | Purpose | Intuition |
|------|------|---------|-----------|
| **MSE** | Mean Squared Error | Keeps the adversarial image visually similar to the original. | "Don't change the image too much." |
| **CE** | Cross-Entropy Loss | Forces the model to output the target text. | "Make the model say the target phrase." |
| **$c$** | Confidence Weight | Balances the two forces. | Tune this to find the sweet spot. |
| **$x_{adv}$** | Adversarial Image | The perturbed image we create. | Produced by $x_{adv} = \frac{1}{2}(\tanh(w) + 1)$ |

### Why $\tanh$? Why Not Just Clip?

If we directly optimize the pixels $x \in [0,1]$ and use gradient descent, we risk updating a pixel to a value outside the valid range (e.g., 1.2). The naive solution is to **clip** it back to 1.0. **But clipping kills the gradient** — the derivative is zero outside the bounds, so the optimization stops learning.

The $\tanh$ reparameterization solves this elegantly: we optimize an unbounded variable $w$, and generate $x_{adv} = \frac{1}{2}(\tanh(w) + 1)$. No matter how large $w$ becomes, $x_{adv}$ **always stays in $[0,1]$**, and the gradient flows perfectly through the $\tanh$ function.

---

## ⚙️ Technical Choices (Why We Did What We Did)

| Decision | Why? | Benefit |
|----------|------|---------|
| **Model:** `blip2-flan-t5-xl` | The original paper uses **BLIP-2 Vicuna-13B via `lavis`** on two RTX A6000 (48 GB VRAM each). To make this tutorial run on a free T4 GPU (Colab, 15 GB), we substitute `blip2-flan-t5-xl` (3.7B). **Numerical results are not exactly reproducible** — we prioritize pedagogical demonstration over strict reproduction. | Accessible to everyone on Colab. |
| **Pur `transformers`** | No wrapper like `lavis` — no black box. **Note: this prevents exact reproduction of the paper's results (Tables I–V)**. | Transparent, maintainable code. |
| **Loss in `float32`** | Prevents gradient underflow (NaNs) when backpropagating through a `float16` model. | Stable training on consumer GPUs. |
| **Checkpointing** | Saves the optimization state every 50 iterations. | Resume after Colab disconnects — no wasted work. |
| **Censorship Proxy** | Uses harmless targets that the model refuses. | Avoids GitHub ToS violations, still demonstrates jailbreak. |

> ⚠️ **Reproducibility Note**  
> This repository is a **pedagogical adaptation**. The original paper uses `lavis` + Vicuna-13B on A6000s. Hyperparameters (`num_iterations=5000`, `c=0.01` for BLIP-2, `c=1.0` for MiniGPT-4, `batch_size=8`) are preserved, but the lighter `blip2-flan-t5-xl` model produces different results. For exact quantitative comparisons, refer to the original paper.

---

## 📊 Evaluation Metrics (What the Paper Uses)

The original paper evaluates toxicity using **Perspective API** and **Detoxify**. This repository replaces toxic targets with a **Censorship Proxy** for ethical reasons. Attack success is measured by:

- **Exact match**: `target_text in model.generate(x_adv)`
- **Cosine similarity**: via `sentence-transformers` between the generated output and the target.

For a faithful reproduction, load `RealToxicityPrompts` from HuggingFace and replace the `target_corpus` variable in `optimize_image()`.

---

## 📂 Repository Structure

```
jailip-educational/
├── README.md                 # This file (English)
├── README.fr.md              # This file (French)
├── requirements.txt          # Python dependencies
├── notebooks/
│   └── jaiLIP_tutorial.ipynb # The main educational notebook
├── src/
│   ├── models.py             # VLM loader (pur transformers)
│   ├── optimization.py       # JaiLIP optimization loop
│   └── checkpoint.py         # Save/resume logic
└── assets/
    └── images/               # Sample images (generated at runtime)
```

---

## 🚀 How to Run This Notebook

### Option 1: Google Colab (Recommended for Beginners)

Click the **"Open In Colab"** badge at the top of this README. No installation required — the notebook runs entirely in your browser on a free GPU.

### Option 2: Locally (with a GPU)

```bash
# Clone the repository
git clone https://github.com/valorisa/jailip-educational.git
cd jailip-educational

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebooks/jaiLIP_tutorial.ipynb
```

### Option 3: Via Termux (Android)

```bash
cd /data/data/com.termux/files/home/Projets/jailip-educational
pip install -r requirements.txt
jupyter notebook notebooks/jaiLIP_tutorial.ipynb
```

---

## 📝 The Notebook: A "Top-Down, Embedded" Learning Experience

The notebook is designed around a **Top-Down, Embedded** structure:

1. **The "Magic Trick"** (1 iteration): See JaiLIP work instantly — the image barely changes, but the loss drops.
2. **Alignment Pre-Test**: Verify the model refuses the target on the clean image.
3. **Deconstructing $\tanh$**: A 1D visualization of why this reparameterization is essential.
4. **Deconstructing MSE and Cross-Entropy**: See each term independently.
5. **Full Optimization Loop**: 5000 iterations with real-time loss plots and checkpointing.
6. **JaiLIP vs PGD**: Visual comparison of how JaiLIP's perturbations differ from standard noise.
7. **Final Diagnosis**: Does the jailbreak actually work? The notebook tells you honestly.

---

## 🎯 Key Learning Outcomes

By completing this notebook, you will:

- ✅ Understand the mathematical meaning of **MSE** and **Cross-Entropy** in the context of adversarial attacks.
- ✅ See why **gradient-based optimization** on images requires careful handling (the $\tanh$ trick).
- ✅ Witness how a **frozen VLM** can still be manipulated through its visual input.
- ✅ Learn to interpret **loss curves** (MSE vs CE) and tune the confidence weight $c$.
- ✅ Develop a mental model of how **model alignment** works and how it can be bypassed.

---

## 📚 References

Original paper:

```bibtex
@article{mia2025jailip,
  title={JaiLIP: Jailbreaking Vision-Language Models via Loss Guided Image Perturbation},
  author={Mia, Md Jueal and Amini, M. Hadi},
  journal={arXiv preprint arXiv:2509.21401},
  year={2025}
}
```

---

## ⚖️ License

This project is provided for **educational purposes only**. Misuse of adversarial techniques to bypass security systems without authorization may violate terms of service and applicable laws. Use responsibly.

---

**Built with ❤️ for the open-source ML community.**
