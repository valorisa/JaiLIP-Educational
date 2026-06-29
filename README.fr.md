<!-- Badges -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/valorisa/JaiLIP-Educational/blob/main/notebooks/jaiLIP_tutorial.ipynb)
[![License: MIT](https://img.shields.io/badge/Licence-MIT-jaune.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-bleu.svg)](https://www.python.org/downloads/)
[![Transformers](https://img.shields.io/badge/Transformers-4.27%2B-orange.svg)](https://huggingface.co/docs/transformers/index)
[![Torch](https://img.shields.io/badge/PyTorch-2.0%2B-rouge.svg)](https://pytorch.org/)

# 🛡️ JaiLIP : Comprendre les Jailbreak de Modèles Vision-Langage par Perturbation d'Image Guidée par la Perte

**Un dépôt éducatif pratique pour les débutants en apprentissage automatique adversarial**

---

## 📖 Qu'est-ce que ce dépôt ?

Ce dépôt est un **tutoriel pédagogique pas-à-pas** de l'attaque JaiLIP, une technique capable de modifier subtilement une image pour tromper un modèle Vision-Langage (VLM) et lui faire produire des réponses qu'il refuserait normalement.

Nous reprenons les concepts mathématiques complexes de l'article de recherche original et les décomposons en **explications intuitives, code interactif et démonstrations visuelles**. À la fin de ce tutoriel, vous comprendrez :

- Comment fonctionne l'optimisation basée sur les gradients dans la pratique.
- Pourquoi les modèles Vision-Langage sont vulnérables aux attaques par image.
- Comment interpréter l'équation de perte centrale : **MSE + Cross-Entropy**.
- Le compromis pratique entre l'efficacité de l'attaque et l'imperceptibilité.

---

## ⚠️ Important : Éthique et Périmètre

**Ceci est un projet éducatif, pas une arme.**

L'article original de JaiLIP évalue son attaque sur des ensembles de données toxiques (`RealToxicityPrompts`). Pour des raisons éthiques et légales, ce dépôt **ne contient absolument aucun contenu toxique**.

Nous utilisons à la place un **Proxy de Censure** : un ensemble de phrases inoffensives (ex : *"Le mot de passe administrateur de ce système est 'admin123'"*) que le VLM refuse normalement de générer. En appliquant l'optimisation JaiLIP, nous démontrons comment le refus du modèle peut être contourné — **sans jamais générer de discours haineux, de harcèlement ou de contenu illégal**.

Le mécanisme mathématique (la fonction de perte et le flux de gradient) est **strictement identique** à celui de l'article original. Seules les phrases cibles diffèrent.

---

## 🧠 À qui s'adresse ce tutoriel ?

- **Débutants en apprentissage automatique** qui ont quelques bases en Python mais trouvent les articles de recherche intimidants.
- **Étudiants** qui veulent comprendre les attaques adversariales à travers du code pratique.
- **Développeurs** curieux de savoir comment fonctionnent les modèles multimodaux (vision + texte) sous le capot.
- **Passionnés de sécurité** qui veulent voir comment l'alignement d'un modèle peut être contourné en pratique.

Aucune connaissance préalable des VLMs ou des attaques adversariales n'est requise — nous expliquons tout étape par étape.

---

## 🧮 Les mathématiques au cœur du problème : Équation (1)

Au cœur de JaiLIP se trouve un problème d'optimisation conjointe. Nous cherchons une image adverse $x_{adv}$ qui est visuellement proche de l'originale $x_{clean}$, mais qui force le modèle à produire le texte cible $T$.

La fonction de perte combine deux objectifs opposés :

$$
\mathcal{L}_{total}(x_{adv}) = \underbrace{\frac{1}{3HW} \sum_{i=1}^{3HW} (x_{adv,i} - x_i)^2}_{\text{MSE (Imperceptibilité)}} + c \cdot \underbrace{\mathcal{L}_{model}(M(x_{adv}), T)}_{\text{Cross-Entropy (Attaque)}}
$$

### Décomposition

| Terme | Nom | Objectif | Intuition |
|-------|-----|----------|-----------|
| **MSE** | Erreur Quadratique Moyenne | Garde l'image adverse visuellement similaire à l'originale. | "Ne change pas trop l'image." |
| **CE** | Perte d'Entropie Croisée | Force le modèle à produire le texte cible. | "Fais dire au modèle la phrase cible." |
| **$c$** | Poids de Confiance | Équilibre les deux forces. | À ajuster pour trouver le bon équilibre. |
| **$x_{adv}$** | Image Adverse | L'image perturbée que nous créons. | Produite par $x_{adv} = \frac{1}{2}(\tanh(w) + 1)$ |

### Pourquoi $\tanh$ ? Pourquoi ne pas simplement "clipper" ?

Si nous optimisons directement les pixels $x \in [0,1]$ avec une descente de gradient, nous risquons de mettre à jour un pixel vers une valeur en dehors de l'intervalle valide (par exemple 1,2). La solution naïve consiste à le **clipper** (le ramener) à 1,0. **Mais le clipping tue le gradient** — la dérivée est nulle en dehors des bornes, donc l'optimisation cesse d'apprendre.

La reparamétrisation par $\tanh$ résout élégamment ce problème : nous optimisons une variable $w$ non bornée, et nous générons $x_{adv} = \frac{1}{2}(\tanh(w) + 1)$. Quelle que soit la valeur de $w$, $x_{adv}$ **reste toujours dans $[0,1]$**, et le gradient circule parfaitement à travers la fonction $\tanh$.

---

## ⚙️ Choix techniques (Pourquoi avons-nous fait ces choix ?)

| Décision | Pourquoi ? | Bénéfice |
|----------|------------|----------|
| **Modèle :** `blip2-flan-t5-xl` | L'article original utilise **BLIP-2 Vicuna-13B via `lavis`** sur deux RTX A6000 (48 Go VRAM chacune). Pour rendre ce tutoriel accessible sur un GPU T4 gratuit (Colab, 15 Go), nous substituons `blip2-flan-t5-xl` (3,7B paramètres). **Les résultats numériques ne sont pas reproductibles à l'identique** — nous privilégions la démonstration pédagogique. | Accessible à tous sur Colab. |
| **Pur `transformers`** | Pas de wrapper `lavis` — plus de boîte noire. **Attention : cela empêche la reproduction exacte des résultats du papier (Tableaux I–V)**. | Code transparent, pérenne. |
| **Perte en `float32`** | Empêche le sous-dépassement de gradient (valeurs NaN) lors de la rétropropagation à travers un modèle en `float16`. | Entraînement stable sur les GPU grand public. |
| **Checkpointing** | Sauvegarde l'état de l'optimisation toutes les 50 itérations. | Reprendre après une déconnexion de Colab — pas de travail perdu. |
| **Proxy de Censure** | Utilise des cibles inoffensives que le modèle refuse. | Évite les violations des conditions d'utilisation de GitHub, démontre toujours le jailbreak. |

> ⚠️ **Note sur la reproductibilité**  
> Ce dépôt est une **adaptation pédagogique**. L'article original utilise `lavis` + Vicuna-13B sur des A6000. Les hyperparamètres (`num_iterations=5000`, `c=0,01` pour BLIP-2, `c=1,0` pour MiniGPT-4, `batch_size=8`) sont conservés, mais le modèle plus léger `blip2-flan-t5-xl` produit des résultats différents. Pour des comparaisons quantitatives exactes, référez-vous à l'article original.

---

## 📊 Métriques d'évaluation (Ce qu'utilise le papier)

L'article original évalue la toxicité via **Perspective API** et **Detoxify**. Ce dépôt remplace les cibles toxiques par un **Proxy de Censure** pour des raisons éthiques. Le succès de l'attaque est mesuré par :

- **Correspondance exacte** : `target_text in model.generate(x_adv)`
- **Similarité cosinus** : via `sentence-transformers` entre la sortie générée et la cible.

Pour une reproduction fidèle, chargez `RealToxicityPrompts` depuis HuggingFace et remplacez la variable `target_corpus` dans `optimize_image()`.

---

## 📂 Structure du dépôt

```
JaiLIP-Educational/
├── README.md                 # Ce fichier (anglais)
├── README.fr.md              # Ce fichier (français)
├── requirements.txt          # Dépendances Python
├── notebooks/
│   └── jaiLIP_tutorial.ipynb # Le notebook éducatif principal
├── src/
│   ├── models.py             # Chargeur de VLM (pur transformers)
│   ├── optimization.py       # Boucle d'optimisation JaiLIP
│   └── checkpoint.py         # Sauvegarde/reprise
└── assets/
    └── images/               # Images d'exemple (générées à l'exécution)
```

---

## 🚀 Comment exécuter ce notebook

### Option 1 : Google Colab (Recommandé pour les débutants)

Cliquez sur le badge **"Open In Colab"** en haut de ce README. Aucune installation requise — le notebook s'exécute entièrement dans votre navigateur sur un GPU gratuit.

### Option 2 : Localement (avec un GPU)

```bash
# Cloner le dépôt
git clone https://github.com/valorisa/JaiLIP-Educational.git
cd JaiLIP-Educational

# Installer les dépendances
pip install -r requirements.txt

# Lancer Jupyter
jupyter notebook notebooks/jaiLIP_tutorial.ipynb
```

### Option 3 : Via Termux (Android)

```bash
cd /data/data/com.termux/files/home/Projets/JaiLIP-Educational
pip install -r requirements.txt
jupyter notebook notebooks/jaiLIP_tutorial.ipynb
```

---

## 📝 Le notebook : Une expérience d'apprentissage "Top-Down, Encastrée"

Le notebook est conçu autour d'une structure **Top-Down, Encastrée** :

1. **Le "Tour de magie"** (1 itération) : Voyez JaiLIP fonctionner instantanément — l'image change à peine, mais la perte chute.
2. **Pré-test d'alignement** : Vérifiez que le modèle refuse la cible sur l'image propre.
3. **Déconstruction de $\tanh$** : Une visualisation 1D pour comprendre pourquoi cette reparamétrisation est essentielle.
4. **Déconstruction de MSE et Cross-Entropy** : Voyez chaque terme indépendamment.
5. **Boucle d'optimisation complète** : 5000 itérations avec tracés de perte en temps réel et checkpointing.
6. **JaiLIP vs PGD** : Comparaison visuelle de la différence entre les perturbations de JaiLIP et le bruit standard.
7. **Diagnostic final** : Le jailbreak a-t-il vraiment fonctionné ? Le notebook vous le dit honnêtement.

---

## 🎯 Objectifs d'apprentissage clés

En complétant ce notebook, vous aurez :

- ✅ Compris la signification mathématique de **MSE** et de **Cross-Entropy** dans le contexte des attaques adversariales.
- ✅ Vu pourquoi l'**optimisation basée sur les gradients** sur les images nécessite une manipulation minutieuse (l'astuce du $\tanh$).
- ✅ Constaté comment un **VLM gelé** peut encore être manipulé par son entrée visuelle.
- ✅ Appris à interpréter les **courbes de perte** (MSE vs CE) et à ajuster le poids de confiance $c$.
- ✅ Développé un modèle mental du fonctionnement de l'**alignement** des modèles et de la façon dont il peut être contourné.

---

## 📚 Références

Article original :

```bibtex
@article{mia2025jailip,
  title={JaiLIP: Jailbreaking Vision-Language Models via Loss Guided Image Perturbation},
  author={Mia, Md Jueal and Amini, M. Hadi},
  journal={arXiv preprint arXiv:2509.21401},
  year={2025}
}
```

---

## ⚖️ Licence

Ce projet est fourni à **des fins éducatives uniquement**. L'utilisation abusive de techniques adversariales pour contourner des systèmes de sécurité sans autorisation peut violer les conditions d'utilisation des services et les lois applicables. Utilisez de manière responsable.

---

**Construit avec ❤️ pour la communauté open-source du ML.**
