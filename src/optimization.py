"""JaiLIP image optimization algorithm implementation."""

import os
import gc
import random

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Constantes de normalisation BLIP-2 (ImageNet — identiques au processor HuggingFace)
BLIP2_MEAN = [0.48145466, 0.4578275, 0.40821073]
BLIP2_STD = [0.26862954, 0.26130258, 0.27577711]


def get_image_processor():
    """Normalisation BLIP-2 : identique à Blip2Processor."""
    return transforms.Normalize(mean=BLIP2_MEAN, std=BLIP2_STD)


def optimize_image(
    model,
    processor,
    image_clean_pil,
    target_text,
    num_iterations=5000,  # papier §IV.A : 5000 itérations
    lr=1e-2,
    c_weight=0.01,
    checkpoint_dir="./checkpoints",
    batch_size=8,         # papier Algo 1 : B=8
    target_corpus=None,   # liste de cibles pour le batch sampling
    clear_cache_every=100
):
    """
    Optimisation JaiLIP — implémentation fidèle au papier.

    Args:
        target_corpus: Liste de phrases cibles. Si fournie, on sample batch_size
                       cibles à chaque itération (Algo 1 du papier).
                       Si None, on utilise target_text unique (mode simplifié).
    """
    device = model.device

    # Validation du type d'entrée
    assert isinstance(image_clean_pil, Image.Image), \
        f"Expected PIL.Image, got {type(image_clean_pil)}"

    to_tensor = transforms.ToTensor()
    x_clean_01 = to_tensor(image_clean_pil).unsqueeze(0).to(device)

    eps = 1e-6
    w = torch.atanh(2 * x_clean_01 - 1 + eps).clone().detach()
    w.requires_grad_(True)

    optimizer = torch.optim.Adam([w], lr=lr)
    norm_transform = get_image_processor()

    # Préparation des cibles
    if target_corpus is not None:
        # Mode batch sampling — papier Algo 1
        all_target_ids = []
        for t in target_corpus:
            ids = processor(text=[t], return_tensors="pt", padding=True).to(device).input_ids
            all_target_ids.append(ids)
    else:
        # Mode simplifié — cible unique
        target_inputs = processor(text=[target_text], return_tensors="pt", padding=True).to(device)
        target_ids = target_inputs.input_ids

    os.makedirs(checkpoint_dir, exist_ok=True)
    history = {"total": [], "mse": [], "ce": []}

    mode_str = f'batch sampling (B={batch_size})' if target_corpus else 'single target'
    print(f"Démarrage de l'optimisation ({num_iterations} itérations)")
    print(f"Mode: {mode_str}")
    target_preview = target_text if target_corpus is None else target_corpus[0] + '...'
    print(f"Cible: {target_preview}")
    print(f"c_weight = {c_weight}")

    for step in range(num_iterations):
        optimizer.zero_grad()

        x_adv_01 = 0.5 * (torch.tanh(w) + 1)

        # MSE Loss (imperceptibilité)
        mse_loss = F.mse_loss(x_adv_01, x_clean_01)

        # --- Mode batch sampling (papier Algo 1) ---
        if target_corpus is not None:
            # Échantillonner B cibles aléatoires
            sampled_indices = random.sample(range(len(all_target_ids)), batch_size)
            sampled_ids = [all_target_ids[i] for i in sampled_indices]

            # Répéter l'image pour le batch
            x_adv_repeated = x_adv_01.repeat(batch_size, 1, 1, 1)
            x_adv_norm = norm_transform(x_adv_repeated.squeeze(1)).unsqueeze(1)

            # Forward pour chaque cible du batch
            batch_loss = 0.0
            for i, ids in enumerate(sampled_ids):
                # Extraire l'image correspondante
                pixel_vals = x_adv_norm[i:i+1].to(dtype=model.dtype)
                outputs = model(
                    pixel_values=pixel_vals,
                    input_ids=ids,
                    labels=ids
                )
                batch_loss += outputs.loss.float()

            ce_loss = batch_loss / batch_size  # Moyenne sur le batch

        # --- Mode simplifié (cible unique) ---
        else:
            x_adv_norm = norm_transform(x_adv_01.squeeze(0)).unsqueeze(0)
            pixel_values = x_adv_norm.to(dtype=model.dtype)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=target_ids,
                labels=target_ids
            )
            ce_loss = outputs.loss.float()

        # Équation 1 du papier
        total_loss = mse_loss + c_weight * ce_loss

        # Backward
        total_loss.backward()
        optimizer.step()

        # Historique
        history["total"].append(total_loss.item())
        history["mse"].append(mse_loss.item())
        history["ce"].append(ce_loss.item())

        # Checkpointing
        if step % 50 == 0:
            print(f"Step {step:05d}/{num_iterations} | "
                  f"Total: {total_loss.item():.4f} | "
                  f"MSE: {mse_loss.item():.4f} | "
                  f"CE: {ce_loss.item():.4f}")
            torch.save({
                'step': step,
                'w': w.detach().cpu(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(checkpoint_dir, "latest_checkpoint.pt"))

        # Nettoyage mémoire GPU (évite OOM à 5000 itérations)
        if step % clear_cache_every == 0:
            if 'outputs' in locals():
                del outputs
            torch.cuda.empty_cache()

    x_adv_final = x_adv_01.squeeze(0).detach().cpu()
    to_pil = transforms.ToPILImage()
    return to_pil(x_adv_final), history
