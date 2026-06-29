"""Checkpoint loading utilities for model state management."""

import os
import warnings

import torch


def load_checkpoint(checkpoint_path, w_tensor, optimizer):
    """
    Charge un checkpoint sauvegardé.
    Note: weights_only=False requis pour la compatibilité avec les objets optimizer.
    """
    if os.path.exists(checkpoint_path):
        # Désactiver le warning FutureWarning pour ne pas effrayer les débutants
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            checkpoint = torch.load(checkpoint_path, weights_only=False)

        w_tensor.data = checkpoint['w'].to(w_tensor.device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"✅ Checkpoint chargé — reprise à l'étape {checkpoint['step']}")
        return checkpoint['step']
    print("ℹ️ Aucun checkpoint trouvé, démarrage depuis zéro.")
    return 0
