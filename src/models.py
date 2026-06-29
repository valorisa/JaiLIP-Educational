import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

def load_vlm(model_id="Salesforce/blip2-flan-t5-xl"):
    """
    Charge BLIP-2 en pur transformers.
    Le modèle est en float16 pour tenir sur T4 (15GB), mais le calcul
    de loss se fera en float32 pour éviter les underflows de gradients.
    """
    processor = Blip2Processor.from_pretrained(model_id)

    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model.eval()

    # Désactivation du dropout pour la reproductibilité
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0

    return processor, model
