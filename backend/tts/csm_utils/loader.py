# backend/tts/csm_utils/loader.py
import json
import torch
# No longer need hf_hub_download here as encodec_model_24khz handles it internally (?)
# from huggingface_hub import hf_hub_download
# No longer need safe_open
# from safetensors import safe_open

# --- Import changed to base encodec library ---
# Using the correct import path based on encodec library structure
from encodec.model import EncodecModel
# -------------------------------------------------------------------

# Using the factory function, repo/filenames might not be needed explicitly
# DEFAULT_AUDIO_TOKENIZER_REPO = "facebook/encodec_24khz"
# DEFAULT_AUDIO_TOKENIZER_CHECKPOINT = "encodec_24khz-d7cc33bc.th"

# --- load_ckpt function not needed for this loading method ---
# def load_ckpt(path): ...

def get_mimi(device="cpu"):
    """
    Loads the Encodec audio tokenizer model using the base encodec library's
    recommended factory function encodec_model_24khz().
    """
    # The repo_id argument is removed as the factory function targets the specific model
    print(f"Initializing EncodecModel (base encodec library's 24khz factory) on device: {device}")
    try:
        # --- Use the standard factory function from the encodec library ---
        # This function typically returns the model architecture with pre-trained weights loaded.
        model = EncodecModel.encodec_model_24khz()
        print("Instantiated base EncodecModel (24khz factory).")

        # --- Optional: Set target bandwidth (common practice) ---
        # Bandwidths can be 1.5, 3.0, 6.0, 12.0, 24.0 kbps
        # 6.0 kbps is a common default balancing quality and size
        target_bandwidth = 6.0
        model.set_target_bandwidth(target_bandwidth)
        print(f"Set target bandwidth to {target_bandwidth} kbps.")

        # --- Move model to the target device and set to eval mode ---
        model = model.to(device)
        model.eval()
        print(f"Encodec model configured and moved to {device}.")
        return model
    except Exception as e:
        print(f"ERROR loading Encodec model using base 'encodec' library factory: {e}")
        raise

# Add any other helper functions needed by generator.py if they were originally in moshi.models.loaders