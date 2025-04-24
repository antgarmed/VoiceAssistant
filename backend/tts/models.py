# backend/tts/models.py
# --- FINAL VERSION incorporating enumerate fix for setup_caches ---
# (Using torchtune 0.4.0)

from dataclasses import dataclass
from typing import Tuple # Correct import

import torch
import torch.nn as nn
import torchtune
from huggingface_hub import PyTorchModelHubMixin
from torchtune.models import llama3

# --- Function definitions for model flavors ---
def llama3_2_1B() -> torchtune.modules.transformer.TransformerDecoder:
    """Defines the Llama3.2 1B architecture variant with GQA."""
    return llama3.llama3(
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,       # Query heads
        num_kv_heads=8,     # Key/Value heads (GQA)
        embed_dim=2048,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
    )

def llama3_2_100M() -> torchtune.modules.transformer.TransformerDecoder:
    """Defines the Llama3.2 100M architecture variant (Decoder)."""
    # Corrected based on checkpoint loading errors: uses GQA
    return llama3.llama3(
        vocab_size=128_256,
        num_layers=4,
        num_heads=8,        # Query heads
        num_kv_heads=2,     # CORRECTED based on K/V proj shape error -> GQA
        embed_dim=1024,
        max_seq_len=2048,
        intermediate_dim=8192, # Verify this intermediate dim for 100M size
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
    )

FLAVORS = {
    "llama-1B": llama3_2_1B,
    "llama-100M": llama3_2_100M, # Points to the corrected 100M definition
}

# --- _prepare_transformer Function ---
def _prepare_transformer(model: torchtune.modules.transformer.TransformerDecoder) -> Tuple[torchtune.modules.transformer.TransformerDecoder, int]:
    """
    Prepares a torchtune TransformerDecoder for use in CSM:
    1. Replaces the token embeddings layer with Identity.
    2. Replaces the final output projection layer with Identity.
    Returns the modified model and its embedding dimension.
    """
    embed_dim = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    print(f"Prepared transformer: Replaced tok_embeddings and output with nn.Identity(). Embed Dim: {embed_dim}")
    return model, embed_dim


@dataclass
class ModelArgs:
    """Configuration arguments for the CSM Model."""
    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int # Should be 2051 (set by generator.py)
    audio_num_codebooks: int


# --- Model Class Definition ---
class Model(nn.Module, PyTorchModelHubMixin):
    """
    The CSM Model architecture. Contains backbone, decoder, embeddings, projection, heads.
    The generation logic (generate_frame) resides in the Generator class.
    """
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        self.backbone, self.backbone_dim = _prepare_transformer(FLAVORS[config.backbone_flavor]())
        self.decoder, self.decoder_dim = _prepare_transformer(FLAVORS[config.decoder_flavor]())

        self.text_embeddings = nn.Embedding(config.text_vocab_size, self.backbone_dim)
        self.audio_embeddings = nn.Embedding(config.audio_vocab_size * config.audio_num_codebooks, self.backbone_dim)
        self.projection = nn.Linear(self.backbone_dim, self.decoder_dim, bias=False)
        self.codebook0_head = nn.Linear(self.backbone_dim, config.audio_vocab_size, bias=False)
        self.audio_head = nn.Parameter(
            torch.empty(config.audio_num_codebooks - 1, self.decoder_dim, config.audio_vocab_size)
        )
        nn.init.normal_(self.audio_head, mean=0.0, std=0.02)

        print("CSM Model Initialized:")
        print(f"  Backbone: {config.backbone_flavor}, Dim: {self.backbone_dim}")
        print(f"  Decoder: {config.decoder_flavor}, Dim: {self.decoder_dim}")
        print(f"  Text Embeddings: {config.text_vocab_size} -> {self.backbone_dim}")
        print(f"  Audio Embeddings: {self.audio_embeddings.num_embeddings} ({config.audio_num_codebooks}x{config.audio_vocab_size}) -> {self.backbone_dim}")
        print(f"  Codebook0 Head Out Dim: {self.codebook0_head.out_features}")
        print(f"  Audio Head Shape: {self.audio_head.shape}")

    # --- setup_caches METHOD (Corrected enumerate for backbone loop) ---
    def setup_caches(self, batch_size: int, dtype: torch.dtype) -> None:
        """Sets up KV caches for backbone and decoder and ensures they are on the correct device."""
        device = next(self.parameters()).device
        print(f"Attempting setup_caches with batch_size={batch_size}, dtype={dtype}, device={device}")
        backbone_max_seq_len = getattr(self.backbone, 'max_seq_len', 2048)
        decoder_max_seq_len = getattr(self.decoder, 'max_seq_len', 2048)
        print(f"  (Info) Backbone max_seq_len: {backbone_max_seq_len}")
        print(f"  (Info) Decoder max_seq_len: {decoder_max_seq_len}")

        try:
            # Call torchtune's setup_caches
            if hasattr(self.backbone, 'setup_caches') and callable(self.backbone.setup_caches):
                print(f"  Calling self.backbone.setup_caches(batch_size={batch_size}, dtype={dtype})")
                self.backbone.setup_caches(batch_size=batch_size, dtype=dtype)
            else:
                print("  Warning: self.backbone does not have a callable setup_caches method.")

            if hasattr(self.decoder, 'setup_caches') and callable(self.decoder.setup_caches):
                print(f"  Calling self.decoder.setup_caches(batch_size={batch_size}, dtype={dtype})")
                self.decoder.setup_caches(batch_size=batch_size, dtype=dtype)
            else:
                print("  Warning: self.decoder does not have a callable setup_caches method.")
            print("Internal setup_caches calls completed.")

            # --- BEGIN MANUAL CACHE TENSOR DEVICE FIX ---
            print(f"Manually moving cache tensors to device: {device}...")
            moved_count = 0
            # Iterate through backbone layers using enumerate
            if hasattr(self.backbone, 'layers'):
                # CORRECTED: Added enumerate here
                for layer_idx, layer in enumerate(self.backbone.layers):
                    if hasattr(layer, 'attn') and hasattr(layer.attn, 'kv_cache'):
                        kv_cache = layer.attn.kv_cache
                        # Check k_cache
                        if kv_cache.k_cache is not None and kv_cache.k_cache.device != device:
                             print(f"    Moving backbone layer {layer_idx} k_cache from {kv_cache.k_cache.device} to {device}")
                             kv_cache.k_cache = kv_cache.k_cache.to(device=device)
                             moved_count += 1
                        # Check v_cache
                        if kv_cache.v_cache is not None and kv_cache.v_cache.device != device:
                             print(f"    Moving backbone layer {layer_idx} v_cache from {kv_cache.v_cache.device} to {device}")
                             kv_cache.v_cache = kv_cache.v_cache.to(device=device)
                             moved_count += 1
                        # Check/move cache_pos
                        if hasattr(kv_cache, 'cache_pos') and isinstance(kv_cache.cache_pos, torch.Tensor):
                            if kv_cache.cache_pos.device != device:
                                print(f"    Moving backbone layer {layer_idx} cache_pos from {kv_cache.cache_pos.device} to {device}")
                                kv_cache.cache_pos = kv_cache.cache_pos.to(device=device)
                                moved_count += 1

            # Iterate through decoder layers (already had enumerate)
            if hasattr(self.decoder, 'layers'):
                 for layer_idx, layer in enumerate(self.decoder.layers):
                      if hasattr(layer, 'attn') and hasattr(layer.attn, 'kv_cache'):
                           kv_cache = layer.attn.kv_cache
                           # Check k_cache
                           if kv_cache.k_cache is not None and kv_cache.k_cache.device != device:
                                print(f"    Moving decoder layer {layer_idx} k_cache from {kv_cache.k_cache.device} to {device}")
                                kv_cache.k_cache = kv_cache.k_cache.to(device=device)
                                moved_count += 1
                           # Check v_cache
                           if kv_cache.v_cache is not None and kv_cache.v_cache.device != device:
                                print(f"    Moving decoder layer {layer_idx} v_cache from {kv_cache.v_cache.device} to {device}")
                                kv_cache.v_cache = kv_cache.v_cache.to(device=device)
                                moved_count += 1
                           # Check/move cache_pos
                           if hasattr(kv_cache, 'cache_pos') and isinstance(kv_cache.cache_pos, torch.Tensor):
                               if kv_cache.cache_pos.device != device:
                                   print(f"    Moving decoder layer {layer_idx} cache_pos from {kv_cache.cache_pos.device} to {device}")
                                   kv_cache.cache_pos = kv_cache.cache_pos.to(device=device)
                                   moved_count += 1

            print(f"Manual cache tensor moving complete. Moved {moved_count} tensors.")
            # --- END MANUAL CACHE TENSOR DEVICE FIX ---

        except Exception as e: # Catch any exception during setup
             print(f"ERROR during setup_caches: {type(e).__name__}: {e}")
             # Reraise to stop the application startup correctly
             raise RuntimeError(f"Failed setup_caches with batch_size={batch_size}, dtype={dtype}") from e

    # --- forward METHOD (Not used for inference via Generator) ---
    def forward(self, *args, **kwargs):
        """Defines the forward pass for training (not used by Generator)."""
        raise NotImplementedError("Direct forward pass is not implemented for inference via Generator. Use Generator.generate().")

# --- END models.py ---