# backend/tts/generator.py
# --- FINAL VERSION - Applying Encodec decode fix and cleanup ---
# TODO: restore causal masks for prod quality after confirming mask=None works

from dataclasses import dataclass
from typing import List, Tuple, Optional
import json
import os
import time

import torch
import torch.nn as nn
# import torchaudio # Removed unused import
from huggingface_hub import hf_hub_download
import models
import csm_utils.loader as loaders
# from tokenizers.processors import TemplateProcessing # Removed unused import
from transformers import AutoTokenizer
from safetensors import safe_open
from models import ModelArgs, Model

# --- Helper functions ---
# Causal mask functions are not used by generate_frame with mask=None,
# but are kept here for potential future re-enabling of masks.
def _create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

def _index_causal_mask(mask: Optional[torch.Tensor], input_pos: torch.Tensor) -> Optional[torch.Tensor]:
    if mask is None: return None
    max_seq_len = mask.size(-1)
    bsz, seq_len = input_pos.shape
    assert torch.all(input_pos < max_seq_len), f"Input position {input_pos.max()} exceeds mask dim {max_seq_len}"
    if mask.device != input_pos.device:
        input_pos = input_pos.to(mask.device)
    input_pos_flat = input_pos.squeeze(0) # Assumes bsz=1
    indexed_mask_rows = mask[input_pos_flat]
    current_kv_seq_len = input_pos.max() + 1
    final_mask = indexed_mask_rows[:, :current_kv_seq_len] # [S_q, S_kv]
    return final_mask

def _multinomial_sample_one_no_sync(probs: torch.Tensor) -> torch.Tensor:
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)

def sample_topk(logits: torch.Tensor, topk: int, temperature: float) -> torch.Tensor:
    logits = logits / max(temperature, 1e-5)
    filter_value: float = -float("Inf")
    k = min(topk, logits.size(-1))
    topk_values, _ = torch.topk(logits, k=k)
    kth_value = topk_values[..., -1, None]
    indices_to_remove = logits < kth_value
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)
    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token


@dataclass
class Segment:
    speaker: int
    text: str
    audio: Optional[torch.Tensor] = None

# --- load_llama3_tokenizer ---
def load_llama3_tokenizer():
    tokenizer_name = "meta-llama/Llama-3.2-1B"
    print(f"Attempting to load tokenizer: {tokenizer_name}")
    hf_token = os.environ.get("HUGGING_FACE_TOKEN")
    print(f"Proceeding with AutoTokenizer.from_pretrained('{tokenizer_name}')...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=hf_token if hf_token else None)
    except Exception as e:
        print(f"ERROR: Failed to load tokenizer '{tokenizer_name}': {e}")
        raise
    if not tokenizer.bos_token or not tokenizer.eos_token:
         print("Warning: Tokenizer might be missing BOS or EOS tokens.")
    print("Tokenizer loaded.")
    return tokenizer

class Generator:
    def __init__(self, model: Model):
        self._model = model
        self._text_tokenizer = load_llama3_tokenizer()
        self.device = next(model.parameters()).device
        print(f"Generator initializing on device: {self.device}")
        try:
            print(f"Attempting to load audio tokenizer (Mimi/Encodec)...")
            mimi = loaders.get_mimi(device=self.device)
            print("Successfully loaded audio tokenizer (Mimi/Encodec).")
            mimi.eval()
            self._audio_tokenizer = mimi
        except Exception as e:
            print(f"ERROR: Failed during audio tokenizer loading: {e}")
            raise RuntimeError(f"Failed to load audio tokenizer: {e}") from e
        self.sample_rate = getattr(self._audio_tokenizer, 'sample_rate', 24000)
        self.frame_rate = getattr(self._audio_tokenizer, 'frame_rate', 75.0)
        self.num_codebooks = self._model.config.audio_num_codebooks
        self.audio_vocab_size = self._model.config.audio_vocab_size
        self.expected_token_cols = self.num_codebooks + 1
        print(f"Generator initialized. Sample rate: {self.sample_rate} Hz, Frame rate: {self.frame_rate} Hz")
        print(f"  Num Codebooks: {self.num_codebooks}, Audio Vocab Size: {self.audio_vocab_size}")

    # --- Embedding methods ---
    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        offset = codebook * self.audio_vocab_size
        tokens_long = tokens.squeeze(-1).long() if tokens.ndim > 1 else tokens.long()
        indices = tokens_long + offset
        try:
            embeddings = self._model.audio_embeddings(indices)
        except IndexError as e:
             print(f"\nERROR (_embed_audio): Index out of bounds.")
             raise e
        return embeddings

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_modalities = tokens.shape
        if num_modalities != self.expected_token_cols:
             raise ValueError(f"Expected {self.expected_token_cols} modalities, got {num_modalities} in tokens shape {tokens.shape}")
        text_tokens = tokens[:, :, -1]
        audio_tokens = tokens[:, :, :self.num_codebooks]
        text_embeds = self._model.text_embeddings(text_tokens.long())
        audio_embeds_sum = torch.zeros_like(text_embeds)
        for i in range(self.num_codebooks):
             cb_tokens = audio_tokens[:, :, i]
             cb_embeds = self._embed_audio(i, cb_tokens)
             audio_embeds_sum += cb_embeds
        total_embeddings = text_embeds + audio_embeds_sum
        return total_embeddings

    # --- Tokenization methods ---
    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}", add_special_tokens=False)
        num_text_tokens = len(text_tokens)
        text_frame = torch.zeros(num_text_tokens, self.expected_token_cols, dtype=torch.long, device=self.device)
        text_frame_mask = torch.zeros(num_text_tokens, self.expected_token_cols, dtype=torch.bool, device=self.device)
        text_frame[:, -1] = torch.tensor(text_tokens, device=self.device)
        text_frame_mask[:, -1] = True
        return text_frame, text_frame_mask

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if audio is None or audio.numel() == 0:
            return torch.empty((0, self.expected_token_cols), dtype=torch.long, device=self.device), torch.empty((0, self.expected_token_cols), dtype=torch.bool, device=self.device)
        if audio.ndim > 1: audio = torch.mean(audio, dim=0) if audio.shape[0] != 1 else audio.squeeze(0)
        assert audio.ndim == 1, f"Audio must be single channel, got ndim={audio.ndim}"
        audio_dev = audio.to(self.device)
        audio_tensor = audio_dev.unsqueeze(0).unsqueeze(0)
        try:
            with torch.no_grad():
                codes_tuple = self._audio_tokenizer.encode(audio_tensor)
            audio_codes = codes_tuple[0][0]
        except Exception as e: print(f"ERROR during audio encoding: {e}"); raise
        audio_tokens_t = audio_codes.squeeze(0).transpose(0, 1)
        num_frames, num_encoded_codebooks = audio_tokens_t.shape
        if num_encoded_codebooks != self.num_codebooks:
             print(f"WARNING: Audio tokenizer generated {num_encoded_codebooks} codebooks, model expects {self.num_codebooks}. Padding/truncating.")
             if num_encoded_codebooks > self.num_codebooks: audio_tokens_t = audio_tokens_t[:, :self.num_codebooks]
             else: padding = torch.zeros((num_frames, self.num_codebooks - num_encoded_codebooks), dtype=audio_tokens_t.dtype, device=self.device); audio_tokens_t = torch.cat([audio_tokens_t, padding], dim=1)
        audio_frame = torch.zeros(num_frames, self.expected_token_cols, dtype=torch.long, device=self.device)
        audio_frame_mask = torch.zeros(num_frames, self.expected_token_cols, dtype=torch.bool, device=self.device)
        audio_frame[:, :self.num_codebooks] = audio_tokens_t
        audio_frame_mask[:, :self.num_codebooks] = True
        return audio_frame, audio_frame_mask

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        if segment.audio is not None and segment.audio.numel() > 0:
            audio_tokens, audio_masks = self._tokenize_audio(segment.audio)
            return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)
        else: return text_tokens, text_masks
    # --- End Tokenization methods ---


    # --- generate_frame METHOD (Passing mask=None to ALL layers) ---
    @torch.inference_mode()
    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
    ) -> torch.Tensor:
        device = tokens.device
        dtype = next(self._model.parameters()).dtype
        b, s_hist = input_pos.shape
        assert b == 1, "generate_frame currently assumes batch size 1"
        input_pos = input_pos.to(device)
        input_embeddings = self._embed_tokens(tokens)

        backbone_input_h = input_embeddings[:, -1:, :]
        backbone_input_pos = input_pos[:, -1:]
        h_backbone = backbone_input_h
        if not hasattr(self._model.backbone, 'layers') or not isinstance(self._model.backbone.layers, nn.ModuleList):
            raise AttributeError("Backbone model structure missing 'layers' attribute.")

        for layer_idx, layer in enumerate(self._model.backbone.layers):
            final_mask_for_layer = None # Pass mask=None
            try:
                h_backbone = layer(h_backbone, input_pos=backbone_input_pos, mask=final_mask_for_layer)
            except Exception as e:
                 print(f"\n--- ERROR in Backbone Layer {layer_idx} ---")
                 print(f"Mask passed to backbone layer {layer_idx}: None")
                 # ... (keep detailed error logging) ...
                 raise RuntimeError(f"Error in backbone layer {layer_idx}") from e

        if hasattr(self._model.backbone, 'norm') and isinstance(self._model.backbone.norm, nn.Module):
            h_backbone = self._model.backbone.norm(h_backbone)
        h_backbone = h_backbone.to(dtype=dtype)
        last_backbone_h = h_backbone.squeeze(1)

        # --- Decoder Logic ---
        c0_logits = self._model.codebook0_head(last_backbone_h)
        c0_sample = sample_topk(c0_logits, topk, temperature)
        c0_embed_backbone = self._embed_audio(0, c0_sample)
        projected_h = self._model.projection(last_backbone_h)
        projected_c0_embed = self._model.projection(c0_embed_backbone)
        generated_samples = [c0_sample.clone()]

        if hasattr(self._model.decoder, 'reset_caches'): self._model.decoder.reset_caches()

        decoder_inputs_proj = [projected_h, projected_c0_embed]
        h_decoder = None
        for step in range(2):
            decoder_step_input = decoder_inputs_proj[step].unsqueeze(1)
            decoder_step_pos = torch.full((b, 1), step, dtype=torch.long, device=device)
            h_decoder_step = decoder_step_input
            if not hasattr(self._model.decoder, 'layers') or not isinstance(self._model.decoder.layers, nn.ModuleList):
                 raise AttributeError("Decoder model structure missing 'layers' attribute.")
            for layer_idx, layer in enumerate(self._model.decoder.layers):
                 final_mask_for_layer = None # Pass mask=None
                 try:
                      h_decoder_step = layer(h_decoder_step, input_pos=decoder_step_pos, mask=final_mask_for_layer)
                 except Exception as e:
                      print(f"\n--- ERROR in Decoder Layer {layer_idx} (Priming Step {step}) ---")
                      print(f"Mask passed to layer: None")
                      # ... (keep detailed error logging) ...
                      raise RuntimeError(f"Error in decoder layer {layer_idx} during priming step {step}") from e
            h_decoder = h_decoder_step

        if hasattr(self._model.decoder, 'norm') and isinstance(self._model.decoder.norm, nn.Module):
             h_decoder = self._model.decoder.norm(h_decoder)
        h_decoder = h_decoder.to(dtype=dtype)
        last_decoder_h = h_decoder.squeeze(1)

        for i in range(1, self.num_codebooks):
            head_index = i - 1
            ci_logits = torch.mm(last_decoder_h, self._model.audio_head[head_index])
            ci_sample = sample_topk(ci_logits, topk, temperature)
            generated_samples.append(ci_sample)
            ci_embed_backbone = self._embed_audio(i, ci_sample)
            ci_embed_decoder = self._model.projection(ci_embed_backbone).unsqueeze(1)
            decoder_step_pos = torch.full((b, 1), i + 1, dtype=torch.long, device=device)
            h_decoder_step = ci_embed_decoder
            for layer_idx, layer in enumerate(self._model.decoder.layers):
                 final_mask_for_layer = None # Pass mask=None
                 try:
                      h_decoder_step = layer(h_decoder_step, input_pos=decoder_step_pos, mask=final_mask_for_layer)
                 except Exception as e:
                      print(f"\n--- ERROR in Decoder Layer {layer_idx} (Generating C{i}) ---")
                      print(f"Mask passed to layer: None")
                      # ... (keep detailed error logging) ...
                      raise RuntimeError(f"Error in decoder layer {layer_idx} generating C{i}") from e
            if hasattr(self._model.decoder, 'norm') and isinstance(self._model.decoder.norm, nn.Module):
                h_decoder_step = self._model.decoder.norm(h_decoder_step)
            last_decoder_h = h_decoder_step.to(dtype=dtype).squeeze(1)

        final_samples = torch.cat(generated_samples, dim=1)
        return final_samples
    # --- END generate_frame ---

    # --- generate METHOD (Applying final Encodec decode fix) ---
    @torch.inference_mode()
    def generate(
        self, text: str, speaker: int, context: List[Segment],
        max_audio_length_ms: float = 90_000, # Default max length
        temperature: float = 0.9, topk: int = 50,
    ) -> torch.Tensor:
        if hasattr(self._model.backbone, 'reset_caches'): self._model.backbone.reset_caches()
        if hasattr(self._model.decoder, 'reset_caches'): self._model.decoder.reset_caches()
        print("KV Caches reset via Generator.generate().")

        actual_max_generation_len = int(max_audio_length_ms / 1000 * self.frame_rate)
        print(f"Target max generation frames: {actual_max_generation_len}")

        # 1. Tokenize Context + Current Text
        prompt_frames = []
        prompt_masks = []
        total_prompt_len = 0
        for segment in context:
            s_tokens, s_masks = self._tokenize_segment(segment)
            prompt_frames.append(s_tokens)
            prompt_masks.append(s_masks)
            total_prompt_len += s_tokens.size(0)
        gen_text_tokens, gen_text_masks = self._tokenize_text_segment(text, speaker)
        prompt_frames.append(gen_text_tokens)
        prompt_masks.append(gen_text_masks)
        total_prompt_len += gen_text_tokens.size(0)
        full_prompt_tokens = torch.cat(prompt_frames, dim=0).unsqueeze(0)
        full_prompt_masks = torch.cat(prompt_masks, dim=0).unsqueeze(0)
        print(f"Total prompt length: {total_prompt_len} frames.")
        max_seq_len = getattr(self._model.backbone, 'max_seq_len', 2048)
        if total_prompt_len >= max_seq_len:
             raise ValueError(f"Prompt too long ({total_prompt_len} frames). Max allowable is {max_seq_len - 1}.")

        # Calculate effective max based on remaining space AND requested max length
        effective_max_gen_len = min(actual_max_generation_len, max_seq_len - total_prompt_len - 1)

        if effective_max_gen_len <= 0:
            print(f"Warning: Prompt length ({total_prompt_len}) or requested max_audio_length leaves no room for generation. Returning empty.")
            return torch.empty(0, device='cpu')
        print(f"Effective max generation frames: {effective_max_gen_len}")

        # 2. Process prompt token-by-token
        print(f"Processing prompt ({total_prompt_len} frames) token-by-token...")
        curr_tokens_hist = []
        curr_masks_hist = []
        for i in range(total_prompt_len):
            print(f"  Processing prompt frame {i+1}/{total_prompt_len}", end='\r')
            single_token_frame = full_prompt_tokens[:, i:i+1, :]
            single_mask_frame = full_prompt_masks[:, i:i+1, :]
            curr_tokens_hist.append(single_token_frame)
            curr_masks_hist.append(single_mask_frame)
            context_tokens = torch.cat(curr_tokens_hist, dim=1)
            context_masks = torch.cat(curr_masks_hist, dim=1)
            context_pos = torch.arange(0, i + 1, device=self.device).unsqueeze(0)
            _ = self.generate_frame(
                tokens=context_tokens, tokens_mask=context_masks, input_pos=context_pos,
                temperature=temperature, topk=topk
            )
        print(f"\nPrompt processing complete.")

        # 3. Autoregressive Generation Loop
        curr_tokens = context_tokens
        curr_masks = context_masks
        curr_pos = context_pos
        generated_audio_codebooks = []
        print(f"Starting generation loop for max {effective_max_gen_len} audio frames...")
        for i in range(effective_max_gen_len):
            print(f"  Gen Step {i+1}/{effective_max_gen_len}, Hist Len: {curr_tokens.size(1)}", end='\r')
            next_audio_frame_codes = self.generate_frame(
                tokens=curr_tokens, tokens_mask=curr_masks, input_pos=curr_pos,
                temperature=temperature, topk=topk
            ).squeeze(0)
            generated_audio_codebooks.append(next_audio_frame_codes)
            next_input_frame = torch.zeros(1, 1, self.expected_token_cols, dtype=torch.long, device=self.device)
            next_input_mask = torch.zeros(1, 1, self.expected_token_cols, dtype=torch.bool, device=self.device)
            next_input_frame[0, 0, :self.num_codebooks] = next_audio_frame_codes
            next_input_mask[0, 0, :self.num_codebooks] = True
            curr_tokens = torch.cat([curr_tokens, next_input_frame], dim=1)
            curr_masks = torch.cat([curr_masks, next_input_mask], dim=1)
            next_pos_val = curr_tokens.size(1) - 1
            next_pos = torch.tensor([[next_pos_val]], device=self.device)
            curr_pos = torch.cat([curr_pos, next_pos], dim=1)
            if curr_tokens.size(1) >= max_seq_len:
                print(f"\nWarning: Reached max sequence length ({max_seq_len}). Stopping generation.")
                break
        else:
             print(f"\nFinished generation loop after {effective_max_gen_len} steps.")

        # 4. Decode generated audio codebooks
        if not generated_audio_codebooks:
            print("Warning: No audio frames generated.")
            return torch.empty(0, device='cpu')

        # --- CORRECTED DECODING INPUT FORMAT (ChatGPT Suggestion + dtype=long) ---
        stacked = torch.stack(generated_audio_codebooks, dim=0)   # [T_gen, N_codebooks]
        # Permute, add batch dim, ensure LONG dtype for embeddings
        codes = stacked.permute(1, 0).unsqueeze(0).long()        # [1, N_codebooks, T_gen]

        # Create scale tensor with matching batch dimension [1, N_codebooks]
        scale = torch.ones((1, codes.size(1)), device=codes.device, dtype=torch.float32)

        # Encodec expects a list containing one tuple: (codes[B, N, T], scale[B, N])
        encoded_frames = [(codes, scale)]
        print(f"\nDecoding {codes.shape[2]} frames using input format: List[Tuple(codes:{codes.shape}, scale:{scale.shape})]")
        # --- END CORRECTION ---

        try:
            with torch.no_grad():
                audio = self._audio_tokenizer.decode(encoded_frames) # Expect [B=1, C=1, T_samples] or [B=1, T_samples]
        except Exception as e:
            print(f"ERROR during audio decoding: {type(e).__name__}: {e}")
            print(f"Input codes shape (passed): {codes.shape}, dtype: {codes.dtype}") # Log dtype
            print(f"Input scale shape (passed): {scale.shape}, dtype: {scale.dtype}")
            # --- Add CUDA sync for safety before re-raising ---
            if self.device.type == 'cuda':
                try:
                    torch.cuda.synchronize() # Wait for any async errors
                    # torch.cuda.empty_cache() # Optional cleanup
                except Exception as sync_err:
                    print(f"  WARNING: Error during CUDA sync/clear after decode error: {sync_err}")
            # --- End CUDA sync ---
            raise

        # Squeeze batch and potentially channel dim, ensure float
        if audio.ndim == 3: # [B, C, T]
             audio = audio.squeeze(1) # -> [B, T]
        audio = audio.squeeze(0).float() # Remove batch -> [T]
        # Move to CPU *after* potential CUDA errors are handled
        audio_cpu = audio.cpu()
        print(f"Generated audio tensor shape: {audio_cpu.shape} ({audio_cpu.shape[0] / self.sample_rate:.2f} seconds)")
        return audio_cpu


# --- load_csm_1b Function (NO CHANGES NEEDED) ---
def load_csm_1b(model_id: str, device: str = "cuda") -> Generator:
    # ... (remains the same) ...
    model_config = ModelArgs(
        backbone_flavor="llama-1B", decoder_flavor="llama-100M",
        text_vocab_size=128_256, audio_vocab_size=2051, audio_num_codebooks=32
    )
    print(f"Using CORRECTED ModelArgs: {model_config}")
    print("Instantiating CSM Model structure...")
    model = Model(config=model_config)
    print("Model structure instantiated.")
    target_filename = "ckpt.safetensors"
    print(f"Loading weights '{target_filename}' from '{model_id}'...")
    cache_dir = os.getenv("HF_HOME")
    try:
        weights_path = hf_hub_download(
            repo_id=model_id, filename=target_filename, cache_dir=cache_dir,
            token=os.getenv("HUGGING_FACE_TOKEN")
        )
    except Exception as e: print(f"ERROR downloading weights: {e}"); raise
    state_dict = {}
    try:
        with safe_open(weights_path, framework="pt", device="cpu") as f:
            for key in f.keys(): state_dict[key] = f.get_tensor(key)
        print(f"State dict loaded from {weights_path} to CPU.")
    except Exception as e: print(f"ERROR loading state dict: {e}"); raise
    print("Loading state dict into model (strict=True)...")
    try:
        model.load_state_dict(state_dict, strict=True)
        print("  State dict loaded successfully (strict=True).")
    except Exception as e:
        print(f"ERROR loading state dict (strict=True): {e}")
        raise RuntimeError(f"Failed loading state dict strictly: {e}") from e
    target_device = torch.device(device)
    dtype = torch.float32
    if target_device.type == 'cuda':
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            print(f"Using dtype {dtype} on CUDA device {target_device}.")
        else:
            print("WARNING: CUDA specified but not available! Using CPU.")
            target_device = torch.device('cpu')
    else:
        print(f"Using dtype {dtype} on CPU device {target_device}.")
    model.to(device=target_device, dtype=dtype)
    model.eval()
    print("Model moved to device and set to eval mode.")
    try:
        print("Setting up model KV caches...")
        model.setup_caches(batch_size=1, dtype=dtype)
        print("KV caches set up successfully.")
    except Exception as e:
        print(f"ERROR setting up caches: {e}")
        raise RuntimeError(f"Failed setting up caches: {e}") from e
    print("Initializing Generator...")
    try:
        generator = Generator(model)
        print("Generator initialization complete.")
        return generator
    except Exception as e: print(f"ERROR initializing Generator: {e}"); raise

# --- END generator.py ---