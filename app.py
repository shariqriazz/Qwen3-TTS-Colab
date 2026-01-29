# %cd /content/Qwen3-TTS-Colab
from subtitle import subtitle_maker
from process_text import text_chunk
from qwen_tts import Qwen3TTSModel
import subprocess
import os
import gradio as gr
import numpy as np
import torch
import soundfile as sf
import random
import json
import shutil
import zipfile
from datetime import datetime
from uuid import uuid4
from pydub import AudioSegment
from pydub.silence import split_on_silence
from huggingface_hub import snapshot_download, scan_cache_dir
from hf_downloader import download_model
import gc 
from huggingface_hub import login

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
  HF_TOKEN=None

# Global model holders
loaded_models = {}
MODEL_SIZES = ["1.7B"]

# Speaker and language choices
SPEAKERS = [
    "Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"
]
LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Russian"]

AVAILABLE_MODELS = {
    "VoiceDesign": {
        "sizes": ["1.7B"],
        "description": "Create custom voices using natural language descriptions",
    },
    "Base": {
        "sizes": ["1.7B"],
        "description": "Voice cloning from reference audio",
    },
    "CustomVoice": {
        "sizes": ["1.7B"],
        "description": "TTS with predefined speakers and style instructions",
    },
}


# --- Helper Functions ---

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def resolve_seed(seed):
    """Resolve seed input; -1 or None => random seed."""
    try:
        seed = int(seed)
    except Exception:
        seed = -1
    if seed == -1:
        seed = random.randint(0, 2147483647)
    return seed

def get_model_path(model_type: str, model_size: str) -> str:
    """Get model path based on type and size."""
    try:
      return snapshot_download(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}")
    except Exception as e:
      return download_model(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}", download_folder="./qwen_tts_model", redownload= False)

# Auto-download all available models on startup (set QWEN3_TTS_SKIP_AUTO_DOWNLOAD=1 to skip)
if os.getenv("QWEN3_TTS_SKIP_AUTO_DOWNLOAD") != "1":
    for _mtype, _info in AVAILABLE_MODELS.items():
        for _msize in _info.get("sizes", []):
            try:
                _ = get_model_path(_mtype, _msize)
            except Exception as e:
                print(f"Auto-download failed for {_mtype} {_msize}: {e}")

def get_available_sizes(model_type: str):
    sizes = AVAILABLE_MODELS.get(model_type, {}).get("sizes", [])
    return gr.update(choices=sizes, value=sizes[0] if sizes else None)

def check_model_downloaded(model_type: str, model_size: str) -> bool:
    repo_id = f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}"
    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == repo_id:
                return True
    except Exception:
        pass
    local_dir = os.path.join("qwen_tts_model", repo_id.split("/")[-1])
    return os.path.isdir(local_dir)

def get_downloaded_models_status() -> str:
    lines = ["### Model Download Status\n"]
    for model_type, info in AVAILABLE_MODELS.items():
        lines.append(f"**{model_type}** - {info['description']}")
        for size in info["sizes"]:
            status = "✅ Downloaded" if check_model_downloaded(model_type, size) else "⬜ Not downloaded"
            lines.append(f"  - {size}: {status}")
        lines.append("")
    return "\n".join(lines)

def download_model_ui(model_type: str, model_size: str):
    if model_size not in AVAILABLE_MODELS.get(model_type, {}).get("sizes", []):
        return f"❌ Invalid combination: {model_type} {model_size}", get_downloaded_models_status()
    if check_model_downloaded(model_type, model_size):
        return f"✅ {model_type} {model_size} is already downloaded!", get_downloaded_models_status()
    try:
        _ = get_model_path(model_type, model_size)
        return f"✅ Successfully downloaded {model_type} {model_size}!", get_downloaded_models_status()
    except Exception as e:
        return f"❌ Error downloading {model_type} {model_size}: {str(e)}", get_downloaded_models_status()

def get_loaded_models_status() -> str:
    if not loaded_models:
        return "No models currently loaded in memory."
    lines = ["**Currently loaded models:**"]
    for (model_type, model_size) in loaded_models.keys():
        lines.append(f"- {model_type} ({model_size})")
    return "\n".join(lines)

def load_model_ui(model_type: str, model_size: str):
    if model_size not in AVAILABLE_MODELS.get(model_type, {}).get("sizes", []):
        return f"❌ Invalid combination: {model_type} {model_size}", get_loaded_models_status()
    key = (model_type, model_size)
    if key in loaded_models:
        return f"✅ {model_type} {model_size} is already loaded!", get_loaded_models_status()
    try:
        _ = get_model(model_type, model_size)
        return f"✅ Successfully loaded {model_type} {model_size}!", get_loaded_models_status()
    except Exception as e:
        return f"❌ Error loading {model_type} {model_size}: {str(e)}", get_loaded_models_status()

def unload_all_models_ui():
    if not loaded_models:
        return "⚠️ No models are currently loaded.", get_loaded_models_status()
    count = len(loaded_models)
    clear_other_models(keep_key=None)
    return f"✅ Unloaded {count} model(s).", get_loaded_models_status()

def clear_other_models(keep_key=None):
    """Delete all loaded models except the current one."""
    global loaded_models
    keys_to_delete = [k for k in loaded_models if k != keep_key]
    for k in keys_to_delete:
        try:
            del loaded_models[k]
        except Exception:
            pass
    for k in keys_to_delete:
        loaded_models.pop(k, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_model(model_type: str, model_size: str):
    """Load model and clear others to avoid OOM in Colab."""
    global loaded_models
    key = (model_type, model_size)
    if key in loaded_models:
        return loaded_models[key]
    
    clear_other_models(keep_key=key)
    model_path = get_model_path(model_type, model_size)
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cuda",
        dtype=torch.bfloat16,
    )
    loaded_models[key] = model
    return model

def _normalize_audio(wav, eps=1e-12, clip=True):
    """Normalize audio to float32 in [-1, 1] range."""
    x = np.asarray(wav)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")
    if clip:
        y = np.clip(y, -1.0, 1.0)
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)
    return y

def _audio_to_tuple(audio):
    """Convert Gradio audio input to (wav, sr) tuple."""
    if audio is None: return None
    if isinstance(audio, str):
        try:
            wav, sr = sf.read(audio)
            wav = _normalize_audio(wav)
            return wav, int(sr)
        except Exception as e:
            print(f"Error reading audio file: {e}")
            return None
    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)
    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr
    return None

def transcribe_reference(audio_path, mode_input, language="English"):
    """Uses subtitle_maker to extract text from the reference audio."""
    should_run = False
    if isinstance(mode_input, bool): should_run = mode_input
    elif isinstance(mode_input, str) and "High-Quality" in mode_input: should_run = True

    if not audio_path or not should_run: return gr.update()
    
    print(f"Starting transcription for: {audio_path}")
    src_lang = language if language != "Auto" else "English"
    try:
        results = subtitle_maker(audio_path, src_lang)
        transcript = results[7]
        return transcript if transcript else "Could not detect speech."
    except Exception as e:
        print(f"Transcription Error: {e}")
        return f"Error during transcription: {str(e)}"

# --- Audio Processing Utils (Disk Based) ---

def remove_silence_function(file_path, minimum_silence=100):
    """Removes silence from an audio file using Pydub."""
    try:
        output_path = file_path.replace(".wav", "_no_silence.wav")
        sound = AudioSegment.from_wav(file_path)
        audio_chunks = split_on_silence(sound,
                                        min_silence_len=minimum_silence,
                                        silence_thresh=-45,
                                        keep_silence=50)
        combined = AudioSegment.empty()
        for chunk in audio_chunks:
            combined += chunk
        combined.export(output_path, format="wav")
        return output_path
    except Exception as e:
        print(f"Error removing silence: {e}")
        return file_path

def process_audio_output(audio_path, make_subtitle, remove_silence, language="Auto"):
    """Handles Silence Removal and Subtitle Generation."""
    # 1. Remove Silence
    final_audio_path = audio_path
    if remove_silence:
        final_audio_path = remove_silence_function(audio_path)
    
    # 2. Generate Subtitles
    default_srt, custom_srt, word_srt, shorts_srt = None, None, None, None
    if make_subtitle:
        try:
            results = subtitle_maker(final_audio_path, language)
            default_srt = results[0]
            custom_srt = results[1]
            word_srt = results[2]
            shorts_srt = results[3]
        except Exception as e:
            print(f"Subtitle generation error: {e}")

    return final_audio_path, default_srt, custom_srt, word_srt, shorts_srt

def stitch_chunk_files(chunk_files, output_filename, gap_seconds=0.0):
    """
    Takes a list of file paths.
    Stitches them into one file.
    Deletes the temporary chunk files.
    """
    if not chunk_files:
        return None

    combined_audio = AudioSegment.empty()
    
    print(f"Stitching {len(chunk_files)} audio files...")
    for i, f in enumerate(chunk_files):
        try:
            segment = AudioSegment.from_wav(f)
            combined_audio += segment
            if gap_seconds and i < len(chunk_files) - 1:
                combined_audio += AudioSegment.silent(duration=int(gap_seconds * 1000))
        except Exception as e:
            print(f"Error appending chunk {f}: {e}")

    # output_filename = f"final_output_{os.getpid()}.wav"
    combined_audio.export(output_filename, format="wav")
    
    # Clean up temp files
    for f in chunk_files:
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception as e:
            print(f"Warning: Could not delete temp file {f}: {e}")
            
    return output_filename

def _write_text(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content or "")

def create_bundle(bundle_prefix: str, audio_path: str, meta: dict, subtitle_paths: list, extra_paths: list = None):
    """Create a zip bundle with audio, metadata, text, prompt, and subtitles."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prefix = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in (bundle_prefix or "bundle"))
    bundle_dir = os.path.join("./ai_tts_voice", f"bundle_{safe_prefix}_{ts}_{uuid4().hex[:8]}")
    os.makedirs(bundle_dir, exist_ok=True)

    # Save metadata
    manifest_path = os.path.join(bundle_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Optional readable files
    if meta.get("text"):
        _write_text(os.path.join(bundle_dir, "text.txt"), meta.get("text", ""))
    if meta.get("prompt"):
        _write_text(os.path.join(bundle_dir, "prompt.txt"), meta.get("prompt", ""))
    if meta.get("reference_text"):
        _write_text(os.path.join(bundle_dir, "reference_text.txt"), meta.get("reference_text", ""))

    # Copy audio
    if audio_path and os.path.exists(audio_path):
        ext = os.path.splitext(audio_path)[1] or ".wav"
        shutil.copy2(audio_path, os.path.join(bundle_dir, f"audio{ext}"))

    # Copy subtitles
    if subtitle_paths:
        subs_dir = os.path.join(bundle_dir, "subtitles")
        os.makedirs(subs_dir, exist_ok=True)
        for p in subtitle_paths:
            if p and os.path.exists(p):
                shutil.copy2(p, os.path.join(subs_dir, os.path.basename(p)))

    # Extra files (e.g., reference audio)
    if extra_paths:
        extra_dir = os.path.join(bundle_dir, "extras")
        os.makedirs(extra_dir, exist_ok=True)
        for p in extra_paths:
            if p and os.path.exists(p):
                shutil.copy2(p, os.path.join(extra_dir, os.path.basename(p)))

    # Zip it
    zip_path = f"{bundle_dir}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(bundle_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, bundle_dir)
                zf.write(full_path, rel_path)

    return zip_path

# --- Generators (Memory Optimized) ---

def generate_voice_design(text, language, voice_description, remove_silence, make_subs, seed, chunk_size, chunk_gap, consistent_voice, base_model_size):
    if not text or not text.strip(): return None, "Error: Text is required.", None, None, None, None, seed, None
    
    try:
        seed = resolve_seed(seed)
        # 1. Chunk Text
        text_chunks, tts_filename = text_chunk(text, language, char_limit=int(chunk_size))
        print(f"Processing {len(text_chunks)} chunks...")
        
        chunk_files = []
        mode_label = "VoiceDesign"

        if consistent_voice and len(text_chunks) > 1:
            # Step 1: Generate a reference voice using VoiceDesign
            tts_design = get_model("VoiceDesign", "1.7B")
            ref_text = text_chunks[0].strip()
            set_seed(seed)
            ref_wavs, ref_sr = tts_design.generate_voice_design(
                text=ref_text,
                language=language,
                instruct=voice_description.strip(),
                non_streaming_mode=True,
                max_new_tokens=2048,
            )
            ref_audio_tuple = (ref_wavs[0], ref_sr)
            del ref_wavs
            torch.cuda.empty_cache()
            gc.collect()

            # Step 2: Clone that reference voice for all chunks
            tts_base = get_model("Base", base_model_size)
            for i, chunk in enumerate(text_chunks):
                set_seed(seed)
                wavs, sr = tts_base.generate_voice_clone(
                    text=chunk.strip(),
                    language=language,
                    ref_audio=ref_audio_tuple,
                    ref_text=ref_text,
                    x_vector_only_mode=False,
                    max_new_tokens=2048,
                )
                temp_filename = f"temp_chunk_{i}_{os.getpid()}.wav"
                sf.write(temp_filename, wavs[0], sr)
                chunk_files.append(temp_filename)

                del wavs
                torch.cuda.empty_cache()
                gc.collect()

            mode_label = f"Consistent Voice (VoiceDesign→Base {base_model_size})"
        else:
            tts = get_model("VoiceDesign", "1.7B")

            # 2. Generate & Save Loop
            for i, chunk in enumerate(text_chunks):
                set_seed(seed)
                wavs, sr = tts.generate_voice_design(
                    text=chunk.strip(),
                    language=language,
                    instruct=voice_description.strip(),
                    non_streaming_mode=True,
                    max_new_tokens=2048,
                )
                
                # Save immediately to disk
                temp_filename = f"temp_chunk_{i}_{os.getpid()}.wav"
                sf.write(temp_filename, wavs[0], sr)
                chunk_files.append(temp_filename)
                
                # Clear memory
                del wavs
                torch.cuda.empty_cache()
                gc.collect()
        
        # 3. Stitch from disk
        stitched_file = stitch_chunk_files(chunk_files, tts_filename, gap_seconds=chunk_gap)
        
        # 4. Post-Process
        final_audio, srt1, srt2, srt3, srt4 = process_audio_output(stitched_file, make_subs, remove_silence, language)
        bundle_meta = {
            "mode": "VoiceDesign",
            "consistent_voice": bool(consistent_voice),
            "base_model_size": base_model_size,
            "model_used": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign" if not consistent_voice else f"Qwen/Qwen3-TTS-12Hz-{base_model_size}-Base",
            "language": language,
            "seed": seed,
            "chunk_size": int(chunk_size),
            "chunk_gap": float(chunk_gap),
            "text": text.strip(),
            "prompt": voice_description.strip(),
        }
        bundle_path = create_bundle(
            "voice_design",
            final_audio,
            bundle_meta,
            [srt1, srt2, srt3, srt4],
        )
        
        status = f"Generation Success! Seed: {seed} | Mode: {mode_label}"
        return final_audio, status, srt1, srt2, srt3, srt4, seed, final_audio, bundle_path

    except Exception as e:
        return None, f"Error: {e}", None, None, None, None, seed if 'seed' in locals() else -1, None, None

def generate_custom_voice(text, language, speaker, instruct, model_size, remove_silence, make_subs, seed, chunk_size, chunk_gap):
    if not text or not text.strip(): return None, "Error: Text is required.", None, None, None, None, seed, None
    
    try:
        seed = resolve_seed(seed)
        text_chunks, tts_filename = text_chunk(text, language, char_limit=int(chunk_size))
        chunk_files = []
        tts = get_model("CustomVoice", model_size)
        formatted_speaker = speaker.lower().replace(" ", "_")

        for i, chunk in enumerate(text_chunks):
            set_seed(seed)
            wavs, sr = tts.generate_custom_voice(
                text=chunk.strip(),
                language=language,
                speaker=formatted_speaker,
                instruct=instruct.strip() if instruct else None,
                non_streaming_mode=True,
                max_new_tokens=2048,
            )
            # Save immediately
            temp_filename = f"temp_custom_{i}_{os.getpid()}.wav"
            sf.write(temp_filename, wavs[0], sr)
            chunk_files.append(temp_filename)
            
            # Clear memory
            del wavs
            torch.cuda.empty_cache()
            gc.collect()
            
        stitched_file = stitch_chunk_files(chunk_files, tts_filename, gap_seconds=chunk_gap)
        final_audio, srt1, srt2, srt3, srt4 = process_audio_output(stitched_file, make_subs, remove_silence, language)
        bundle_meta = {
            "mode": "CustomVoice",
            "model_used": f"Qwen/Qwen3-TTS-12Hz-{model_size}-CustomVoice",
            "language": language,
            "speaker": speaker,
            "seed": seed,
            "chunk_size": int(chunk_size),
            "chunk_gap": float(chunk_gap),
            "text": text.strip(),
            "prompt": (instruct or "").strip(),
        }
        bundle_path = create_bundle(
            "custom_voice",
            final_audio,
            bundle_meta,
            [srt1, srt2, srt3, srt4],
        )
        status = f"Generation Success! Seed: {seed}"
        return final_audio, status, srt1, srt2, srt3, srt4, seed, final_audio, bundle_path

    except Exception as e:
        return None, f"Error: {e}", None, None, None, None, seed if 'seed' in locals() else -1, None, None

def smart_generate_clone(ref_audio, ref_text, target_text, language, mode, model_size, remove_silence, make_subs, seed, chunk_size, chunk_gap):
    if not target_text or not target_text.strip(): return None, "Error: Target text is required.", None, None, None, None, seed, None
    if not ref_audio: return None, "Error: Ref audio required.", None, None, None, None, seed, None

    # 1. Mode & Transcript Logic
    use_xvector_only = ("Fast" in mode)
    final_ref_text = ref_text
    audio_tuple = _audio_to_tuple(ref_audio)

    if not use_xvector_only:
        if not final_ref_text or not final_ref_text.strip():
            print("Auto-transcribing reference...")
            try:
                final_ref_text = transcribe_reference(ref_audio, True, language)
                if not final_ref_text or "Error" in final_ref_text:
                     return None, f"Transcription failed: {final_ref_text}", None, None, None, None, seed, None
            except Exception as e:
                return None, f"Transcribe Error: {e}", None, None, None, None, seed, None
    else:
        final_ref_text = None

    try:
        # 2. Chunk Target Text
        seed = resolve_seed(seed)
        text_chunks, tts_filename = text_chunk(target_text, language, char_limit=int(chunk_size))
        chunk_files = []
        tts = get_model("Base", model_size)

        # 3. Generate Loop
        for i, chunk in enumerate(text_chunks):
            set_seed(seed)
            wavs, sr = tts.generate_voice_clone(
                text=chunk.strip(),
                language=language,
                ref_audio=audio_tuple,
                ref_text=final_ref_text.strip() if final_ref_text else None,
                x_vector_only_mode=use_xvector_only,
                max_new_tokens=2048,
            )
            # Save immediately
            temp_filename = f"temp_clone_{i}_{os.getpid()}.wav"
            sf.write(temp_filename, wavs[0], sr)
            chunk_files.append(temp_filename)

            # Clear memory
            del wavs
            torch.cuda.empty_cache()
            gc.collect()

        # 4. Stitch & Process
        stitched_file = stitch_chunk_files(chunk_files, tts_filename, gap_seconds=chunk_gap)
        final_audio, srt1, srt2, srt3, srt4 = process_audio_output(stitched_file, make_subs, remove_silence, language)
        bundle_meta = {
            "mode": "VoiceClone",
            "model_used": f"Qwen/Qwen3-TTS-12Hz-{model_size}-Base",
            "language": language,
            "seed": seed,
            "chunk_size": int(chunk_size),
            "chunk_gap": float(chunk_gap),
            "text": target_text.strip(),
            "reference_text": final_ref_text.strip() if final_ref_text else None,
            "x_vector_only_mode": bool(use_xvector_only),
        }
        extra_files = [ref_audio] if isinstance(ref_audio, str) else []
        bundle_path = create_bundle(
            "voice_clone",
            final_audio,
            bundle_meta,
            [srt1, srt2, srt3, srt4],
            extra_paths=extra_files,
        )
        status = f"Success! Mode: {mode} | Seed: {seed}"
        return final_audio, status, srt1, srt2, srt3, srt4, seed, final_audio, bundle_path

    except Exception as e:
        return None, f"Error: {e}", None, None, None, None, seed if 'seed' in locals() else -1, None, None


# --- UI Construction ---

def on_mode_change(mode):
    return gr.update(visible=("High-Quality" in mode))

def build_ui():
    theme = gr.themes.Soft(font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"])
    css = ".gradio-container {max-width: none !important;} .tab-content {padding: 20px;}"

    with gr.Blocks(theme=theme, css=css, title="Qwen3-TTS Demo") as demo:
        gr.HTML("""
        <div style="text-align: center; margin: 20px auto; max-width: 800px;">
            <h1 style="font-size: 2.5em; margin-bottom: 5px;">🎙️ Qwen3-TTS </h1>
            <a href="https://colab.research.google.com/github/shariqriazz/Qwen3-TTS-Colab/blob/main/Qwen3_TTS_Colab.ipynb" target="_blank" style="display: inline-block; padding: 10px 20px; background-color: #4285F4; color: white; border-radius: 6px; text-decoration: none; font-size: 1em;">🥳 Run on Google Colab</a>
        </div>""")

        with gr.Tabs():
            # --- Tab 0: Models ---
            with gr.Tab("Models"):
                with gr.Accordion("📥 Download Models", open=True):
                    gr.Markdown("*Models can be downloaded here or will auto-download when you generate in any tab.*")
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Row():
                                download_model_type = gr.Dropdown(
                                    label="Type",
                                    choices=list(AVAILABLE_MODELS.keys()),
                                    value="CustomVoice",
                                    interactive=True,
                                    scale=2,
                                )
                                download_model_size = gr.Dropdown(
                                    label="Size",
                                    choices=MODEL_SIZES,
                                    value="1.7B",
                                    interactive=True,
                                    scale=1,
                                )
                            download_btn = gr.Button("Download", variant="primary", size="sm")
                            download_status = gr.Textbox(label="Status", lines=1, interactive=False)
                        with gr.Column(scale=2):
                            models_status = gr.Markdown(value=get_downloaded_models_status)

                download_model_type.change(
                    get_available_sizes,
                    inputs=[download_model_type],
                    outputs=[download_model_size],
                )

                download_btn.click(
                    download_model_ui,
                    inputs=[download_model_type, download_model_size],
                    outputs=[download_status, models_status],
                )

                with gr.Accordion("🚀 Load / Unload Models", open=False):
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Row():
                                load_model_type = gr.Dropdown(
                                    label="Type",
                                    choices=list(AVAILABLE_MODELS.keys()),
                                    value="CustomVoice",
                                    interactive=True,
                                    scale=2,
                                )
                                load_model_size = gr.Dropdown(
                                    label="Size",
                                    choices=MODEL_SIZES,
                                    value="1.7B",
                                    interactive=True,
                                    scale=1,
                                )
                            with gr.Row():
                                load_btn = gr.Button("Load to GPU", variant="primary", size="sm")
                                unload_all_btn = gr.Button("Unload All", variant="stop", size="sm")
                            load_status = gr.Textbox(label="Status", lines=1, interactive=False)
                        with gr.Column(scale=2):
                            load_refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
                            load_loaded_status = gr.Markdown(value=get_loaded_models_status)

                load_model_type.change(
                    get_available_sizes,
                    inputs=[load_model_type],
                    outputs=[load_model_size],
                )

                load_refresh_btn.click(
                    lambda: get_loaded_models_status(),
                    inputs=[],
                    outputs=[load_loaded_status],
                )

                load_btn.click(
                    load_model_ui,
                    inputs=[load_model_type, load_model_size],
                    outputs=[load_status, load_loaded_status],
                )

                unload_all_btn.click(
                    unload_all_models_ui,
                    inputs=[],
                    outputs=[load_status, load_loaded_status],
                )
            # --- Tab 1: Voice Design ---
            with gr.Tab("Voice Design"):
                with gr.Row():
                    with gr.Column(scale=2):
                        design_text = gr.Textbox(label="Text to Synthesize", lines=4, value="It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!",
                                                 placeholder="Enter the text you want to convert to speech...")
                        design_language = gr.Dropdown(label="Language", choices=LANGUAGES, value="Auto")
                        design_instruct = gr.Textbox(label="Voice Description", lines=3,  placeholder="Describe the voice characteristics you want...",
                            value="Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice.")
                        design_btn = gr.Button("Generate with Custom Voice", variant="primary")
                        with gr.Accordion("More options", open=False):
                            with gr.Row():
                                design_seed = gr.Number(label="Seed (-1 = Auto)", value=-1, precision=0)
                                design_chunk_size = gr.Slider(label="Chunk Size (chars)", minimum=50, maximum=500, value=280, step=10)
                            with gr.Row():
                                design_chunk_gap = gr.Slider(label="Chunk Gap (s)", minimum=0.0, maximum=3.0, value=0.0, step=0.01)
                                design_base_size = gr.Dropdown(label="Base Model Size (for consistency)", choices=MODEL_SIZES, value="1.7B")
                            with gr.Row():
                                design_consistent = gr.Checkbox(label="Consistent Voice for Long Text (VoiceDesign→Base)", value=True)
                                design_rem_silence = gr.Checkbox(label="Remove Silence", value=False)
                                design_make_subs = gr.Checkbox(label="Generate Subtitles", value=False)
                        
                        

                    with gr.Column(scale=2):
                        design_audio_out = gr.Audio(label="Generated Audio", type="filepath")
                        design_status = gr.Textbox(label="Status", interactive=False)
                        design_audio_file = gr.File(label="Download Audio")
                        design_bundle_file = gr.File(label="Download Bundle (ZIP)")
                        
                        with gr.Accordion("📝 Subtitles", open=False):
                            with gr.Row():
                                d_srt1 = gr.File(label="Original (Whisper)")
                                d_srt2 = gr.File(label="Readable")
                            with gr.Row():
                                d_srt3 = gr.File(label="Word-level")
                                d_srt4 = gr.File(label="Shorts/Reels")

                design_btn.click(
                    generate_voice_design, 
                    inputs=[design_text, design_language, design_instruct, design_rem_silence, design_make_subs, design_seed, design_chunk_size, design_chunk_gap, design_consistent, design_base_size], 
                    outputs=[design_audio_out, design_status, d_srt1, d_srt2, d_srt3, d_srt4, design_seed, design_audio_file, design_bundle_file]
                )

            # --- Tab 2: Voice Clone ---
            with gr.Tab("Voice Clone (Base)"):
                with gr.Row():
                    with gr.Column(scale=2):
                        clone_target_text = gr.Textbox(label="Target Text", lines=3, placeholder="Enter the text you want the cloned voice to speak...")
                        clone_ref_audio = gr.Audio(label="Reference Audio (Upload a voice sample to clone)", type="filepath")
                        
                        with gr.Row():
                            clone_language = gr.Dropdown(label="Language", choices=LANGUAGES, value="Auto",scale=1)
                            clone_model_size = gr.Dropdown(label="Model Size", choices=MODEL_SIZES, value="1.7B",scale=1)
                            clone_mode = gr.Dropdown(
                                label="Mode",
                                choices=["High-Quality (Audio + Transcript)", "Fast (Audio Only)"],
                                value="High-Quality (Audio + Transcript)",
                                interactive=True,
                                scale=2
                            )
                        
                        clone_ref_text = gr.Textbox(label="Reference Text", lines=2, visible=True)
                        clone_btn = gr.Button("Clone & Generate", variant="primary")
                        with gr.Accordion("More options", open=False):
                            with gr.Row():
                                clone_seed = gr.Number(label="Seed (-1 = Auto)", value=-1, precision=0)
                                clone_chunk_size = gr.Slider(label="Chunk Size (chars)", minimum=50, maximum=500, value=280, step=10)
                            with gr.Row():
                                clone_chunk_gap = gr.Slider(label="Chunk Gap (s)", minimum=0.0, maximum=3.0, value=0.0, step=0.01)
                                clone_rem_silence = gr.Checkbox(label="Remove Silence", value=False)
                                clone_make_subs = gr.Checkbox(label="Generate Subtitles", value=False)

                        

                    with gr.Column(scale=2):
                        clone_audio_out = gr.Audio(label="Generated Audio", type="filepath")
                        clone_status = gr.Textbox(label="Status", interactive=False)
                        clone_audio_file = gr.File(label="Download Audio")
                        clone_bundle_file = gr.File(label="Download Bundle (ZIP)")
                        
                        with gr.Accordion("📝 Subtitles", open=False):
                            with gr.Row():
                                c_srt1 = gr.File(label="Original")
                                c_srt2 = gr.File(label="Readable")
                            with gr.Row():
                                c_srt3 = gr.File(label="Word-level")
                                c_srt4 = gr.File(label="Shorts/Reels")

                clone_mode.change(on_mode_change, inputs=[clone_mode], outputs=[clone_ref_text])
                clone_ref_audio.change(transcribe_reference, inputs=[clone_ref_audio, clone_mode, clone_language], outputs=[clone_ref_text])
                
                clone_btn.click(
                    smart_generate_clone,
                    inputs=[clone_ref_audio, clone_ref_text, clone_target_text, clone_language, clone_mode, clone_model_size, clone_rem_silence, clone_make_subs, clone_seed, clone_chunk_size, clone_chunk_gap],
                    outputs=[clone_audio_out, clone_status, c_srt1, c_srt2, c_srt3, c_srt4, clone_seed, clone_audio_file, clone_bundle_file]
                )

            # --- Tab 3: TTS (CustomVoice) ---
            with gr.Tab("TTS (CustomVoice)"):
                with gr.Row():
                    with gr.Column(scale=2):
                        tts_text = gr.Textbox(label="Text", lines=4,   placeholder="Enter the text you want to convert to speech...",
                            value="Hello! Welcome to Text-to-Speech system. This is a demo of our TTS capabilities.")
                        with gr.Row():
                            tts_language = gr.Dropdown(label="Language", choices=LANGUAGES, value="English")
                            tts_speaker = gr.Dropdown(label="Speaker", choices=SPEAKERS, value="Ryan")
                        with gr.Row():
                            tts_instruct = gr.Textbox(label="Style Instruction (Optional)", lines=2,placeholder="e.g., Speak in a cheerful and energetic tone")
                            tts_model_size = gr.Dropdown(label="Size", choices=MODEL_SIZES, value="1.7B")
                        tts_btn = gr.Button("Generate Speech", variant="primary")
                        with gr.Accordion("More options", open=False):
                            with gr.Row():
                                tts_seed = gr.Number(label="Seed (-1 = Auto)", value=-1, precision=0)
                                tts_chunk_size = gr.Slider(label="Chunk Size (chars)", minimum=50, maximum=500, value=280, step=10)
                            with gr.Row():
                                tts_chunk_gap = gr.Slider(label="Chunk Gap (s)", minimum=0.0, maximum=3.0, value=0.0, step=0.01)
                                tts_rem_silence = gr.Checkbox(label="Remove Silence", value=False)
                                tts_make_subs = gr.Checkbox(label="Generate Subtitles", value=False)
                            
                        

                    with gr.Column(scale=2):
                        tts_audio_out = gr.Audio(label="Generated Audio", type="filepath")
                        tts_status = gr.Textbox(label="Status", interactive=False)
                        tts_audio_file = gr.File(label="Download Audio")
                        tts_bundle_file = gr.File(label="Download Bundle (ZIP)")
                        
                        with gr.Accordion("📝 Subtitles", open=False):
                            with gr.Row():
                                t_srt1 = gr.File(label="Original")
                                t_srt2 = gr.File(label="Readable")
                            with gr.Row():
                                t_srt3 = gr.File(label="Word-level")
                                t_srt4 = gr.File(label="Shorts/Reels")

                tts_btn.click(
                    generate_custom_voice, 
                    inputs=[tts_text, tts_language, tts_speaker, tts_instruct, tts_model_size, tts_rem_silence, tts_make_subs, tts_seed, tts_chunk_size, tts_chunk_gap], 
                    outputs=[tts_audio_out, tts_status, t_srt1, t_srt2, t_srt3, t_srt4, tts_seed, tts_audio_file, tts_bundle_file]
                )
            # --- Tab 4: About ---
            with gr.Tab("About"):
                gr.Markdown("""
                # Qwen3-TTS 
                A unified Text-to-Speech demo featuring three powerful modes:
                - **Voice Design**: Create custom voices using natural language descriptions
                - **Voice Clone (Base)**: Clone any voice from a reference audio
                - **TTS (CustomVoice)**: Generate speech with predefined speakers and optional style instructions

                Built with [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Qwen Team.
                """)

                gr.HTML("""
                <hr>
                <p style="color: red; font-weight: bold; font-size: 16px;">
                ⚠️ NOTE
                </p>
                <p>
                This Gradio UI is not affiliated with the official Qwen3-TTS project and is based on the
                official Qwen3-TTS demo UI:<br>
                <a href="https://huggingface.co/spaces/Qwen/Qwen3-TTS" target="_blank">
                https://huggingface.co/spaces/Qwen/Qwen3-TTS
                </a>
                </p>

                <p><b>Additional features:</b></p>
                <ul>
                  <li>Automatic transcription support using faster-whisper-large-v3-turbo-ct2</li>
                  <li>Long text input support</li>
                  <li>Because we are using Whisper, subtitles are also added</li>
                </ul>
                """)

             
    return demo

# if __name__ == "__main__":
#     demo = build_ui()
#     demo.launch(share=True, debug=True)



import click
@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
def main(share,debug):
    demo = build_ui()
    # demo.launch(share=True, debug=True)
    demo.queue().launch(share=share,debug=debug)

if __name__ == "__main__":
    main()    
