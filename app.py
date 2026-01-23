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
from pydub import AudioSegment
from pydub.silence import split_on_silence
from huggingface_hub import snapshot_download
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
MODEL_SIZES = ["0.6B", "1.7B"]

# Speaker and language choices
SPEAKERS = [
    "Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"
]
LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Russian"]

# --- Helper Functions ---

def get_model_path(model_type: str, model_size: str) -> str:
    """Get model path based on type and size."""
    try:
      return snapshot_download(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}")
    except Exception as e:
      return download_model(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}", download_folder="./qwen_tts_model", redownload= False)

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

def stitch_chunk_files(chunk_files,output_filename):
    """
    Takes a list of file paths.
    Stitches them into one file.
    Deletes the temporary chunk files.
    """
    if not chunk_files:
        return None

    combined_audio = AudioSegment.empty()
    
    print(f"Stitching {len(chunk_files)} audio files...")
    for f in chunk_files:
        try:
            segment = AudioSegment.from_wav(f)
            combined_audio += segment
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

# --- Generators (Memory Optimized) ---

def generate_voice_design(text, language, voice_description, remove_silence, make_subs):
    if not text or not text.strip(): return None, "Error: Text is required.", None, None, None, None
    
    try:
        # 1. Chunk Text
        text_chunks, tts_filename = text_chunk(text, language, char_limit=280)
        print(f"Processing {len(text_chunks)} chunks...")
        
        chunk_files = []
        tts = get_model("VoiceDesign", "1.7B")

        # 2. Generate & Save Loop
        for i, chunk in enumerate(text_chunks):
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
        stitched_file = stitch_chunk_files(chunk_files,tts_filename)
        
        # 4. Post-Process
        final_audio, srt1, srt2, srt3, srt4 = process_audio_output(stitched_file, make_subs, remove_silence, language)
        
        return final_audio, "Generation Success!", srt1, srt2, srt3, srt4

    except Exception as e:
        return None, f"Error: {e}", None, None, None, None

def generate_custom_voice(text, language, speaker, instruct, model_size, remove_silence, make_subs):
    if not text or not text.strip(): return None, "Error: Text is required.", None, None, None, None
    
    try:
        text_chunks, tts_filename = text_chunk(text, language, char_limit=280)
        chunk_files = []
        tts = get_model("CustomVoice", model_size)
        formatted_speaker = speaker.lower().replace(" ", "_")

        for i, chunk in enumerate(text_chunks):
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
            
        stitched_file = stitch_chunk_files(chunk_files,tts_filename)
        final_audio, srt1, srt2, srt3, srt4 = process_audio_output(stitched_file, make_subs, remove_silence, language)
        return final_audio, "Generation Success!", srt1, srt2, srt3, srt4

    except Exception as e:
        return None, f"Error: {e}", None, None, None, None

def smart_generate_clone(ref_audio, ref_text, target_text, language, mode, model_size, remove_silence, make_subs):
    if not target_text or not target_text.strip(): return None, "Error: Target text is required.", None, None, None, None
    if not ref_audio: return None, "Error: Ref audio required.", None, None, None, None

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
                     return None, f"Transcription failed: {final_ref_text}", None, None, None, None
            except Exception as e:
                return None, f"Transcribe Error: {e}", None, None, None, None
    else:
        final_ref_text = None

    try:
        # 2. Chunk Target Text
        text_chunks, tts_filename = text_chunk(target_text, language, char_limit=280)
        chunk_files = []
        tts = get_model("Base", model_size)

        # 3. Generate Loop
        for i, chunk in enumerate(text_chunks):
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
        stitched_file = stitch_chunk_files(chunk_files,tts_filename)
        final_audio, srt1, srt2, srt3, srt4 = process_audio_output(stitched_file, make_subs, remove_silence, language)
        return final_audio, f"Success! Mode: {mode}", srt1, srt2, srt3, srt4

    except Exception as e:
        return None, f"Error: {e}", None, None, None, None


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
            <a href="https://colab.research.google.com/github/NeuralFalconYT/Qwen3-TTS-Colab/blob/main/Qwen3_TTS_Colab.ipynb" target="_blank" style="display: inline-block; padding: 10px 20px; background-color: #4285F4; color: white; border-radius: 6px; text-decoration: none; font-size: 1em;">🥳 Run on Google Colab</a>
        </div>""")

        with gr.Tabs():
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
                              design_rem_silence = gr.Checkbox(label="Remove Silence", value=False)
                              design_make_subs = gr.Checkbox(label="Generate Subtitles", value=False)
                        
                        

                    with gr.Column(scale=2):
                        design_audio_out = gr.Audio(label="Generated Audio", type="filepath")
                        design_status = gr.Textbox(label="Status", interactive=False)
                        
                        with gr.Accordion("📝 Subtitles", open=False):
                            with gr.Row():
                                d_srt1 = gr.File(label="Original (Whisper)")
                                d_srt2 = gr.File(label="Readable")
                            with gr.Row():
                                d_srt3 = gr.File(label="Word-level")
                                d_srt4 = gr.File(label="Shorts/Reels")

                design_btn.click(
                    generate_voice_design, 
                    inputs=[design_text, design_language, design_instruct, design_rem_silence, design_make_subs], 
                    outputs=[design_audio_out, design_status, d_srt1, d_srt2, d_srt3, d_srt4]
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
                              clone_rem_silence = gr.Checkbox(label="Remove Silence", value=False)
                              clone_make_subs = gr.Checkbox(label="Generate Subtitles", value=False)

                        

                    with gr.Column(scale=2):
                        clone_audio_out = gr.Audio(label="Generated Audio", type="filepath")
                        clone_status = gr.Textbox(label="Status", interactive=False)
                        
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
                    inputs=[clone_ref_audio, clone_ref_text, clone_target_text, clone_language, clone_mode, clone_model_size, clone_rem_silence, clone_make_subs],
                    outputs=[clone_audio_out, clone_status, c_srt1, c_srt2, c_srt3, c_srt4]
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
                              tts_rem_silence = gr.Checkbox(label="Remove Silence", value=False)
                              tts_make_subs = gr.Checkbox(label="Generate Subtitles", value=False)
                            
                        

                    with gr.Column(scale=2):
                        tts_audio_out = gr.Audio(label="Generated Audio", type="filepath")
                        tts_status = gr.Textbox(label="Status", interactive=False)
                        
                        with gr.Accordion("📝 Subtitles", open=False):
                            with gr.Row():
                                t_srt1 = gr.File(label="Original")
                                t_srt2 = gr.File(label="Readable")
                            with gr.Row():
                                t_srt3 = gr.File(label="Word-level")
                                t_srt4 = gr.File(label="Shorts/Reels")

                tts_btn.click(
                    generate_custom_voice, 
                    inputs=[tts_text, tts_language, tts_speaker, tts_instruct, tts_model_size, tts_rem_silence, tts_make_subs], 
                    outputs=[tts_audio_out, tts_status, t_srt1, t_srt2, t_srt3, t_srt4]
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
    demo.launch(share=True, debug=True)
    demo.queue().launch(debug=debug, share=share)

if __name__ == "__main__":
    main()    
