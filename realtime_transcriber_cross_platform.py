#!/usr/bin/env python3
"""
Whisper Real-Time Transcriber
Cross-platform: Windows and Linux (Ubuntu)

Usage:
    python realtime_transcriber_cross_platform.py
"""

import threading
import queue
import numpy as np
import sounddevice as sd
import time
import re
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import os
import sys
import json
import requests
from collections import deque

# Platform detection
IS_WINDOWS = sys.platform == 'win32'
IS_LINUX = sys.platform.startswith('linux')

# Platform-specific imports
if IS_WINDOWS:
    import keyboard
    import pyperclip
    try:
        import pystray
        from PIL import Image, ImageDraw
        HAS_TRAY = True
    except ImportError:
        HAS_TRAY = False
else:
    # Linux alternatives
    HAS_TRAY = False
    try:
        from pynput import keyboard as pynput_keyboard
        HAS_PYNPUT = True
    except ImportError:
        HAS_PYNPUT = False
    try:
        import pyperclip
        HAS_PYPERCLIP = True
    except ImportError:
        HAS_PYPERCLIP = False


def get_device_info():
    """Detect available compute devices."""
    devices = {"cpu": {"available": True, "name": "CPU", "compute_type": "int8"}}
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            cuda_version = torch.version.cuda
            print(f"CUDA version: {cuda_version}")
            print(f"GPU: {gpu_name}")
            print(f"VRAM: {vram:.1f}GB")
            devices["cuda"] = {
                "available": True,
                "name": f"{gpu_name} ({vram:.1f}GB)",
                "compute_type": "float16"
            }
        else:
            print("GPU not detected. Reasons could be:")
            print("  - PyTorch CPU-only version installed")
            print("  - CUDA not installed or not in PATH")
            print("  - No NVIDIA GPU in system")
            print("  - GPU drivers not installed")
            if hasattr(torch.version, 'cuda') and torch.version.cuda is None:
                print("  -> PyTorch was compiled without CUDA support")
                print("  -> To fix: pip uninstall torch")
                print("  -> Then: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    except Exception as e:
        print(f"Error detecting GPU: {e}")
    
    return devices


class HotkeyManager:
    """Cross-platform hotkey management."""
    
    def __init__(self, callback):
        self.callback = callback
        self.hotkey = "ctrl+f9"
        self.registered = False
        self.listener = None
    
    def register(self):
        if IS_WINDOWS:
            try:
                import keyboard
                keyboard.add_hotkey(self.hotkey, self.callback, suppress=True)
                self.registered = True
                print(f"Hotkey registered: {self.hotkey.upper()}")
            except Exception as e:
                print(f"Hotkey failed (try running as admin): {e}")
        elif IS_LINUX and HAS_PYNPUT:
            try:
                from pynput import keyboard as pynput_keyboard
                
                # Track pressed keys
                self.pressed_keys = set()
                
                def on_press(key):
                    try:
                        if key == pynput_keyboard.Key.ctrl_l or key == pynput_keyboard.Key.ctrl_r:
                            self.pressed_keys.add('ctrl')
                        elif hasattr(key, 'vk') and key.vk == 120:  # F9
                            self.pressed_keys.add('f9')
                        elif hasattr(key, 'name') and key.name == 'f9':
                            self.pressed_keys.add('f9')
                        
                        if 'ctrl' in self.pressed_keys and 'f9' in self.pressed_keys:
                            self.callback()
                            self.pressed_keys.clear()
                    except:
                        pass
                
                def on_release(key):
                    try:
                        if key == pynput_keyboard.Key.ctrl_l or key == pynput_keyboard.Key.ctrl_r:
                            self.pressed_keys.discard('ctrl')
                        elif hasattr(key, 'vk') and key.vk == 120:
                            self.pressed_keys.discard('f9')
                        elif hasattr(key, 'name') and key.name == 'f9':
                            self.pressed_keys.discard('f9')
                    except:
                        pass
                
                self.listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
                self.listener.start()
                self.registered = True
                print(f"Hotkey registered: Ctrl+F9")
            except Exception as e:
                print(f"Hotkey failed: {e}")
        else:
            print("Hotkey not available on this platform")
    
    def unregister(self):
        if IS_WINDOWS and self.registered:
            try:
                import keyboard
                keyboard.unhook_all_hotkeys()
            except:
                pass
        elif self.listener:
            try:
                self.listener.stop()
            except:
                pass


class ClipboardManager:
    """Cross-platform clipboard operations."""
    
    def __init__(self):
        self.lock = threading.Lock()
    
    def type_text(self, text):
        """Type text at cursor position using clipboard paste."""
        if not text:
            return
        
        with self.lock:
            try:
                if IS_WINDOWS:
                    import keyboard
                    import pyperclip
                    
                    try:
                        original = pyperclip.paste()
                    except:
                        original = ""
                    
                    pyperclip.copy(text + " ")
                    time.sleep(0.02)
                    keyboard.send('ctrl+v')
                    time.sleep(0.05)
                    
                    if original:
                        pyperclip.copy(original)
                
                elif IS_LINUX:
                    if HAS_PYPERCLIP and HAS_PYNPUT:
                        from pynput.keyboard import Controller, Key
                        
                        try:
                            original = pyperclip.paste()
                        except:
                            original = ""
                        
                        pyperclip.copy(text + " ")
                        time.sleep(0.02)
                        
                        kb = Controller()
                        kb.press(Key.ctrl)
                        kb.press('v')
                        kb.release('v')
                        kb.release(Key.ctrl)
                        time.sleep(0.05)
                        
                        if original:
                            pyperclip.copy(original)
                    else:
                        print(f"[Would type]: {text}")
                        
            except Exception as e:
                print(f"Error typing text: {e}")


class OllamaClient:
    """Client for Ollama LLM API."""
    
    def __init__(self, server_url="http://192.168.50.134:11434", model="llama3.2:3b"):
        self.server_url = server_url.rstrip('/')
        self.model = model
        self.timeout = 10.0
    
    def test_connection(self):
        """Test if Ollama server is accessible."""
        try:
            response = requests.get(f"{self.server_url}/api/tags", timeout=2.0)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self):
        """Get list of available models from Ollama server."""
        try:
            response = requests.get(f"{self.server_url}/api/tags", timeout=2.0)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []
    
    def correct_text(self, chunks, pause_duration=0.0):
        """Send chunks to LLM for correction."""
        try:
            # Build prompt
            chunks_text = "\n".join(f"{i+1}: {chunk}" for i, chunk in enumerate(chunks))
            
            prompt = f"""Fix any sentence fragments or punctuation errors in these transcribed text segments.
Preserve the original wording as much as possible.
Only merge fragments if they clearly belong together.
If pause was long (>5 seconds), treat as separate paragraphs.

Segments (in order):
{chunks_text}

Pause after last segment: {pause_duration:.1f} seconds

Output only the corrected text with no explanations or comments:"""
            
            # Call Ollama API
            response = requests.post(
                f"{self.server_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                corrected = result.get('response', '').strip()
                return corrected if corrected else " ".join(chunks)
            else:
                print(f"LLM API error: {response.status_code}")
                return " ".join(chunks)
                
        except Exception as e:
            print(f"LLM correction error: {e}")
            # Fallback: just join the chunks
            return " ".join(chunks)


class TranscriptionBuffer:
    """Buffer for holding transcription chunks before LLM processing."""
    
    def __init__(self, max_size=5, pause_threshold=3.0):
        self.chunks = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        self.pause_threshold = pause_threshold
        self.last_output_time = None
    
    def add_chunk(self, text, timestamp=None):
        """Add a chunk to the buffer."""
        if timestamp is None:
            timestamp = time.time()
        self.chunks.append(text)
        self.timestamps.append(timestamp)
    
    def should_process(self):
        """Check if buffer should be processed (user paused)."""
        if len(self.chunks) == 0:
            return False
        
        time_since_last = time.time() - self.timestamps[-1]
        return time_since_last >= self.pause_threshold
    
    def get_chunks(self):
        """Get all chunks in buffer."""
        return list(self.chunks)
    
    def get_pause_duration(self):
        """Get duration since last chunk."""
        if len(self.timestamps) == 0:
            return 0.0
        return time.time() - self.timestamps[-1]
    
    def clear(self):
        """Clear the buffer."""
        self.chunks.clear()
        self.timestamps.clear()
    
    def size(self):
        """Get number of chunks in buffer."""
        return len(self.chunks)
    
    def is_empty(self):
        """Check if buffer is empty."""
        return len(self.chunks) == 0


class TranscriberGUI:
    
    HALLUCINATIONS = [
        "thank you", "thanks for watching", "thanks for listening",
        "subscribe", "like and subscribe", "see you next time",
        "bye", "goodbye", "the end", "music", "applause", "laughter",
        "silence", "...", "you", "i'm sorry", "sorry", "okay", "ok",
        "um", "uh", "hmm", "mhm",
        "_", "__", "___",  # Filter underscore patterns
    ]
    
    MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
    
    # Medical vocabulary file for context injection
    MEDICAL_VOCAB_FILE = "medical_vocabulary.txt"
    
    def __init__(self):
        # Platform info
        print(f"Platform: {'Windows' if IS_WINDOWS else 'Linux'}")
        
        # Detect devices
        self.available_devices = get_device_info()
        
        if "cuda" in self.available_devices:
            self.device = "cuda"
            self.compute_type = "float16"
        else:
            self.device = "cpu"
            self.compute_type = "int8"
        
        print(f"Available devices: {list(self.available_devices.keys())}")
        print(f"Selected device: {self.device.upper()}")
        
        # Audio settings
        self.sample_rate = 16000
        self.microphone_gain = 2.5  # Default gain multiplier (1.0 = no gain, 2.5 = 2.5x amplification)
        
        # Detection settings
        self.silence_threshold = 0.015
        self.silence_duration = 1.2
        self.min_audio_duration = 0.8
        self.max_audio_duration = 15.0
        self.min_speech_ratio = 0.15
        
        # State
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.model = None
        self.current_model_size = None
        
        # Transcription tracking for hang detection
        self.transcription_in_progress = False
        self.last_transcription_start = None
        self.transcription_timeout = 30.0  # seconds
        self.active_transcription_threads = 0
        
        # Audio buffer
        self.audio_lock = threading.Lock()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.silence_samples = 0
        self.speech_samples = 0
        
        # Speaking state
        self.is_speaking = False
        
        # Context
        self.previous_transcription = ""
        self.all_transcriptions = []
        self.last_output_time = None  # Track last output for single-word filtering
        
        # Medical vocabulary for context injection
        self.use_medical_vocab = False
        self.medical_vocabulary = self._load_medical_vocabulary()
        
        # LLM correction system
        self.use_llm_correction = False
        self.llm_server = "http://192.168.50.134:11434"
        self.llm_model = "llama3.2:3b"
        self.llm_client = None
        self.transcription_buffer = TranscriptionBuffer(max_size=5, pause_threshold=3.0)
        self.llm_processing = False
        self.last_chunk_time = time.time()
        
        # Platform-specific managers
        self.clipboard = ClipboardManager()
        self.hotkey_manager = None
        
        # Tray icon (Windows only)
        self.tray_icon = None
        
        # Model path
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(self.model_path, exist_ok=True)
        
        self._create_gui()
        self._setup_platform_features()
        
        # Default model
        default_model = "tiny" if self.device == "cpu" else "base"
        self.model_var.set(default_model)
        self.root.after(100, lambda: self._load_model(default_model))
        
        print(f"Medical vocabulary loaded: {len(self.medical_vocabulary)} terms")

    def _setup_platform_features(self):
        """Set up platform-specific features."""
        
        # Hotkey for recording
        self.hotkey_manager = HotkeyManager(lambda: self.root.after(0, self._toggle_recording))
        self.hotkey_manager.register()
        
        # Hotkey for buffer flush (Ctrl+F10)
        if IS_WINDOWS:
            try:
                import keyboard
                keyboard.add_hotkey('ctrl+f10', lambda: self.root.after(0, self._manual_flush_buffer), suppress=True)
            except:
                pass
        elif HAS_PYNPUT:
            # Pynput hotkey for F10 is more complex, skip for now
            pass
        
        # Tray icon (Windows only)
        if IS_WINDOWS and HAS_TRAY:
            self._create_tray()
    
    def _toggle_llm_correction(self):
        """Toggle LLM correction on/off."""
        self.use_llm_correction = self.llm_enabled_var.get()
        
        if self.use_llm_correction:
            # Initialize LLM client
            server = self.llm_server_var.get()
            model = self.llm_model_var.get()
            self.llm_client = OllamaClient(server_url=server, model=model)
            
            # Test connection
            if self.llm_client.test_connection():
                self.llm_status_label.config(text="✓ Connected", fg="green")
                print(f"LLM correction enabled: {server} - {model}")
            else:
                self.llm_status_label.config(text="⚠ Connection failed", fg="red")
                messagebox.showwarning("LLM Connection", 
                                      f"Could not connect to Ollama server at {server}\n"
                                      "Transcription will continue without LLM correction.")
        else:
            self.llm_client = None
            self.llm_status_label.config(text="")
            print("LLM correction disabled")
    
    def _test_llm_connection(self):
        """Test connection to LLM server."""
        server = self.llm_server_var.get()
        test_client = OllamaClient(server_url=server)
        
        if test_client.test_connection():
            models = test_client.get_available_models()
            messagebox.showinfo("LLM Connection Test", 
                               f"✓ Connected to {server}\n\n"
                               f"Available models: {len(models)}\n"
                               f"{', '.join(models[:5])}"
                               f"{'...' if len(models) > 5 else ''}")
            self.llm_status_label.config(text="✓ Connected", fg="green")
        else:
            messagebox.showerror("LLM Connection Test", 
                                f"✗ Could not connect to {server}\n\n"
                                "Please check:\n"
                                "1. Ollama is running\n"
                                "2. Server address is correct\n"
                                "3. Network connectivity")
            self.llm_status_label.config(text="⚠ Not connected", fg="red")
    
    def _refresh_llm_models(self):
        """Refresh available LLM models."""
        server = self.llm_server_var.get()
        client = OllamaClient(server_url=server)
        
        models = client.get_available_models()
        if models:
            self.llm_model_combo['values'] = models
            messagebox.showinfo("Models Refreshed", 
                               f"Found {len(models)} models:\n\n" + "\n".join(models))
        else:
            messagebox.showwarning("Models Refresh", 
                                  "Could not retrieve models from server.\n"
                                  "Make sure Ollama is running.")
    
    def _manual_flush_buffer(self):
        """Manually flush the transcription buffer (force immediate processing)."""
        if not self.use_llm_correction or not self.llm_client:
            return
        
        if self.transcription_buffer.is_empty():
            self.llm_status_label.config(text="Buffer empty", fg="gray")
            return
        
        # Process buffer immediately
        self.llm_status_label.config(text="🔄 Flushing buffer...", fg="orange")
        threading.Thread(target=self._process_buffer_with_llm, daemon=True).start()
    
    def _process_buffer_with_llm(self):
        """Process transcription buffer with LLM correction."""
        if self.llm_processing or self.transcription_buffer.is_empty():
            return
        
        self.llm_processing = True
        
        try:
            chunks = self.transcription_buffer.get_chunks()
            pause_duration = self.transcription_buffer.get_pause_duration()
            
            self.root.after(0, lambda: self.llm_status_label.config(
                text=f"🔄 Correcting {len(chunks)} chunks...", fg="orange"))
            
            # Call LLM for correction
            corrected_text = self.llm_client.correct_text(chunks, pause_duration)
            
            # Output corrected text
            if corrected_text:
                self.text_queue.put(corrected_text)
                print(f"[LLM-CORRECTED] {corrected_text}")
                self.last_output_time = time.time()
            
            # Clear buffer
            self.transcription_buffer.clear()
            
            self.root.after(0, lambda: self.llm_status_label.config(text="✓ Corrected", fg="green"))
            self.root.after(2000, lambda: self.llm_status_label.config(text="", fg="gray"))
            
        except Exception as e:
            print(f"LLM processing error: {e}")
            # Fallback: output uncorrected chunks
            chunks = self.transcription_buffer.get_chunks()
            for chunk in chunks:
                self.text_queue.put(chunk)
            self.transcription_buffer.clear()
            self.root.after(0, lambda: self.llm_status_label.config(text="⚠ Error", fg="red"))
        finally:
            self.llm_processing = False
    
    
    def _load_medical_vocabulary(self):
        """Load medical vocabulary from file."""
        vocab_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.MEDICAL_VOCAB_FILE)
        vocabulary = set()
        
        if os.path.exists(vocab_file):
            try:
                with open(vocab_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            vocabulary.add(line.lower())
                print(f"Loaded {len(vocabulary)} medical terms from {self.MEDICAL_VOCAB_FILE}")
            except Exception as e:
                print(f"Warning: Could not load medical vocabulary: {e}")
        else:
            print(f"Warning: {self.MEDICAL_VOCAB_FILE} not found. Creating default file.")
            self._create_default_medical_vocabulary(vocab_file)
            vocabulary = self._load_medical_vocabulary()
        
        return vocabulary
    
    def _create_default_medical_vocabulary(self, filepath):
        """Create a default medical vocabulary file."""
        default_vocab = [
            "# Medical Vocabulary File",
            "# Add medical terms one per line (case-insensitive)",
            "# Lines starting with # are comments",
            "",
            "# Common Medical Terms",
            "acetaminophen",
            "antibiotic",
            "cardiovascular",
            "diabetes",
            "hypertension",
            "prescription",
            "diagnosis",
            "prognosis",
            "symptomatic",
            "asymptomatic",
        ]
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(default_vocab))
            print(f"Created default {self.MEDICAL_VOCAB_FILE}")
        except Exception as e:
            print(f"Error creating default vocabulary file: {e}")

    def _create_tray(self):
        """Create system tray icon (Windows only)."""
        def create_image(recording=False):
            size = 64
            image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            color = 'red' if recording else 'gray'
            draw.ellipse([4, 4, size-4, size-4], fill=color)
            draw.ellipse([20, 20, size-20, size-20], fill='darkred' if recording else 'darkgray')
            return image
        
        self.create_tray_image = create_image
        
        menu = pystray.Menu(
            pystray.MenuItem("Show", self._show_window),
            pystray.MenuItem("Toggle (Ctrl+F9)", lambda: self.root.after(0, self._toggle_recording)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._quit)
        )
        
        self.tray_icon = pystray.Icon("WhisperTranscriber", create_image(False), "Whisper Transcriber", menu)
        threading.Thread(target=self.tray_icon.run, daemon=True).start()

    def _update_tray(self, recording=False):
        """Update tray icon state."""
        if self.tray_icon and hasattr(self, 'create_tray_image'):
            self.tray_icon.icon = self.create_tray_image(recording)
            self.tray_icon.title = "Whisper - RECORDING" if recording else "Whisper Transcriber"

    def _load_model(self, model_size):
        if self.current_model_size == model_size and self.model is not None:
            return
        
        if self.device == "cpu" and model_size in ["medium", "large-v2", "large-v3"]:
            if not messagebox.askyesno("Warning", 
                f"{model_size} model may be very slow on CPU.\nContinue?"):
                self.model_var.set(self.current_model_size or "tiny")
                return
        
        self.status_label.config(text=f"Loading {model_size}...")
        self.root.update()
        
        try:
            from faster_whisper import WhisperModel
            import torch
            
            if self.model is not None:
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            self.model = WhisperModel(
                model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.model_path
            )
            self.current_model_size = model_size
            print(f"Model {model_size} loaded")
            self.status_label.config(text="Ready")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.status_label.config(text="Error")
            messagebox.showerror("Error", f"Failed to load model:\n{e}")

    def _switch_device(self, new_device):
        if new_device == self.device:
            return
        if new_device not in self.available_devices:
            messagebox.showerror("Error", f"Device {new_device} not available")
            self.device_var.set(self.device)
            return
        
        if self.is_recording:
            self._stop_recording()
        
        self.device = new_device
        self.compute_type = self.available_devices[new_device]["compute_type"]
        
        if self.current_model_size:
            self._load_model(self.current_model_size)

    def _is_hallucination(self, text):
        if not text:
            return True
        text_lower = text.lower().strip()
        text_clean = re.sub(r'[^\w\s]', '', text_lower).strip()
        
        # Check for underscore patterns (common transcription glitch)
        if '_' in text or text.count('_') > 5:
            return True
        
        # Check if text is mostly underscores
        if len(text) > 0 and text.count('_') / len(text) > 0.3:
            return True
        
        for h in self.HALLUCINATIONS:
            if text_clean == h:
                return True
        
        words = text_clean.split()
        
        # Fix for single-word corrections: Allow single words if recent output exists
        # This enables users to correct typos by highlighting and re-dictating
        if len(words) <= 1 and len(text_clean) < 10:
            # Check if we recently output text (within last 10 seconds)
            if hasattr(self, 'last_output_time') and self.last_output_time:
                time_since_output = time.time() - self.last_output_time
                if time_since_output < 10.0:
                    # Recent output exists, likely a correction - allow it
                    return False
            # Check if word is in medical vocabulary
            if self.medical_vocabulary and text_clean in self.medical_vocabulary:
                return False
            # Otherwise, likely a hallucination
            return True
            
        if len(words) >= 2 and len(set(words)) == 1:
            return True
        return False

    def _clean_transcription(self, text):
        if not text:
            return ""
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\s+[.,:;]+\s+', ' ', text)
        text = re.sub(r'^[\s.,:;]+', '', text)
        for pattern in [r'\[.*?\]', r'\(.*?\)', r'♪.*?♪', r'♪']:
            text = re.sub(pattern, '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _calculate_speech_ratio(self, audio):
        frame_size = int(self.sample_rate * 0.02)
        num_frames = len(audio) // frame_size
        if num_frames == 0:
            return 0
        speech_frames = sum(1 for i in range(num_frames) 
                          if np.sqrt(np.mean(audio[i*frame_size:(i+1)*frame_size]**2)) >= self.silence_threshold)
        return speech_frames / num_frames

    def _create_gui(self):
        self.root = tk.Tk()
        self.root.title("Whisper Transcriber")
        self.root.geometry("800x600")
        
        # Control frame
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        hotkey_text = "Ctrl+F9" if (IS_WINDOWS or HAS_PYNPUT) else ""
        btn_text = f"Start ({hotkey_text})" if hotkey_text else "Start"
        
        self.record_btn = tk.Button(
            control_frame, text=btn_text,
            command=self._toggle_recording,
            bg="green", fg="white", width=18,
            font=("Helvetica", 10, "bold")
        )
        self.record_btn.pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="Clear", command=self._clear_text, width=8).pack(side=tk.LEFT, padx=2)
        tk.Button(control_frame, text="Copy", command=self._copy_text, width=8).pack(side=tk.LEFT, padx=2)
        
        # Medical Vocabulary checkbox
        self.medical_var = tk.BooleanVar(value=False)
        self.medical_check = tk.Checkbutton(
            control_frame, 
            text="Use Medical Vocabulary",
            variable=self.medical_var,
            command=lambda: setattr(self, 'use_medical_vocab', self.medical_var.get()),
            font=("Helvetica", 9)
        )
        self.medical_check.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(control_frame, text="Initializing...", font=("Helvetica", 10))
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        # Device and Model frame
        device_frame = tk.Frame(self.root)
        device_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(device_frame, text="Device:", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        
        self.device_var = tk.StringVar(value=self.device)
        
        for dev in ["cpu", "cuda"]:
            state = tk.NORMAL if dev in self.available_devices else tk.DISABLED
            dev_info = self.available_devices.get(dev, {})
            dev_name = dev_info.get("name", dev.upper())
            
            rb = tk.Radiobutton(
                device_frame,
                text=dev_name if dev == "cpu" else f"GPU: {dev_name}",
                variable=self.device_var, value=dev,
                command=lambda d=dev: self._switch_device(d),
                state=state
            )
            rb.pack(side=tk.LEFT, padx=5)
        
        tk.Label(device_frame, text="  |  ", fg="gray").pack(side=tk.LEFT)
        tk.Label(device_frame, text="Model:", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        
        self.model_var = tk.StringVar(value="tiny")
        
        for m in ["tiny", "base", "small", "medium", "large-v3"]:
            tk.Radiobutton(
                device_frame, text=m,
                variable=self.model_var, value=m,
                command=lambda: self._load_model(self.model_var.get())
            ).pack(side=tk.LEFT, padx=3)
        
        # Output frame
        output_frame = tk.Frame(self.root)
        output_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(output_frame, text="Output:", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)
        
        self.output_var = tk.StringVar(value="both")
        for text, value in [("Window", "window"), ("Cursor", "cursor"), ("Both", "both")]:
            tk.Radiobutton(output_frame, text=text, variable=self.output_var, value=value).pack(side=tk.LEFT, padx=5)
        
        if hotkey_text:
            tk.Label(output_frame, text=f"  |  Hotkey: {hotkey_text}", fg="blue", 
                    font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=10)
        
        # Settings frame
        settings_frame = tk.Frame(self.root)
        settings_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(settings_frame, text="Silence:").pack(side=tk.LEFT)
        self.threshold_var = tk.DoubleVar(value=0.015)
        tk.Scale(settings_frame, from_=0.005, to=0.05, resolution=0.005, orient=tk.HORIZONTAL,
                variable=self.threshold_var, length=70, 
                command=lambda v: setattr(self, 'silence_threshold', float(v))).pack(side=tk.LEFT)
        
        tk.Label(settings_frame, text="Pause:").pack(side=tk.LEFT, padx=(5, 0))
        self.pause_var = tk.DoubleVar(value=1.2)
        tk.Scale(settings_frame, from_=0.5, to=3.0, resolution=0.1, orient=tk.HORIZONTAL,
                variable=self.pause_var, length=70,
                command=lambda v: setattr(self, 'silence_duration', float(v))).pack(side=tk.LEFT)
        
        tk.Label(settings_frame, text="Max:").pack(side=tk.LEFT, padx=(5, 0))
        self.max_chunk_var = tk.DoubleVar(value=15.0)
        tk.Scale(settings_frame, from_=10.0, to=30.0, resolution=1.0, orient=tk.HORIZONTAL,
                variable=self.max_chunk_var, length=70,
                command=lambda v: setattr(self, 'max_audio_duration', float(v))).pack(side=tk.LEFT)
        
        tk.Label(settings_frame, text="Mic Gain:").pack(side=tk.LEFT, padx=(5, 0))
        self.gain_var = tk.DoubleVar(value=2.5)
        tk.Scale(settings_frame, from_=1.0, to=5.0, resolution=0.1, orient=tk.HORIZONTAL,
                variable=self.gain_var, length=70,
                command=lambda v: setattr(self, 'microphone_gain', float(v))).pack(side=tk.LEFT)
        
        self.filter_var = tk.BooleanVar(value=True)
        tk.Checkbutton(settings_frame, text="Filter hallucinations", variable=self.filter_var).pack(side=tk.LEFT, padx=10)
        
        # LLM Correction frame
        llm_frame = tk.Frame(self.root)
        llm_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.llm_enabled_var = tk.BooleanVar(value=False)
        self.llm_check = tk.Checkbutton(
            llm_frame,
            text="🤖 LLM Correction",
            variable=self.llm_enabled_var,
            command=self._toggle_llm_correction,
            font=("Helvetica", 10, "bold")
        )
        self.llm_check.pack(side=tk.LEFT, padx=5)
        
        tk.Label(llm_frame, text="Server:").pack(side=tk.LEFT, padx=(10, 2))
        self.llm_server_var = tk.StringVar(value=self.llm_server)
        server_entry = tk.Entry(llm_frame, textvariable=self.llm_server_var, width=25)
        server_entry.pack(side=tk.LEFT, padx=2)
        
        tk.Label(llm_frame, text="Model:").pack(side=tk.LEFT, padx=(10, 2))
        self.llm_model_var = tk.StringVar(value=self.llm_model)
        self.llm_model_combo = ttk.Combobox(llm_frame, textvariable=self.llm_model_var, width=15, state='readonly')
        self.llm_model_combo['values'] = [self.llm_model]
        self.llm_model_combo.pack(side=tk.LEFT, padx=2)
        
        tk.Button(llm_frame, text="Test", command=self._test_llm_connection, width=6).pack(side=tk.LEFT, padx=5)
        tk.Button(llm_frame, text="Refresh Models", command=self._refresh_llm_models, width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(llm_frame, text="Flush Buffer (Ctrl+F10)", command=self._manual_flush_buffer, width=20).pack(side=tk.LEFT, padx=5)
        
        self.llm_status_label = tk.Label(llm_frame, text="", font=("Helvetica", 9), fg="gray")
        self.llm_status_label.pack(side=tk.RIGHT, padx=5)
        
        # Level frame
        level_frame = tk.Frame(self.root)
        level_frame.pack(fill=tk.X, padx=10, pady=2)
        
        tk.Label(level_frame, text="Audio:").pack(side=tk.LEFT)
        self.level_canvas = tk.Canvas(level_frame, width=150, height=18, bg='black')
        self.level_canvas.pack(side=tk.LEFT, padx=5)
        self.level_bar = self.level_canvas.create_rectangle(0, 0, 0, 18, fill='green')
        
        self.speaking_label = tk.Label(level_frame, text="", font=("Helvetica", 9), fg="blue", width=10)
        self.speaking_label.pack(side=tk.LEFT, padx=3)
        
        self.buffer_label = tk.Label(level_frame, text="", font=("Helvetica", 9), fg="gray", width=10)
        self.buffer_label.pack(side=tk.LEFT, padx=3)
        
        self.speech_ratio_label = tk.Label(level_frame, text="", font=("Helvetica", 9), fg="gray", width=12)
        self.speech_ratio_label.pack(side=tk.LEFT, padx=3)
        
        self.perf_label = tk.Label(level_frame, text="", font=("Helvetica", 9), fg="gray")
        self.perf_label.pack(side=tk.RIGHT, padx=5)
        
        # Text area
        self.text_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, font=("Helvetica", 12))
        self.text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._update_gui()

    def _toggle_recording(self):
        # Check if model is loaded
        if self.model is None:
            messagebox.showwarning("Warning", "Model still loading...")
            return
        
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        self.is_recording = True
        
        with self.audio_lock:
            self.audio_buffer = np.array([], dtype=np.float32)
        self.silence_samples = 0
        self.speech_samples = 0
        self.is_speaking = False
        self.previous_transcription = ""
        self.all_transcriptions = []
        
        # Clear transcription buffer
        self.transcription_buffer.clear()
        self.last_chunk_time = time.time()
        
        while not self.audio_queue.empty():
            try: self.audio_queue.get_nowait()
            except queue.Empty: break
        
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
        # Start buffer check thread if LLM is enabled
        if self.use_llm_correction:
            self.buffer_check_thread = threading.Thread(target=self._buffer_check_loop, daemon=True)
            self.buffer_check_thread.start()
        
        self.audio_stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype=np.float32,
            callback=self._audio_callback, blocksize=int(self.sample_rate * 0.05)
        )
        self.audio_stream.start()
        
        hotkey_text = "Ctrl+F9" if (IS_WINDOWS or HAS_PYNPUT) else ""
        btn_text = f"Stop ({hotkey_text})" if hotkey_text else "Stop"
        self.record_btn.config(text=btn_text, bg="red")
        self.status_label.config(text="Listening...")
        
        self._update_tray(True)
    
    def _buffer_check_loop(self):
        """Periodically check if buffer should be processed."""
        while self.is_recording:
            time.sleep(0.5)  # Check every 500ms
            
            if not self.use_llm_correction or not self.llm_client:
                continue
            
            if self.transcription_buffer.should_process() and not self.llm_processing:
                # User paused, process buffer
                self._process_buffer_with_llm()

    def _stop_recording(self):
        self.is_recording = False
        time.sleep(0.2)
        
        # Process any remaining buffer
        if self.use_llm_correction and not self.transcription_buffer.is_empty():
            self._process_buffer_with_llm()
            time.sleep(1.0)  # Wait for LLM processing
        
        with self.audio_lock:
            if len(self.audio_buffer) > self.sample_rate * self.min_audio_duration:
                if self._calculate_speech_ratio(self.audio_buffer) >= self.min_speech_ratio:
                    self._transcribe(self.audio_buffer.copy(), is_final=True)
                self.audio_buffer = np.array([], dtype=np.float32)
        
        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop()
            self.audio_stream.close()
        
        hotkey_text = "Ctrl+F9" if (IS_WINDOWS or HAS_PYNPUT) else ""
        btn_text = f"Start ({hotkey_text})" if hotkey_text else "Start"
        self.record_btn.config(text=btn_text, bg="green")
        self.status_label.config(text="Ready")
        self.speaking_label.config(text="")
        self.buffer_label.config(text="")
        self.speech_ratio_label.config(text="")
        self.perf_label.config(text="")
        self.level_canvas.coords(self.level_bar, 0, 0, 0, 18)
        
        self._update_tray(False)

    def _audio_callback(self, indata, frames, time_info, status):
        if self.is_recording:
            # Apply microphone gain
            gained_audio = indata.copy().flatten() * self.microphone_gain
            # Clip to prevent distortion
            gained_audio = np.clip(gained_audio, -1.0, 1.0)
            self.audio_queue.put(gained_audio)

    def _process_loop(self):
        while self.is_recording:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            rms = np.sqrt(np.mean(chunk ** 2))
            self.root.after(0, self._update_level, rms)
            
            is_silence = rms < self.silence_threshold
            
            with self.audio_lock:
                self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
                buffer_duration = len(self.audio_buffer) / self.sample_rate
                
                if not is_silence:
                    self.speech_samples += len(chunk)
                    if not self.is_speaking:
                        self.is_speaking = True
                        self.root.after(0, lambda: self.speaking_label.config(text="Speaking..."))
                    self.silence_samples = 0
                else:
                    self.silence_samples += len(chunk)
                
                self.root.after(0, lambda d=buffer_duration: self.buffer_label.config(text=f"Buf: {d:.1f}s"))
                
                if len(self.audio_buffer) > 0:
                    ratio = self.speech_samples / len(self.audio_buffer)
                    self.root.after(0, lambda r=ratio: self.speech_ratio_label.config(text=f"Speech: {r:.0%}"))
                
                silence_duration_current = self.silence_samples / self.sample_rate
                
                should_transcribe = False
                is_continuation = False
                
                if buffer_duration >= self.min_audio_duration:
                    if self.is_speaking and silence_duration_current >= self.silence_duration:
                        should_transcribe = True
                        self.is_speaking = False
                        self.root.after(0, lambda: self.speaking_label.config(text=""))
                    elif buffer_duration >= self.max_audio_duration:
                        should_transcribe = True
                        is_continuation = True
                
                if should_transcribe:
                    # Check if too many transcription threads are active (potential hang)
                    if self.active_transcription_threads >= 3:
                        print(f"WARNING: {self.active_transcription_threads} transcription threads active. Possible hang detected.")
                        # Skip this transcription to prevent further backup
                        self.audio_buffer = np.array([], dtype=np.float32)
                        self.silence_samples = 0
                        self.speech_samples = 0
                        continue
                    
                    speech_ratio = self.speech_samples / len(self.audio_buffer) if len(self.audio_buffer) > 0 else 0
                    
                    if speech_ratio < self.min_speech_ratio:
                        self.audio_buffer = np.array([], dtype=np.float32)
                        self.silence_samples = 0
                        self.speech_samples = 0
                        continue
                    
                    if silence_duration_current > 0.2:
                        trim = min(int((silence_duration_current - 0.2) * self.sample_rate),
                                  len(self.audio_buffer) - int(self.sample_rate * self.min_audio_duration))
                        audio_to_process = self.audio_buffer[:-trim].copy() if trim > 0 else self.audio_buffer.copy()
                    else:
                        audio_to_process = self.audio_buffer.copy()
                    
                    if is_continuation:
                        overlap = int(self.sample_rate * 0.5)
                        if len(self.audio_buffer) > overlap:
                            self.audio_buffer = self.audio_buffer[-overlap:]
                            self.speech_samples = int(self.speech_samples * (overlap / len(audio_to_process)))
                        else:
                            self.audio_buffer = np.array([], dtype=np.float32)
                            self.speech_samples = 0
                    else:
                        self.audio_buffer = np.array([], dtype=np.float32)
                        self.speech_samples = 0
                    self.silence_samples = 0
                    
                    self.active_transcription_threads += 1
                    threading.Thread(target=self._transcribe_wrapper, args=(audio_to_process, False, is_continuation), daemon=True).start()

    def _update_level(self, rms):
        level = min(150, int(rms * 1500))
        color = 'gray' if rms < self.silence_threshold else 'green' if rms < self.silence_threshold * 3 else 'yellow' if rms < self.silence_threshold * 6 else 'red'
        self.level_canvas.coords(self.level_bar, 0, 0, level, 18)
        self.level_canvas.itemconfig(self.level_bar, fill=color)

    def _transcribe_wrapper(self, audio, is_final=False, is_continuation=False):
        """Wrapper to track active transcription threads."""
        try:
            self._transcribe(audio, is_final, is_continuation)
        finally:
            self.active_transcription_threads -= 1
            if self.active_transcription_threads < 0:
                self.active_transcription_threads = 0
    
    def _transcribe(self, audio, is_final=False, is_continuation=False):
        if len(audio) < self.sample_rate * self.min_audio_duration:
            return
        
        if self._calculate_speech_ratio(audio) < self.min_speech_ratio:
            return
        
        try:
            self.root.after(0, lambda: self.status_label.config(text="Transcribing..."))
            
            start_time = time.time()
            
            # Always use faster-whisper general model
            full_text = self._transcribe_with_general_model(audio)
            
            elapsed = time.time() - start_time
            audio_duration = len(audio) / self.sample_rate
            rtf = elapsed / audio_duration
            
            # Warn if transcription is unusually slow
            if rtf > 3.0:
                print(f"WARNING: Slow transcription detected (RTF: {rtf:.2f}x)")
            
            self.root.after(0, lambda: self.perf_label.config(
                text=f"RTF: {rtf:.2f}x",
                fg="green" if rtf < 0.5 else "orange" if rtf < 1 else "red"
            ))
            
            cleaned_text = self._clean_transcription(full_text)
            
            # Log raw output if it looks suspicious
            if cleaned_text and ('_' in cleaned_text or len(cleaned_text) > 500):
                print(f"SUSPICIOUS OUTPUT: {cleaned_text[:200]}...")
            
            if self.filter_var.get() and self._is_hallucination(cleaned_text):
                self.root.after(0, lambda: self.status_label.config(text="Listening..."))
                print(f"Filtered hallucination: {cleaned_text[:50]}...")
                return
            
            if cleaned_text:
                if self.previous_transcription:
                    cleaned_text = self._remove_overlap(self.previous_transcription, cleaned_text)
                
                if cleaned_text:
                    self.previous_transcription = cleaned_text
                    self.all_transcriptions.append(cleaned_text)
                    if len(self.all_transcriptions) > 10:
                        self.all_transcriptions = self.all_transcriptions[-10:]
                    
                    # Handle output based on LLM correction mode
                    if self.use_llm_correction and self.llm_client:
                        # Add to buffer for LLM processing
                        self.transcription_buffer.add_chunk(cleaned_text)
                        self.last_chunk_time = time.time()
                        
                        # Update buffer status
                        buf_size = self.transcription_buffer.size()
                        self.root.after(0, lambda: self.llm_status_label.config(
                            text=f"📝 Buffered: {buf_size} chunks", fg="blue"))
                        
                        mode_tag = "LLM-BUFFERED"
                        print(f"[{mode_tag}] Added to buffer: {cleaned_text[:50]}...")
                        
                        # Check if should process buffer (user paused)
                        if self.transcription_buffer.should_process():
                            threading.Thread(target=self._process_buffer_with_llm, daemon=True).start()
                    else:
                        # Direct output (no LLM correction)
                        self.text_queue.put(cleaned_text)
                        self.last_output_time = time.time()
                        mode_tag = "MED-VOCAB" if self.use_medical_vocab else self.device.upper()
                        print(f"[{mode_tag}] {cleaned_text}")
            
            self.root.after(0, lambda: self.status_label.config(text="Listening..."))
            
        except Exception as e:
            import traceback
            print(f"Transcription error: {e}")
            print(traceback.format_exc())
            self.root.after(0, lambda: self.status_label.config(text="Error - Check Console"))
            # Try to recover by clearing the model cache
            if "CUDA" in str(e) or "memory" in str(e).lower():
                print("Attempting to clear CUDA cache...")
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
    
    def _transcribe_with_general_model(self, audio):
        """Transcribe using faster-whisper general model with optional medical vocabulary."""
        initial_prompt = None
        
        # Build context from previous transcriptions
        if self.all_transcriptions:
            context = " ".join(self.all_transcriptions[-3:])
            words = context.split()
            if len(words) > 5:
                initial_prompt = " ".join(words[-30:])
        
        # Inject medical vocabulary into prompt if enabled
        if self.use_medical_vocab and self.medical_vocabulary:
            # Add a subset of medical terms to help recognition
            vocab_sample = list(self.medical_vocabulary)[:25]
            vocab_hint = ", ".join(vocab_sample)
            if initial_prompt:
                initial_prompt = f"Medical context: {vocab_hint}. {initial_prompt}"
            else:
                initial_prompt = f"Medical terms: {vocab_hint}"
        
        # Limit initial_prompt length to prevent issues
        if initial_prompt and len(initial_prompt) > 500:
            initial_prompt = initial_prompt[:500]
        
        segments, info = self.model.transcribe(
            audio, language="en", beam_size=5, best_of=5, temperature=0.0,
            vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500, threshold=0.5),
            word_timestamps=False, initial_prompt=initial_prompt,
            no_speech_threshold=0.5, log_prob_threshold=-1.0, compression_ratio_threshold=2.4,
        )
        
        text_parts = [seg.text.strip() for seg in segments 
                     if not (hasattr(seg, 'no_speech_prob') and seg.no_speech_prob > 0.5)]
        
        result = " ".join(text_parts).strip()
        
        # Additional validation - reject if result is mostly underscores
        if result and result.count('_') > len(result) * 0.5:
            print(f"Rejecting underscore-heavy output: {result[:100]}")
            return ""
        
        return result
    


    def _remove_overlap(self, previous, current):
        if not previous or not current:
            return current
        
        prev_words = previous.lower().split()
        curr_words = current.split()
        curr_lower = [w.lower() for w in curr_words]
        
        if len(prev_words) < 2 or len(curr_words) < 2:
            return current
        
        best_overlap, best_pos = 0, 0
        for size in range(2, min(15, len(prev_words), len(curr_words)) + 1):
            prev_end = prev_words[-size:]
            for start in range(min(5, len(curr_lower) - size + 1)):
                if curr_lower[start:start+size] == prev_end and size > best_overlap:
                    best_overlap, best_pos = size, start + size
        
        if best_overlap >= 2:
            result = " ".join(curr_words[best_pos:])
            return result if result else ""
        return current

    def _update_gui(self):
        try:
            while True:
                text = self.text_queue.get_nowait()
                if text:
                    mode = self.output_var.get()
                    
                    if mode in ("window", "both"):
                        current = self.text_area.get(1.0, tk.END).strip()
                        if current and not current.endswith((' ', '\n')):
                            self.text_area.insert(tk.END, " ")
                        self.text_area.insert(tk.END, text)
                        self.text_area.see(tk.END)
                    
                    if mode in ("cursor", "both"):
                        threading.Thread(target=self.clipboard.type_text, args=(text,), daemon=True).start()
        except queue.Empty:
            pass
        self.root.after(50, self._update_gui)

    def _clear_text(self):
        self.text_area.delete(1.0, tk.END)
        self.previous_transcription = ""
        self.all_transcriptions = []

    def _copy_text(self):
        text = self.text_area.get(1.0, tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_label.config(text="Copied!")
            self.root.after(2000, lambda: self.status_label.config(
                text="Listening..." if self.is_recording else "Ready"))

    def _show_window(self, *args):
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def _on_close(self):
        if IS_WINDOWS and HAS_TRAY:
            self.root.withdraw()
        else:
            self._quit()

    def _quit(self, *args):
        self.is_recording = False
        time.sleep(0.3)
        
        if self.hotkey_manager:
            self.hotkey_manager.unregister()
        
        try:
            if hasattr(self, 'audio_stream'):
                self.audio_stream.stop()
                self.audio_stream.close()
        except:
            pass
        
        if self.tray_icon:
            self.tray_icon.stop()
        
        self.root.quit()
        self.root.destroy()

    def run(self):
        print("\n" + "="*50)
        print("Whisper Transcriber")
        print("="*50)
        platform_name = "Windows" if IS_WINDOWS else "Linux"
        print(f"Platform: {platform_name}")
        print(f"Device: {self.device.upper()}")
        if IS_WINDOWS or HAS_PYNPUT:
            print("Hotkey: Ctrl+F9")
        print("="*50 + "\n")
        self.root.mainloop()


if __name__ == "__main__":
    app = TranscriberGUI()
    app.run()
