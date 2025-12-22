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
from tkinter import scrolledtext, messagebox
import os
import sys

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
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            devices["cuda"] = {
                "available": True,
                "name": f"{gpu_name} ({vram:.1f}GB)",
                "compute_type": "float16"
            }
    except:
        pass
    
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


class TranscriberGUI:
    
    HALLUCINATIONS = [
        "thank you", "thanks for watching", "thanks for listening",
        "subscribe", "like and subscribe", "see you next time",
        "bye", "goodbye", "the end", "music", "applause", "laughter",
        "silence", "...", "you", "i'm sorry", "sorry", "okay", "ok",
        "um", "uh", "hmm", "mhm",
    ]
    
    MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
    
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

    def _setup_platform_features(self):
        """Set up platform-specific features."""
        
        # Hotkey
        self.hotkey_manager = HotkeyManager(lambda: self.root.after(0, self._toggle_recording))
        self.hotkey_manager.register()
        
        # Tray icon (Windows only)
        if IS_WINDOWS and HAS_TRAY:
            self._create_tray()

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
        
        for h in self.HALLUCINATIONS:
            if text_clean == h:
                return True
        
        words = text_clean.split()
        if len(words) <= 1 and len(text_clean) < 10:
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
        
        self.filter_var = tk.BooleanVar(value=True)
        tk.Checkbutton(settings_frame, text="Filter hallucinations", variable=self.filter_var).pack(side=tk.LEFT, padx=10)
        
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
        
        while not self.audio_queue.empty():
            try: self.audio_queue.get_nowait()
            except queue.Empty: break
        
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
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

    def _stop_recording(self):
        self.is_recording = False
        time.sleep(0.2)
        
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
            self.audio_queue.put(indata.copy().flatten())

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
                    
                    threading.Thread(target=self._transcribe, args=(audio_to_process, False, is_continuation), daemon=True).start()

    def _update_level(self, rms):
        level = min(150, int(rms * 1500))
        color = 'gray' if rms < self.silence_threshold else 'green' if rms < self.silence_threshold * 3 else 'yellow' if rms < self.silence_threshold * 6 else 'red'
        self.level_canvas.coords(self.level_bar, 0, 0, level, 18)
        self.level_canvas.itemconfig(self.level_bar, fill=color)

    def _transcribe(self, audio, is_final=False, is_continuation=False):
        if len(audio) < self.sample_rate * self.min_audio_duration:
            return
        
        if self._calculate_speech_ratio(audio) < self.min_speech_ratio:
            return
        
        try:
            self.root.after(0, lambda: self.status_label.config(text="Transcribing..."))
            
            start_time = time.time()
            
            initial_prompt = None
            if self.all_transcriptions:
                context = " ".join(self.all_transcriptions[-3:])
                words = context.split()
                if len(words) > 5:
                    initial_prompt = " ".join(words[-30:])
            
            segments, info = self.model.transcribe(
                audio, language="en", beam_size=5, best_of=5, temperature=0.0,
                vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500, threshold=0.5),
                word_timestamps=False, initial_prompt=initial_prompt,
                no_speech_threshold=0.5, log_prob_threshold=-1.0, compression_ratio_threshold=2.4,
            )
            
            elapsed = time.time() - start_time
            audio_duration = len(audio) / self.sample_rate
            rtf = elapsed / audio_duration
            
            self.root.after(0, lambda: self.perf_label.config(
                text=f"RTF: {rtf:.2f}x",
                fg="green" if rtf < 0.5 else "orange" if rtf < 1 else "red"
            ))
            
            text_parts = [seg.text.strip() for seg in segments 
                         if not (hasattr(seg, 'no_speech_prob') and seg.no_speech_prob > 0.5)]
            
            full_text = " ".join(text_parts).strip()
            cleaned_text = self._clean_transcription(full_text)
            
            if self.filter_var.get() and self._is_hallucination(cleaned_text):
                self.root.after(0, lambda: self.status_label.config(text="Listening..."))
                return
            
            if cleaned_text:
                if self.previous_transcription:
                    cleaned_text = self._remove_overlap(self.previous_transcription, cleaned_text)
                
                if cleaned_text:
                    self.previous_transcription = cleaned_text
                    self.all_transcriptions.append(cleaned_text)
                    if len(self.all_transcriptions) > 10:
                        self.all_transcriptions = self.all_transcriptions[-10:]
                    self.text_queue.put(cleaned_text)
                    print(f"[{self.device.upper()}] {cleaned_text}")
            
            self.root.after(0, lambda: self.status_label.config(text="Listening..."))
            
        except Exception as e:
            print(f"Transcription error: {e}")
            self.root.after(0, lambda: self.status_label.config(text="Error"))

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
