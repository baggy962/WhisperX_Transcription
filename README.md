# Whisper Real-Time Transcriber

A cross-platform real-time speech-to-text transcription application powered by OpenAI Whisper with LLM-based intelligent correction, customizable medical vocabulary injection, and advanced post-processing for enhanced transcription accuracy.

![Platform](https://img.shields.io/badge/Platform-Windows%2011%20%7C%20Linux-blue)
![Python](https://img.shields.io/badge/Python-3.11-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Features

### Core Transcription
- **Real-time Transcription**: Continuous speech-to-text conversion with Voice Activity Detection (VAD)
- **Multiple Whisper Models**: Choose from tiny, base, small, medium, large-v2, large-v3
- **GPU Acceleration**: Automatic CUDA detection and utilization (RTX 5090 supported)
- **Microphone Gain Control**: Adjustable 1.0x-5.0x amplification for quiet microphones
- **Global Hotkeys**: Ctrl+F9 to toggle recording, Ctrl+F10 to flush buffer
- **Flexible Output**: Display in window, type at cursor position, or both
- **Cross-Platform**: Works on Windows 11 and Linux (Ubuntu)
- **System Tray Integration**: Minimize to system tray (Windows)

### 🤖 NEW: LLM Intelligent Correction (v4.0)
- **Buffered Correction**: Automatically fixes sentence fragments and punctuation
- **Ollama Integration**: Uses local LLM (llama3.2:3b by default) for post-processing
- **Pause Detection**: Processes buffer when you pause (3-second threshold)
- **Model Selection**: Choose from any installed Ollama model via GUI
- **Clean Output**: Corrected text appears in window AND at cursor position
- **Manual Flush**: Force immediate processing with Ctrl+F10 hotkey
- **[See Full LLM Guide →](LLM_CORRECTION_GUIDE.md)**

### Medical & Vocabulary Features
- **Medical Vocabulary Injection**: Optional medical terminology context for improved transcription
- **Customizable Vocabulary**: Edit `medical_vocabulary.txt` to add your own terms
- **Context-Aware Filtering**: Intelligent hallucination filtering with single-word correction support
- **Hallucination Filtering**: Smart filtering of common transcription artifacts

## 📋 Requirements

### System Requirements
- **Operating System**: Windows 11 or Linux (Ubuntu 20.04+)
- **Python**: 3.11 (recommended)
- **Miniconda**: Already installed
- **Ollama** (Optional, for LLM correction): Local LLM server
  - Download from: https://ollama.ai
  - Install models: `ollama pull llama3.2:3b`
  - Default server: http://192.168.50.134:11434 (configurable)
- **CUDA**: (Optional) For GPU acceleration - must be installed separately
  - Download from: https://developer.nvidia.com/cuda-downloads
  - For RTX 50-series (5090): CUDA 12.4+ required
  - For RTX 40-series: CUDA 12.1 recommended
  - For RTX 30-series: CUDA 12.1 or 11.8
  - For RTX 20-series: CUDA 11.8 recommended
  - **Note**: CUDA is NOT included in the installation scripts

### Hardware Requirements
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **RAM**: 
  - Minimum: 4GB (for tiny/base models on CPU)
  - Recommended: 8GB+ (for larger models)
  - With Medical Vocabulary: 8GB+ recommended
  - With LLM Correction: 16GB+ recommended
- **GPU** (Optional but recommended):
  - NVIDIA GPU with 4GB+ VRAM for base/small models
  - 6GB+ VRAM for medium models
  - 8GB+ VRAM for large models
  - RTX 5090 (32GB): Excellent performance with all models + LLM
- **Microphone**: Any standard audio input device

## 🚀 Quick Start

### 1. Clone or Download
```bash
cd /path/to/your/workspace
git clone https://github.com/baggy962/WhisperX_Transcription.git
cd WhisperX_Transcription
```

### 2. Run Setup (First Time Only)

#### For Command Prompt (cmd.exe) - Recommended
Double-click `setup.bat` or run from command prompt:
```cmd
setup.bat
```

#### For PowerShell
```powershell
.\setup.ps1
```

This will:
- Create a conda environment named `whisper-transcriber`
- Install Python 3.11
- Install PyTorch (with CUDA support if available)
- Install all required dependencies

### 3. GPU Setup (Optional but Recommended)

If you have an NVIDIA GPU and CUDA installed, run the appropriate fix script:

#### For RTX 5090 (Blackwell - Compute Capability 10.0)
```cmd
fix_rtx5090.bat
```

#### For Other GPUs
```cmd
fix_gpu.bat
```
OR
```powershell
.\fix_gpu.ps1
```

#### Diagnostic Tool
To check your GPU and get recommendations:
```cmd
diagnose_gpu.bat
```

### 4. Launch the Application

#### For Command Prompt
Double-click `launch.bat` or run:
```cmd
launch.bat
```

#### For PowerShell
```powershell
.\launch.ps1
```

#### Manual Launch
```cmd
conda activate whisper-transcriber
python realtime_transcriber_cross_platform.py
```

## 📖 Detailed Installation

### Prerequisites Check

#### Verify Miniconda Installation
```cmd
conda --version
```
If conda is not found, download and install from: https://docs.conda.io/en/latest/miniconda.html

#### Verify CUDA Installation (for GPU support)
```cmd
nvcc --version
```
If CUDA is not found and you have an NVIDIA GPU, install from: https://developer.nvidia.com/cuda-downloads

#### Verify GPU Detection
```cmd
nvidia-smi
```
Should show your NVIDIA GPU name and VRAM

### Manual Installation Steps

If you prefer manual installation:

```cmd
# Create conda environment
conda create -n whisper-transcriber python=3.11 -y

# Activate environment
conda activate whisper-transcriber

# Install PyTorch with CUDA 12.4 (for RTX 5090)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# OR install PyTorch with CUDA 12.1 (for RTX 40/30-series)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# OR install PyTorch with CUDA 11.8 (for RTX 20/GTX 10-series)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# OR install PyTorch CPU-only (if no CUDA)
pip install torch torchvision torchaudio

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Usage

### Basic Operation

1. **Launch** the application using `launch.bat` or `launch.ps1`
2. **Select Device**: Choose between CPU or GPU (if available)
3. **Choose Model**: Select from tiny, base, small, medium, or large-v3
4. **Medical Vocabulary**: Check "Use Medical Vocabulary" for medical transcription
5. **Adjust Mic Gain**: Set microphone gain (1.0x - 5.0x, default 2.5x)
6. **Configure Output**: Choose Window, Cursor, or Both
7. **Start Recording**: Click "Start" button or press **Ctrl+F9**
8. **Speak**: The transcription will appear in real-time
9. **Stop Recording**: Click "Stop" or press **Ctrl+F9** again

### Hotkey Controls

- **Ctrl+F9**: Toggle recording on/off (global hotkey works even when app is minimized)
- **Ctrl+F10**: Flush LLM buffer (force immediate processing of buffered text)

### 🤖 LLM Intelligent Correction (NEW in v4.0)

#### What is LLM Correction?

LLM Correction uses a local Large Language Model (Ollama) to automatically fix:
- ✅ Sentence fragments caused by pauses
- ✅ Incorrect punctuation and capitalization  
- ✅ Unnatural transcription breaks
- ✅ Grammar and flow issues

**Example:**
```
WITHOUT LLM:                    WITH LLM:
"The patient has acute"    →   "The patient has acute hypertension
"Hypertension with"             with chest pain. Blood pressure
"chest pain"                    is 180 over 95."
"Blood pressure is 180"
"over 95"
```

#### How to Use LLM Correction

1. **Install Ollama** (if not already installed):
   ```bash
   curl https://ollama.ai/install.sh | sh
   ollama pull llama3.2:3b
   ```

2. **Enable in GUI**:
   - Check the **"🤖 LLM Correction"** checkbox
   - Configure server URL (default: http://192.168.50.134:11434)
   - Select model from dropdown
   - Click "Test" to verify connection

3. **Dictate normally**:
   - Text is buffered as you speak
   - Pause for 3+ seconds to trigger processing
   - LLM corrects and outputs clean text
   - Or press **Ctrl+F10** to force immediate processing

4. **See Results**:
   - Status shows: `📝 Buffered: 3 chunks` while accumulating
   - Then: `🔄 Correcting...` during LLM processing
   - Finally: `✓ Corrected` when complete

#### When to Use LLM Correction

✅ **Use when:**
- Dictating medical notes or formal documents
- Accuracy is more important than speed
- You naturally pause between thoughts
- You want clean output without manual editing

❌ **Disable when:**
- You need instant feedback
- Taking quick, fragmented notes
- Ollama is not available
- Speed is critical

**Performance:** Adds 3-5 second delay (includes pause + processing), but output is significantly cleaner.

**[→ Full LLM Correction Guide](LLM_CORRECTION_GUIDE.md)** for detailed setup and troubleshooting.

### Model Selection Guide

| Model | Size | Speed | Accuracy | Best For | RAM/VRAM |
|-------|------|-------|----------|----------|----------|
| tiny | 39 MB | Fastest | Good | Quick notes, CPU mode | 2GB |
| base | 74 MB | Fast | Better | General use, CPU/GPU | 2GB |
| small | 244 MB | Moderate | Very Good | Balanced accuracy/speed | 4GB |
| medium | 769 MB | Slower | Excellent | High accuracy needs | 6GB |
| large-v3 | 1550 MB | Slowest | Best | Maximum accuracy | 8GB+ |

### Medical Transcription with Vocabulary Injection

#### How It Works
Instead of using a separate medical model, this app uses **vocabulary injection** - adding medical terms to the transcription context to guide the standard Whisper model. This approach:
- ✅ Works reliably on all GPUs (no CUDA compatibility issues)
- ✅ Faster than dedicated medical models
- ✅ Uses proven, stable Whisper models
- ✅ Customizable with your own terminology

#### Using Medical Vocabulary
1. Check the **"Use Medical Vocabulary"** checkbox
2. The app will inject medical terms from `medical_vocabulary.txt` into the transcription context
3. Whisper will be guided to recognize these terms more accurately

#### Customizing Medical Vocabulary
Edit `medical_vocabulary.txt` to add your custom medical terms:

```text
# Add your terms here (one per line, case-insensitive)
# Lines starting with # are comments

# Common Medical Terms
acetaminophen
antibiotic
cardiovascular
laparotomy
thoracotomy

# Add your custom medical terms below
your_custom_medical_term
```

The app will:
- Load vocabulary on startup
- Inject a sample of terms into each transcription context
- Guide Whisper to recognize your medical terminology
- Provide better accuracy for medical transcriptions

### Microphone Gain Settings

If your microphone is too quiet (e.g., showing only 38% volume at 100% Windows level):

- **Recommended**: Set Mic Gain to **3.0-3.5x**
- **For very quiet mics**: Try **4.0-5.0x**
- **After Windows mic boost**: Use **2.0-2.5x**
- Audio is automatically clipped to prevent distortion

## ⚙️ Configuration

### Audio Settings (adjustable in GUI)

- **Silence Threshold**: Audio level below which speech is considered silence (0.005 - 0.05)
- **Pause Duration**: How long to wait after silence before transcribing (0.5 - 3.0 seconds)
- **Max Chunk**: Maximum duration of a single transcription chunk (10 - 30 seconds)
- **Mic Gain**: Microphone amplification (1.0x - 5.0x, default 2.5x)

### Advanced Settings

Edit the Python script to customize:
- Sample rate (default: 16000 Hz)
- Minimum audio duration (default: 0.8 seconds)
- Minimum speech ratio (default: 0.15)

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Not Detected
**Problem**: GPU not being used even though CUDA is installed

**Solution**:
```cmd
# Check CUDA version
nvcc --version

# Check GPU
nvidia-smi

# Run diagnostic tool
diagnose_gpu.bat

# For RTX 5090, use dedicated fix
fix_rtx5090.bat

# For other GPUs
fix_gpu.bat
```

**Manual Fix**:
```cmd
conda activate whisper-transcriber
pip uninstall torch torchvision torchaudio -y

# For RTX 5090 (Compute Capability 10.0) - requires CUDA 12.4+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# For RTX 40/30-series
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For RTX 20-series or older
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Low Microphone Volume
**Problem**: Microphone input shows only 38% even at 100% Windows volume

**Solution**:
- Use the **Mic Gain** slider in the app
- Set to **3.0x** or higher
- Audio is automatically clipped to prevent distortion

#### 3. Hotkey Not Working
**Problem**: Ctrl+F9 doesn't toggle recording

**Solution**:
- **Windows**: Run the application as Administrator
- **Linux**: Install pynput: `pip install pynput`

#### 4. Audio Device Not Found
**Problem**: "No audio device" error

**Solution**:
- Check microphone is connected and working
- Windows: Check Privacy Settings → Microphone access
- Linux: Install PortAudio: `sudo apt-get install portaudio19-dev`

#### 5. Memory Error with Large Models
**Problem**: Out of memory error

**Solution**:
- Use a smaller model (tiny, base, or small)
- Close other applications
- Use CPU mode instead of GPU for large models

#### 6. Batch File Stops After "Checking conda installation..."
**Problem**: Setup stops in PowerShell or cmd

**Solution**:
- Use `setup.ps1` for PowerShell: `.\setup.ps1`
- Use `setup.bat` for Command Prompt (cmd.exe)
- OR double-click the file in File Explorer

### Performance Tips

1. **For Best Speed**: Use GPU with tiny/base model
2. **For Best Accuracy**: Use GPU with large-v3 + medical vocabulary
3. **CPU Mode**: Stick to tiny or base models
4. **Medical Vocabulary**: Works best with medium/large models on GPU
5. **RTX 5090**: Can handle large-v3 in real-time with RTF < 0.3x

### Expected Performance (Real-Time Factor - RTF)

| GPU | Model | RTF | Notes |
|-----|-------|-----|-------|
| RTX 5090 | large-v3 | 0.3x | 10x faster than real-time |
| RTX 5090 | base | 0.1x | 30x faster than real-time |
| RTX 4090 | large-v3 | 0.4x | 8x faster than real-time |
| RTX 3080 | large-v3 | 0.6x | 5x faster than real-time |
| CPU (i9) | base | 2.0x | 2x slower than real-time |
| CPU (i9) | large-v3 | 8.0x | 8x slower than real-time |

*RTF < 1.0 means faster than real-time*

## 📁 Project Structure

```
WhisperX_Transcription/
├── realtime_transcriber_cross_platform.py   # Main application
├── setup.bat                                 # Environment setup (cmd)
├── setup.ps1                                 # Environment setup (PowerShell)
├── launch.bat                                # Application launcher (cmd)
├── launch.ps1                                # Application launcher (PowerShell)
├── fix_gpu.bat                               # GPU fix script (general)
├── fix_gpu.ps1                               # GPU fix script (PowerShell)
├── fix_rtx5090.bat                           # GPU fix for RTX 5090
├── diagnose_gpu.bat                          # GPU diagnostic tool
├── requirements.txt                          # Python dependencies
├── medical_vocabulary.txt                    # Medical terms dictionary
├── README.md                                 # This file
└── models/                                   # Whisper models (auto-created)
```

## 🔄 Updates and Maintenance

### Updating Dependencies
```cmd
conda activate whisper-transcriber
pip install -r requirements.txt --upgrade
```

### Updating Medical Vocabulary
Simply edit `medical_vocabulary.txt` and restart the application. The new terms will be loaded automatically.

### Clearing Model Cache
To free up disk space, you can delete the model directory:
- `models/` - Whisper models

Models will be re-downloaded when needed.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Original Whisper model
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Optimized Whisper implementation
- Community contributors for medical vocabulary terms

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub: https://github.com/baggy962/WhisperX_Transcription
- Check existing issues for solutions
- Review the troubleshooting section above

## 🔮 Future Enhancements

- [ ] Support for multiple languages with vocabulary injection
- [ ] Export transcriptions to file (txt, docx, pdf)
- [ ] Timestamped transcription option
- [ ] Speaker diarization support
- [ ] Batch processing of audio files
- [ ] Real-time translation mode

---

**Version**: 3.0.0  
**Last Updated**: December 2024  
**Tested On**: Windows 11, Ubuntu 22.04, RTX 5090
