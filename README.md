# Whisper Real-Time Transcriber

A cross-platform real-time speech-to-text transcription application powered by OpenAI Whisper and specialized medical transcription models.

![Platform](https://img.shields.io/badge/Platform-Windows%2011%20%7C%20Linux-blue)
![Python](https://img.shields.io/badge/Python-3.11-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Features

- **Real-time Transcription**: Continuous speech-to-text conversion with Voice Activity Detection (VAD)
- **Dual Mode Operation**:
  - **General Mode**: Standard Whisper models (tiny, base, small, medium, large-v2, large-v3)
  - **Medical Mode**: Specialized medical transcription using [Crystalcareai/Whisper-Medicalv1](https://huggingface.co/Crystalcareai/Whisper-Medicalv1)
- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Global Hotkey**: Toggle recording with Ctrl+F9
- **Flexible Output**: Display in window, type at cursor position, or both
- **Medical Vocabulary**: Customizable medical terminology dictionary (`medical_vocabulary.txt`)
- **Hallucination Filtering**: Intelligent filtering of common transcription artifacts
- **Cross-Platform**: Works on Windows 11 and Linux (Ubuntu)
- **System Tray Integration**: Minimize to system tray (Windows)

## 📋 Requirements

### System Requirements
- **Operating System**: Windows 11 or Linux (Ubuntu 20.04+)
- **Python**: 3.11 (recommended)
- **Miniconda**: Already installed
- **CUDA**: (Optional) For GPU acceleration - must be installed separately
  - Download from: https://developer.nvidia.com/cuda-downloads
  - Recommended: CUDA 11.8 or 12.1+
  - **Note**: CUDA is NOT included in the installation scripts

### Hardware Requirements
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **RAM**: 
  - Minimum: 4GB (for tiny/base models on CPU)
  - Recommended: 8GB+ (for larger models)
  - Medical Mode: 8GB+ recommended
- **GPU** (Optional but recommended):
  - NVIDIA GPU with 4GB+ VRAM for base/small models
  - 6GB+ VRAM for medium models
  - 8GB+ VRAM for large models
- **Microphone**: Any standard audio input device

## 🚀 Quick Start

### 1. Clone or Download
```bash
cd /path/to/your/workspace
git clone <repository-url>
cd webapp
```

### 2. Run Setup (First Time Only)
Double-click `setup.bat` or run from command prompt:
```cmd
setup.bat
```

This will:
- Create a conda environment named `whisper-transcriber`
- Install Python 3.11
- Install PyTorch (with CUDA support if available)
- Install all required dependencies

### 3. Launch the Application
Double-click `launch.bat` or run from command prompt:
```cmd
launch.bat
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

### Manual Installation Steps

If you prefer manual installation:

```cmd
# Create conda environment
conda create -n whisper-transcriber python=3.11 -y

# Activate environment
conda activate whisper-transcriber

# Install PyTorch with CUDA (if you have CUDA installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# OR install PyTorch CPU-only (if no CUDA)
pip install torch torchvision torchaudio

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Usage

### Basic Operation

1. **Launch** the application using `launch.bat`
2. **Select Device**: Choose between CPU or GPU (if available)
3. **Choose Model**: Select from tiny, base, small, medium, or large-v3
4. **Select Mode**:
   - Keep "Medical Mode: OFF" for general transcription
   - Click "Medical Mode: ON" for medical transcription
5. **Configure Output**: Choose Window, Cursor, or Both
6. **Start Recording**: Click "Start" button or press **Ctrl+F9**
7. **Speak**: The transcription will appear in real-time
8. **Stop Recording**: Click "Stop" or press **Ctrl+F9** again

### Hotkey Controls

- **Ctrl+F9**: Toggle recording on/off (global hotkey works even when app is minimized)

### Model Selection Guide

| Model | Size | Speed | Accuracy | Best For | RAM/VRAM |
|-------|------|-------|----------|----------|----------|
| tiny | 39 MB | Fastest | Good | Quick notes, CPU mode | 2GB |
| base | 74 MB | Fast | Better | General use, CPU/GPU | 2GB |
| small | 244 MB | Moderate | Very Good | Balanced accuracy/speed | 4GB |
| medium | 769 MB | Slower | Excellent | High accuracy needs | 6GB |
| large-v3 | 1550 MB | Slowest | Best | Maximum accuracy | 8GB+ |
| **Medical** | ~1500 MB | Moderate | Excellent (Medical) | Medical transcription | 8GB+ |

### Medical Transcription Mode

#### Activating Medical Mode
1. Click the **"Medical Mode: OFF"** button
2. Button will turn green and show **"Medical Mode: ON"**
3. The medical model will download on first use (~1.5GB)
4. Start recording as normal

#### Medical Vocabulary Customization
Edit `medical_vocabulary.txt` to add your custom medical terms:

```text
# Add your terms here (one per line)
laparotomy
thoracotomy
your_custom_medical_term
```

The app will:
- Load vocabulary on startup
- Use it to enhance transcription accuracy
- Provide better recognition of medical terminology

## ⚙️ Configuration

### Audio Settings (adjustable in GUI)

- **Silence Threshold**: Audio level below which speech is considered silence (0.005 - 0.05)
- **Pause Duration**: How long to wait after silence before transcribing (0.5 - 3.0 seconds)
- **Max Chunk**: Maximum duration of a single transcription chunk (10 - 30 seconds)

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
# Verify CUDA installation
nvcc --version

# If CUDA is installed, reinstall PyTorch with CUDA
conda activate whisper-transcriber
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Hotkey Not Working
**Problem**: Ctrl+F9 doesn't toggle recording

**Solution**:
- **Windows**: Run the application as Administrator
- **Linux**: Install pynput: `pip install pynput`

#### 3. Medical Model Fails to Load
**Problem**: Error when switching to Medical Mode

**Solution**:
```cmd
conda activate whisper-transcriber
pip install transformers huggingface-hub --upgrade
```

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

### Performance Tips

1. **For Best Speed**: Use GPU with tiny/base model
2. **For Best Accuracy**: Use GPU with large-v3 or Medical model
3. **CPU Mode**: Stick to tiny or base models
4. **Medical Mode**: Requires 8GB+ RAM and benefits significantly from GPU

## 📁 Project Structure

```
webapp/
├── realtime_transcriber_cross_platform.py   # Main application
├── setup.bat                                 # Environment setup script
├── launch.bat                                # Application launcher
├── requirements.txt                          # Python dependencies
├── medical_vocabulary.txt                    # Medical terms dictionary
├── README.md                                 # This file
├── models/                                   # General Whisper models (auto-created)
└── medical_models/                           # Medical model cache (auto-created)
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
To free up disk space, you can delete the model directories:
- `models/` - General Whisper models
- `medical_models/` - Medical model cache

Models will be re-downloaded when needed.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Original Whisper model
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Optimized Whisper implementation
- [Crystalcareai/Whisper-Medicalv1](https://huggingface.co/Crystalcareai/Whisper-Medicalv1) - Medical transcription model
- [Hugging Face](https://huggingface.co/) - Model hosting and transformers library

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review the troubleshooting section above

## 🔮 Future Enhancements

- [ ] Support for multiple languages in medical mode
- [ ] Custom model fine-tuning guide
- [ ] Export transcriptions to file (txt, docx, pdf)
- [ ] Timestamped transcription option
- [ ] Speaker diarization support
- [ ] Integration with Electronic Health Records (EHR) systems

---

**Version**: 2.0.0  
**Last Updated**: December 2024  
**Tested On**: Windows 11, Ubuntu 22.04
