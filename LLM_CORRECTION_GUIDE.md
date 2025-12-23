# LLM Buffered Correction System - User Guide

## Overview

The Whisper Real-Time Transcriber now includes an intelligent LLM (Large Language Model) correction system that fixes sentence fragments, punctuation errors, and transcription breaks automatically.

**Version**: 4.0.0  
**Added**: December 23, 2024  
**Ollama Integration**: llama3.2:3b (configurable)

---

## What Problem Does This Solve?

### Before LLM Correction:
```
Whisper Output:
"The patient has acute."
"Hypertension with chest pain"
"Blood pressure is 180 over"
"95"
```
❌ Sentence fragments  
❌ Incorrect punctuation  
❌ Unnatural breaks  

### After LLM Correction:
```
LLM-Corrected Output:
"The patient has acute hypertension with chest pain. Blood pressure is 180 over 95."
```
✅ Complete sentences  
✅ Proper punctuation  
✅ Natural flow  

---

## How It Works

### Architecture

```
Speech → Whisper → Buffer → LLM Correction → Output (Window/Cursor)
         (0.5s)    (hold)   (1-2s)           (corrected)
```

### Processing Flow

1. **You speak** → Whisper transcribes in real-time
2. **Text chunks buffered** → Held for correction (not output yet)
3. **You pause** → System detects silence (3+ seconds)
4. **LLM processes** → Merges fragments, fixes punctuation
5. **Corrected output** → Clean text appears in window AND at cursor

---

## Setup Instructions

### 1. Make Sure Ollama is Running

Your Ollama server should be running at: **http://192.168.50.134:11434**

Check by running:
```bash
curl http://192.168.50.134:11434/api/tags
```

Should return a list of available models.

### 2. Verify Model Availability

Make sure `llama3.2:3b` is installed:
```bash
ollama list
```

If not installed:
```bash
ollama pull llama3.2:3b
```

### 3. Enable LLM Correction in GUI

1. Launch the transcriber
2. Find the **"🤖 LLM Correction"** section (new row in GUI)
3. Check the checkbox to enable
4. Click **"Test"** button to verify connection
5. Click **"Refresh Models"** to see available models

---

## GUI Controls

### LLM Correction Row

```
┌─────────────────────────────────────────────────────────────────────────┐
│ ☑ 🤖 LLM Correction                                                     │
│ Server: [http://192.168.50.134:11434]                                  │
│ Model: [llama3.2:3b ▼]                                                  │
│ [Test] [Refresh Models] [Flush Buffer (Ctrl+F10)]  ✓ Connected         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Controls Explained

| Control | Purpose |
|---------|---------|
| **☑ LLM Correction** | Toggle buffered correction on/off |
| **Server** | Ollama server URL (editable) |
| **Model** | Select LLM model from dropdown |
| **Test** | Test connection to Ollama server |
| **Refresh Models** | Load available models from server |
| **Flush Buffer** | Force immediate processing |
| **Status Label** | Shows connection/processing status |

### Status Indicators

| Status | Meaning |
|--------|---------|
| `✓ Connected` | LLM server accessible |
| `⚠ Connection failed` | Cannot reach server |
| `📝 Buffered: 3 chunks` | Chunks waiting for processing |
| `🔄 Correcting...` | LLM is processing |
| `✓ Corrected` | Processing complete |
| `⚠ Error` | LLM processing failed |

---

## Usage Workflow

### Basic Transcription with LLM Correction

1. **Enable LLM Correction** (check the checkbox)
2. **Start Recording** (click Start or press Ctrl+F9)
3. **Speak naturally** - Don't worry about pauses or fragments
4. **Pause briefly** (3+ seconds) when you want text processed
5. **See corrected output** - Clean text appears after processing
6. **Continue speaking** - Buffer accumulates again

### Example Session

**You speak:**
```
"The patient presents with acute" [pause]
"hypertension and elevated" [pause]
"blood pressure readings"
[3 second pause]
```

**Console shows:**
```
[LLM-BUFFERED] Added to buffer: The patient presents with acute...
[LLM-BUFFERED] Added to buffer: hypertension and elevated...
[LLM-BUFFERED] Added to buffer: blood pressure readings...
[LLM-CORRECTED] The patient presents with acute hypertension and elevated blood pressure readings.
```

**Output appears:**
```
The patient presents with acute hypertension and elevated blood pressure readings.
```

---

## Configuration Options

### Buffer Settings

Located in code (can be adjusted if needed):
```python
TranscriptionBuffer(
    max_size=5,           # Hold up to 5 chunks
    pause_threshold=3.0   # Process after 3 seconds of silence
)
```

### LLM Settings

- **Server URL**: Default `http://192.168.50.134:11434` (editable in GUI)
- **Model**: Default `llama3.2:3b` (selectable in GUI)
- **Timeout**: 10 seconds per LLM request
- **Temperature**: 0.3 (low randomness for consistency)

### Recommended Models

| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| **llama3.2:3b** | ⚡ Fast (0.5-1s) | Good | General transcription |
| llama3.2:8b | Medium (1-2s) | Better | Complex sentences |
| llama3:8b | Medium (1-2s) | Better | Medical context |
| mistral:7b | Medium (1-2s) | Good | Alternative option |

---

## Operating Modes

### Mode 1: LLM Correction ENABLED (Buffered)

✅ **Pros:**
- Clean, corrected output
- Fixes fragments automatically
- Works with cursor output (no retroactive deletion needed)
- Better for careful, accurate transcription

⚠️ **Cons:**
- 1-3 second delay before text appears
- Requires pause between phrases
- Depends on LLM server availability

**Best for:** Medical notes, formal documents, accurate transcription

### Mode 2: LLM Correction DISABLED (Immediate)

✅ **Pros:**
- Instant output (~0.5s)
- No buffering delay
- No external dependencies
- Familiar behavior

⚠️ **Cons:**
- May have sentence fragments
- Requires manual correction
- Punctuation may be imperfect

**Best for:** Quick notes, rapid dictation, brainstorming

---

## Hotkeys

| Hotkey | Action |
|--------|--------|
| **Ctrl+F9** | Toggle recording on/off |
| **Ctrl+F10** | Flush buffer immediately (force LLM processing) |

### When to Use Ctrl+F10

- You want corrected text NOW (don't want to wait for pause)
- You finished a thought and want to see output
- You're switching to a new topic
- Testing if LLM is working

---

## Single-Word Correction Fix

### Problem (Old Behavior):
1. Transcription outputs: "The patient has hyperension" (typo)
2. You highlight "hyperension" and say: "hypertension"
3. System filters it as hallucination ❌
4. Nothing happens

### Solution (New Behavior):
1. Transcription outputs: "The patient has hyperension" (typo)
2. You highlight "hyperension" and say: "hypertension"
3. System detects recent output (within 10 seconds)
4. Allows single word (context-aware) ✅
5. "hypertension" is transcribed correctly

**Also whitelisted:**
- Medical vocabulary terms (from medical_vocabulary.txt)
- Any word if recent output exists

---

## Troubleshooting

### "⚠ Connection failed" in Status

**Problem:** Cannot reach Ollama server

**Solutions:**
1. Check Ollama is running: `curl http://192.168.50.134:11434/api/tags`
2. Verify server IP/port in GUI
3. Check network connectivity
4. Try clicking **"Test"** button again

### Buffer Never Processes

**Problem:** Chunks accumulate but never get corrected

**Solutions:**
1. Pause for 3+ seconds (required for auto-processing)
2. Press **Ctrl+F10** to force immediate processing
3. Check console for LLM errors
4. Verify model is loaded: `ollama list`

### LLM Takes Too Long (>5 seconds)

**Problem:** Slow LLM processing

**Solutions:**
1. Use smaller model (llama3.2:3b instead of 8b)
2. Check Ollama server resources (CPU/GPU usage)
3. Reduce buffer size in code
4. Consider local Ollama instance (not remote)

### Corrections Are Wrong

**Problem:** LLM merges sentences incorrectly

**Solutions:**
1. Use longer pauses (5+ seconds) between topics
2. Speak more complete sentences
3. Try different model (llama3:8b, mistral:7b)
4. Adjust LLM prompt in code (temperature, instructions)

### Missing Text After Correction

**Problem:** Some chunks disappear

**Solutions:**
1. Check console logs: `[LLM-BUFFERED]` and `[LLM-CORRECTED]`
2. Verify LLM didn't filter as hallucination
3. Try disabling hallucination filter temporarily
4. Increase buffer size

---

## Performance Expectations

### Latency Breakdown

| Stage | Time | Notes |
|-------|------|-------|
| **Whisper transcription** | 0.3-0.5s | GPU-accelerated (RTX 5090) |
| **Buffer accumulation** | Variable | Waits for 3s pause |
| **LLM correction** | 0.5-2.0s | Depends on model size |
| **Total delay** | ~3-5s | From speech to corrected output |

### With llama3.2:3b on Your Setup:
- ⚡ **Whisper**: 0.3s (very fast)
- 📝 **Buffering**: 3s (pause threshold)
- 🤖 **LLM**: 1-1.5s (3B model, network overhead)
- ✅ **Total**: ~4-5s typical

### Is This Acceptable?

**For medical transcription:** YES ✅
- Accuracy more important than speed
- Natural pause time anyway
- Enables corrections without manual editing

**For rapid-fire notes:** Maybe ⚠️
- Consider disabling LLM for speed
- Or use smaller buffer threshold (2s instead of 3s)

---

## Advanced Configuration

### Changing Buffer Threshold

Edit code (line ~427):
```python
self.transcription_buffer = TranscriptionBuffer(
    max_size=5,
    pause_threshold=2.0  # Change from 3.0 to 2.0 for faster processing
)
```

### Changing LLM Timeout

Edit code (OllamaClient class):
```python
self.timeout = 10.0  # Change to 15.0 for slower connections
```

### Custom LLM Prompt

Edit `OllamaClient.correct_text()` method to customize how LLM corrects text.

Current prompt focuses on:
- Merging fragments
- Fixing punctuation
- Preserving original wording
- Respecting long pauses (5+ seconds)

---

## Comparison: With vs Without LLM

### Test Case: Complex Medical Note

**Input Speech:**
```
"Patient presents with" [pause]
"acute chest pain radiating to" [pause]
"the left arm" [pause 5s]
"Blood pressure 180 over 95" [pause]
"Heart rate elevated at" [pause]
"120 beats per minute"
```

### WITHOUT LLM (Immediate Output):
```
Patient presents with
acute chest pain radiating to
the left arm
Blood pressure 180 over 95
Heart rate elevated at
120 beats per minute
```
❌ Requires manual editing

### WITH LLM (Buffered Correction):
```
Patient presents with acute chest pain radiating to the left arm.

Blood pressure 180 over 95. Heart rate elevated at 120 beats per minute.
```
✅ Ready to use immediately

---

## FAQ

### Q: Can I use a different LLM model?

**A:** Yes! Click "Refresh Models" to see all available models. Select any model from the dropdown.

### Q: Does this work offline?

**A:** Yes, as long as Ollama is running locally. The default config assumes Ollama is on your local network.

### Q: Can I disable LLM for specific dictation sessions?

**A:** Yes! Just uncheck the "🤖 LLM Correction" checkbox. Changes take effect immediately.

### Q: Will this slow down my transcription?

**A:** Yes, by 3-5 seconds. But text is cleaner and requires less manual editing.

### Q: Can I use this with cursor output mode?

**A:** Yes! In fact, buffered mode is BETTER for cursor output because text is corrected before typing (no need to delete).

### Q: What if Ollama crashes during dictation?

**A:** System falls back to outputting uncorrected chunks. No data loss.

### Q: Can I use GPT-4 or Claude instead?

**A:** Not currently. The system is designed for Ollama. Adding OpenAI/Anthropic API support is possible but not implemented.

---

## Best Practices

### For General Transcription:

1. ✅ Enable LLM correction
2. ✅ Pause naturally between thoughts (3+ seconds)
3. ✅ Speak complete phrases/sentences
4. ✅ Use Ctrl+F10 when switching topics
5. ✅ Review output in window before relying on cursor mode

### For Medical Transcription:

1. ✅ Enable BOTH Medical Vocabulary + LLM Correction
2. ✅ Use larger model (llama3:8b) for better medical context
3. ✅ Pause between patient observations
4. ✅ Use consistent medical terminology
5. ✅ Verify critical values (blood pressure, medications)

### For Quick Notes:

1. ❌ Disable LLM correction (use immediate mode)
2. ✅ Accept fragments and fix later
3. ✅ Prioritize speed over perfection
4. ✅ Use window output for review

---

## Technical Details

### API Endpoint

**Ollama Generate API:**
```
POST http://192.168.50.134:11434/api/generate
```

**Request:**
```json
{
  "model": "llama3.2:3b",
  "prompt": "Fix any sentence fragments...",
  "stream": false,
  "options": {
    "temperature": 0.3,
    "top_p": 0.9
  }
}
```

**Response:**
```json
{
  "response": "Corrected text here",
  "done": true
}
```

### Buffer Management

- **Data structure:** `collections.deque` (FIFO queue)
- **Max size:** 5 chunks
- **Pause detection:** Compares `time.time()` with last chunk timestamp
- **Thread-safe:** Separate thread checks buffer every 500ms

### LLM Integration

- **Async processing:** Background thread for non-blocking
- **Fallback strategy:** Outputs uncorrected on error
- **Timeout handling:** 10-second timeout per request
- **Error recovery:** Graceful degradation

---

## Support

For issues or questions:

1. Check console output for errors
2. Test LLM connection with "Test" button
3. Verify Ollama is running: `ollama list`
4. Review this guide's Troubleshooting section
5. Open GitHub issue with logs

---

## Version History

**v4.0.0** (2024-12-23)
- ✨ Initial LLM buffered correction system
- ✨ Ollama integration
- ✨ Model selection GUI
- 🐛 Fixed single-word hallucination filtering
- 🐛 Added context-aware correction support

**v3.1.0** (2024-12-23)
- 🐛 Improved transcription hang detection
- 🐛 Enhanced error logging

**v3.0.0** (2024-12-22)
- ✨ Medical vocabulary injection
- ✨ Cross-platform support

---

**Ready to get started?** Enable LLM Correction and experience cleaner, more accurate transcription! 🎉
