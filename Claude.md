# VoiceChanger (VCClient) - Agent Documentation

## Project Overview

VoiceChanger is a **real-time voice conversion application** that uses AI models to transform human speech in real-time. It supports multiple voice conversion models and operates in both standalone and network-distributed configurations.

- **Repository:** https://github.com/w-okada/voice-changer
- **Version:** v2.0.78-beta
- **License:** ISC (with custom licenses for specific voice models)
- **Primary Language:** Japanese (with i18n support for English, Korean, Chinese, etc.)

### Supported Voice Models
| Model | Description | Status |
|-------|-------------|--------|
| RVC | Retrieval-based Voice Conversion | Primary, fully supported |
| Beatrice v2 | Commercial voice model | Static slot only |
| MMVC v13/v15 | Japanese voice conversion | Supported |
| so-vits-svc 4.0 | Singing voice synthesis | Supported |
| DDSP-SVC | Differentiable DSP | Supported |
| Diffusion-SVC | Diffusion-based conversion | Supported |
| LLVC | Lightweight conversion | Supported |
| EasyVC | Simplified RVC variant | Supported |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Browser/Client                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ React Demo App (client/demo)                                 │   │
│  │   └── VoiceChangerClient (client/lib)                       │   │
│  │         ├── ServerRestClient (REST API)                     │   │
│  │         └── VoiceChangerWorkletNode (AudioWorklet)          │   │
│  └─────────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTP/WebSocket
┌──────────────────────────────▼──────────────────────────────────────┐
│                         Python Server                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ MMVCServerSIO.py (Entry Point)                              │   │
│  │   ├── MMVC_Rest (FastAPI REST endpoints)                    │   │
│  │   └── MMVC_SocketIOApp (Socket.IO real-time streaming)      │   │
│  │                                                             │   │
│  │ VoiceChangerManager (Singleton orchestrator)                │   │
│  │   └── ModelSlotManager                                      │   │
│  │         └── Model Implementations (RVC, Beatrice, etc.)     │   │
│  │               ├── Inferencers (PyTorch, ONNX)               │   │
│  │               ├── Embedders (ContentVec, Hubert, Whisper)   │   │
│  │               └── PitchExtractors (crepe, rmvpe, fcpe)      │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Audio Processing Flow
```
Browser Mic → AudioContext → VoiceChangerWorkletNode
    ↓
REST/Socket.IO → Python Server
    ↓
VoiceChangerManager → Model.inference()
    ↓
Response → VoiceChangerWorkletNode → Browser Output
```

### Sample Rate Chain
```
Input (48kHz) → Resample (16kHz) → Model Processing → Model Output (40kHz) → Resample (48kHz) → Output
```

---

## Directory Structure

```
voice-changer/
├── server/                     # Python backend (FastAPI + Socket.IO)
│   ├── MMVCServerSIO.py       # MAIN ENTRY POINT
│   ├── requirements.txt       # Python dependencies (Linux/Windows)
│   ├── requirements-mac.txt   # Mac-specific dependencies
│   ├── setup-mac.sh           # Mac conda environment setup
│   ├── start-mac.sh           # Mac server startup script
│   ├── const.py               # Types, enums, constants
│   ├── Exceptions.py          # Custom exception classes
│   ├── restapi/               # REST API endpoints
│   │   ├── MMVC_Rest.py       # FastAPI app setup
│   │   ├── MMVC_Rest_VoiceChanger.py  # Voice conversion endpoints
│   │   └── MMVC_Rest_Fileuploader.py  # File upload & model management
│   ├── sio/                   # Socket.IO server
│   │   ├── MMVC_SocketIOApp.py
│   │   ├── MMVC_SocketIOServer.py
│   │   └── MMVC_Namespace.py  # Real-time audio handler
│   ├── data/                  # Data structures
│   │   └── ModelSlot.py       # Model slot definitions
│   └── voice_changer/         # Core voice conversion logic
│       ├── VoiceChangerManager.py    # Main orchestrator (singleton)
│       ├── VoiceChanger.py           # V1 inference pipeline
│       ├── VoiceChangerV2.py         # V2 optimized pipeline
│       ├── ModelSlotManager.py       # Model slot handling
│       ├── RVC/                      # RVC model implementation
│       │   ├── RVC.py, RVCr2.py      # RVC entry points
│       │   ├── RVCSettings.py        # RVC configuration
│       │   ├── pipeline/             # Processing pipeline
│       │   │   ├── Pipeline.py       # Main pipeline class
│       │   │   └── PipelineGenerator.py  # Pipeline factory
│       │   ├── inferencer/           # Inference backends
│       │   │   ├── RVCInferencerv2.py
│       │   │   ├── RVCInferencerv2Nono.py  # No-f0 models
│       │   │   └── rvc_models/       # Model architectures
│       │   ├── embedder/             # Feature extractors
│       │   │   ├── EmbedderManager.py
│       │   │   ├── FairseqHubert.py  # HuBERT embedder
│       │   │   ├── FairseqContentvec.py
│       │   │   └── Whisper.py
│       │   ├── pitchExtractor/       # Pitch detection
│       │   │   ├── PitchExtractorManager.py
│       │   │   ├── RMVPEPitchExtractor.py
│       │   │   └── CrepePitchExtractor.py
│       │   └── deviceManager/        # GPU/CPU management
│       │       └── DeviceManager.py
│       ├── Beatrice/                 # Beatrice model
│       ├── DDSP_SVC/                 # DDSP-SVC model
│       ├── DiffusionSVC/             # Diffusion-SVC model
│       ├── SoVitsSvc40/              # so-vits-svc model
│       ├── LLVC/                     # LLVC model
│       ├── EasyVC/                   # EasyVC model
│       └── Local/                    # Local device audio I/O
│           └── ServerDevice.py
│
├── client/                    # TypeScript/React frontend
│   ├── lib/                   # Reusable client library
│   │   ├── package.json
│   │   ├── src/
│   │   │   ├── VoiceChangerClient.ts      # Main client class
│   │   │   ├── client/
│   │   │   │   ├── ServerRestClient.ts    # REST communication
│   │   │   │   └── VoiceChangerWorkletNode.ts  # Audio processing
│   │   │   └── const.ts                   # Types and constants
│   │   └── worklet/           # AudioWorklet code
│   │
│   └── demo/                  # Demo web application
│       ├── package.json
│       ├── src/
│       │   ├── 000_index.tsx  # React entry point
│       │   ├── 001_provider/  # React context providers
│       │   ├── 001_globalHooks/  # Global state hooks
│       │   └── components/    # UI components
│       └── public/            # Static assets
│
├── recorder/                  # Voice recording UI (separate app)
├── docker/                    # Docker build files
├── docker_vcclient/          # VCClient Docker image
├── docker_trainer/           # Training Docker image
├── docker-compose.yml        # Multi-container orchestration
├── tutorials/                 # User documentation
├── docs_i18n/                # Internationalized documentation
├── README_mac.md             # Mac-specific documentation
└── Claude.md                 # This file (agent documentation)
```

---

## Key Files Reference

### Server Core
| File | Purpose |
|------|---------|
| `server/MMVCServerSIO.py` | Server entry point - argument parsing, startup, HTTPS setup |
| `server/const.py` | Type definitions (`VoiceChangerType`, `PitchExtractorType`, `EmbedderType`) |
| `server/Exceptions.py` | Custom exceptions (`NoModeLoadedException`, `HalfPrecisionChangingException`) |
| `server/voice_changer/VoiceChangerManager.py` | Singleton orchestrator for all models |
| `server/voice_changer/VoiceChanger.py` | Audio processing with SOLA buffering |
| `server/voice_changer/ModelSlotManager.py` | Model slot management (up to 500 slots) |

### RVC Implementation
| File | Purpose |
|------|---------|
| `server/voice_changer/RVC/RVCr2.py` | RVC v2 model entry point |
| `server/voice_changer/RVC/pipeline/Pipeline.py` | Audio processing pipeline |
| `server/voice_changer/RVC/pipeline/PipelineGenerator.py` | Pipeline factory (with Mac CPU fallback) |
| `server/voice_changer/RVC/embedder/FairseqHubert.py` | HuBERT feature extraction |
| `server/voice_changer/RVC/inferencer/RVCInferencerv2Nono.py` | No-f0 model inference |
| `server/voice_changer/RVC/deviceManager/DeviceManager.py` | CUDA/MPS/CPU device selection |

### REST API
| File | Purpose |
|------|---------|
| `server/restapi/MMVC_Rest.py` | FastAPI app factory with static file mounting |
| `server/restapi/MMVC_Rest_VoiceChanger.py` | `/test` endpoint for voice conversion |
| `server/restapi/MMVC_Rest_Fileuploader.py` | File upload, model loading, settings |

### Client
| File | Purpose |
|------|---------|
| `client/lib/src/VoiceChangerClient.ts` | Main client class |
| `client/lib/src/client/VoiceChangerWorkletNode.ts` | Real-time audio processing |
| `client/demo/src/000_index.tsx` | Demo app entry point |

---

## API Endpoints

### Voice Conversion
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/test` | POST | Convert audio frame (base64 encoded) |

### File Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload_file` | POST | Upload model file chunks |
| `/concat_uploaded_file` | POST | Concatenate uploaded chunks |
| `/load_model` | POST | Load model into slot |

### Settings & Info
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/info` | GET | Get current system state |
| `/performance` | GET | Get performance metrics |
| `/update_settings` | POST | Update voice changer settings |

### Model Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/merge_model` | POST | Merge multiple models |
| `/update_model_info` | POST | Update model metadata |
| `/onnx` | GET | Export model to ONNX |

### Socket.IO Events
| Event | Direction | Description |
|-------|-----------|-------------|
| `connect` | Client→Server | Client connects |
| `request_message` | Client→Server | Send audio for conversion |
| `response` | Server→Client | Receive converted audio |
| `disconnect` | Client→Server | Client disconnects |

---

## Development Commands

### Server (Python)

#### Linux/Windows Setup
```bash
# Setup environment (requires Anaconda)
conda create -n vcclient-dev python=3.10
conda activate vcclient-dev

# Install dependencies
cd server
pip install -r requirements.txt

# Run server
python3 MMVCServerSIO.py -p 18888 --https true \
    --content_vec_500 pretrain/checkpoint_best_legacy_500.pt \
    --hubert_base pretrain/hubert_base.pt \
    --model_dir model_dir
```

#### macOS Setup (Apple Silicon)
```bash
# Navigate to server directory
cd server

# Run setup script (creates conda environment)
./setup-mac.sh

# Start the server
./start-mac.sh

# Or with custom port
PORT=8080 ./start-mac.sh
```

### Client (TypeScript/React)
```bash
# Build client library
cd client/lib
npm install
npm run build:dev

# Run demo app with dev server
cd client/demo
npm install
npm run build:dev
npm start
```

### Docker
```bash
# Build VCClient image
npm run build:docker:vcclient

# Run with docker-compose
docker-compose up
```

---

## Type Definitions

```python
# server/const.py

VoiceChangerType = Literal[
    "MMVCv13", "MMVCv15", "so-vits-svc-40", "DDSP-SVC",
    "RVC", "Diffusion-SVC", "Beatrice", "LLVC", "EasyVC"
]

PitchExtractorType = Literal[
    "harvest", "dio", "crepe", "crepe_full", "crepe_tiny",
    "rmvpe", "rmvpe_onnx", "fcpe"
]

EmbedderType = Literal[
    "hubert_base", "contentvec", "whisper"
]
```

---

## Important Patterns

### 1. Protocol-Based Model Interface
Models implement `VoiceChangerModel` Protocol (not inheritance):
```python
# server/voice_changer/utils/VoiceChangerModel.py
class VoiceChangerModel(Protocol):
    def loadModel(settings: LoadModelParams) -> None
    def inference(data: Any, settings: InferenceParams) -> Any
    def generate_input(data: Any) -> Tuple[Any, int, Any]
    def get_processing_sampling_rate() -> int
    def update_settings(settings: ServerSettings) -> None
```

### 2. Singleton Managers
- `VoiceChangerManager` - Single instance manages all model operations
- `ModelSlotManager` - Manages model slots for quick switching
- `DeviceManager` - GPU/CPU device selection

### 3. V1 vs V2 Pipeline
- `VoiceChanger` (V1): Resampling in orchestrator
- `VoiceChangerV2`: Resampling moved to model layer (more efficient)

### 4. AudioWorklet Processing
Client uses Web Audio API AudioWorklet for low-latency real-time processing.

### 5. Dual Communication
- REST API: File operations, configuration
- Socket.IO: Real-time audio streaming

### 6. Model Slot System
- Up to 500 model slots for quick switching
- Each slot stores: model files, configuration, speaker info
- Metadata stored as JSON in `model_dir/{slotIndex}/params.json`

---

## RVC Model Types

### f0 vs nono Models
| Type | F0 Support | Pitch Shifting | Use Case |
|------|------------|----------------|----------|
| **f0 models** | Yes | Supports "tran" parameter | Natural voice conversion with pitch control |
| **nono models** | No | Not supported | Timbre conversion only, faster |

### Inferencer Selection
- `RVCInferencerv2` - Standard f0 models
- `RVCInferencerv2Nono` - No-f0 models
- `OnnxRVCInferencer` - ONNX optimized inference

---

## macOS (Apple Silicon) Support

### Overview
Voice-Changer supports macOS with Apple Silicon (M1/M2/M3/M4) using CPU-based inference. MPS (Metal Performance Shaders) is detected but **not used** for RVC inference due to compatibility issues.

### Setup Files
| File | Purpose |
|------|---------|
| `server/requirements-mac.txt` | Mac-specific dependencies |
| `server/setup-mac.sh` | Conda environment setup script |
| `server/start-mac.sh` | Server startup script |
| `README_mac.md` | Mac-specific documentation |

### Key Differences from Linux/Windows
| Component | Linux/Windows | macOS |
|-----------|---------------|-------|
| GPU | NVIDIA CUDA | CPU (MPS detected but not used) |
| ONNX Runtime | `onnxruntime-gpu` | `onnxruntime` (CPU) |
| Half Precision | FP16 on supported GPUs | FP32 only |
| Fairseq | Standard pip install | Custom fork required |

### macOS Dependencies
```txt
# requirements-mac.txt key differences:
onnxruntime==1.16.0              # CPU version (not onnxruntime-gpu)
fairseq @ git+https://github.com/brandonkovacs/fairseq.git  # Mac-compatible fork
pyworld==0.3.5                   # Pitch extraction
```

### Quick Start (macOS)
```bash
# 1. Navigate to server directory
cd server

# 2. Run setup script (creates conda environment 'vcclient')
./setup-mac.sh

# 3. Start the server
./start-mac.sh

# Web UI available at: https://localhost:18888
```

### Environment Variables
```bash
# Set in start-mac.sh
export PYTORCH_ENABLE_MPS_FALLBACK=1  # Required for unsupported MPS ops
```

---

## macOS MPS Compatibility Issues (RESOLVED)

### Problem
RVC inference on MPS (Metal Performance Shaders) produced garbage audio - a repeating "ahhhh" sound instead of proper voice conversion. The same models worked correctly on Linux with CUDA.

### Root Cause
The MPS backend produces incorrect numerical results for certain operations used in the RVC pipeline:
1. Fairseq HuBERT model operations
2. RVC synthesizer model operations
3. Some PyTorch operations fallback silently with incorrect results

### Solution
Force the entire RVC pipeline to run on **CPU** when MPS is detected. This is slower but produces correct audio output.

### Files Modified for macOS Support

#### 1. `server/voice_changer/RVC/pipeline/PipelineGenerator.py`
```python
def createPipeline(params, modelSlot, gpu, f0Detector):
    dev = DeviceManager.get_instance().getDevice(gpu)
    half = DeviceManager.get_instance().halfPrecisionAvailable(gpu)

    # IMPORTANT: MPS backend produces garbage audio for RVC models.
    # Force CPU for the entire pipeline on Mac until MPS issues are resolved.
    if dev.type == "mps":
        print("[PipelineGenerator] MPS detected - forcing CPU for entire pipeline")
        dev = torch.device("cpu")
        half = False
    # ... rest of pipeline creation
```

#### 2. `server/voice_changer/RVC/pipeline/Pipeline.py`
```python
# Replaced CUDA-specific autocast with device-agnostic version
from contextlib import nullcontext

def _get_autocast_context(self):
    """Get device-appropriate autocast context manager."""
    if self.isHalf and self.device.type == "cuda":
        from torch.cuda.amp import autocast
        return autocast(enabled=True)
    return nullcontext()
```

#### 3. `server/voice_changer/RVC/embedder/FairseqHubert.py`
```python
def loadModel(self, file, dev, isHalf=True):
    # ... model loading ...

    # Force CPU for HuBERT on MPS devices
    if dev.type == "mps":
        print("[FairseqHubert] MPS detected - forcing CPU for HuBERT")
        self._use_cpu_for_inference = True
        model = model.to(torch.device("cpu"))
    else:
        self._use_cpu_for_inference = False
        model = model.to(dev)
```

#### 4. `server/voice_changer/RVC/inferencer/RVCInferencerv2Nono.py`
```python
def loadModel(self, file, gpu):
    dev = DeviceManager.get_instance().getDevice(gpu)

    # Force CPU for RVC inference on MPS devices
    if dev.type == "mps":
        print("[RVCInferencerv2Nono] MPS detected - forcing CPU for RVC inference")
        dev = torch.device("cpu")
        isHalf = False
    # ... rest of model loading
```

#### 5. `server/restapi/MMVC_Rest.py`
```python
# Fixed PyInstaller check for running from source on Mac
if sys.platform.startswith("darwin") and hasattr(sys, "_MEIPASS"):
    # Running from PyInstaller bundle on macOS
    p1 = os.path.dirname(sys._MEIPASS)
```

### Server Startup Log (macOS)
When running on Mac, you should see these messages:
```
Device status:
  CUDA devices: 0
  MPS available: True

[PipelineGenerator] MPS detected - forcing CPU for entire pipeline
[RVCInferencerv2Nono] MPS detected - forcing CPU for RVC inference
[FairseqHubert] MPS detected - forcing CPU for HuBERT
```

---

## macOS Known Limitations

| Limitation | Description | Workaround |
|------------|-------------|------------|
| No MPS acceleration | RVC runs on CPU | None (MPS produces incorrect results) |
| Slower inference | CPU is ~2-5x slower than GPU | Use smaller chunk sizes |
| No VoRAS model | Linux-only inferencer | Use standard RVC inferencer |
| ONNX on CPU | No CoreML provider | Use PyTorch models for better CPU performance |
| No FP16 | CPU uses FP32 only | Slightly higher memory usage |

---

## macOS Troubleshooting

### "MPS available: False"
1. Ensure you're on Apple Silicon (M1/M2/M3/M4), not Intel Mac
2. Update macOS to latest version
3. Reinstall PyTorch: `pip install --force-reinstall torch==2.0.1`

### "Module not found" errors
```bash
# Ensure conda environment is activated
conda activate vcclient
```

### Server crashes with MPS errors
MPS fallback should handle this, but if crashes persist:
```bash
# Force CPU by setting GPU to -1 in the web UI
# Or restart server (CPU is now default on Mac)
```

### Slow performance
1. Use smaller chunk sizes in web UI settings
2. Close other CPU-intensive applications
3. Use ONNX pitch extractors (`rmvpe_onnx`) for better CPU utilization

### Port already in use
```bash
# Find and kill process using port
lsof -i :18888
kill -9 <PID>

# Or use different port
PORT=8080 ./start-mac.sh
```

---

## Pretrained Models

Models are auto-downloaded from HuggingFace on first run. Stored in `server/pretrain/`:

| Model | Size | Purpose |
|-------|------|---------|
| `hubert_base.pt` | ~360MB | Feature extraction (HuBERT) |
| `content_vec_500.onnx` | ~100MB | Feature extraction (ONNX) |
| `rmvpe.pt` | ~140MB | Pitch extraction |
| `rmvpe.onnx` | ~140MB | Pitch extraction (ONNX) |
| `crepe_onnx_full.onnx` | ~80MB | Pitch extraction (Crepe) |
| `nsf_hifigan/` | ~55MB | Neural vocoder |

---

## Common Tasks

### Adding a New Voice Model
1. Create directory in `server/voice_changer/NewModel/`
2. Implement `VoiceChangerModel` Protocol
3. Add model type to `VoiceChangerType` in `const.py`
4. Register in `VoiceChangerManager`
5. Add UI support in `client/demo/`

### Modifying REST Endpoints
- Endpoints in `server/restapi/MMVC_Rest_VoiceChanger.py`
- Update `ServerRestClient.ts` for client changes

### Modifying Audio Processing
- Server-side: `server/voice_changer/VoiceChanger.py` or model's `inference()`
- Client-side: `client/lib/src/client/VoiceChangerWorkletNode.ts`

### Adding UI Components
- Components in `client/demo/src/components/`
- Global state in `client/demo/src/001_globalHooks/`

---

## Testing

Currently minimal test coverage. Run Python tests:
```bash
cd server
pytest
```

---

## Dependencies

### Server (Key Python packages)
| Package | Version | Purpose |
|---------|---------|---------|
| FastAPI | 0.95.1 | Web framework |
| uvicorn | 0.21.1 | ASGI server |
| python-socketio | 5.8.0 | Real-time communication |
| torch | 2.0.1 | Deep learning |
| torchaudio | 2.0.2 | Audio processing |
| onnxruntime | 1.16.0 (Mac) / 1.13.1 (GPU) | Optimized inference |
| librosa | 0.9.1 | Audio processing |
| numpy | 1.23.5 | Numerical computing |
| scipy | 1.10.1 | Scientific computing |
| faiss-cpu | 1.7.3 | Vector similarity search |
| torchcrepe | 0.0.18 | Pitch extraction |
| torchfcpe | 0.0.3 | FCPE pitch extraction |

### Client (Key npm packages)
| Package | Purpose |
|---------|---------|
| React 18.2 | UI framework |
| socket.io-client | Real-time communication |
| amazon-chime-sdk-js | Audio processing utilities |
| onnxruntime-web | Client-side inference |

---

## Notes for AI Agents

### Critical Information
1. **Primary language is Japanese** - README files, comments, and some code use Japanese. English translations in `docs_i18n/`.

2. **RVC is the main model** - Most development focuses on RVC (Retrieval-based Voice Conversion). Other models have varying maintenance levels.

3. **Real-time constraints** - Audio processing must be low-latency. Be careful with changes that add processing time.

4. **macOS uses CPU only** - Despite MPS being detected, all RVC inference runs on CPU due to compatibility issues.

5. **GPU support varies**:
   - Linux/Windows: NVIDIA CUDA (best performance)
   - macOS: CPU only (MPS produces incorrect results)
   - ONNX: CPU on Mac, CUDA on Linux/Windows

6. **Cross-platform** - Code must work on Windows, macOS (Apple Silicon), Linux, and Google Colab.

7. **Model weights not in repo** - Pretrained models downloaded via HuggingFace on first run.

### When Debugging macOS Issues
1. Check if MPS fallback is enabled: `PYTORCH_ENABLE_MPS_FALLBACK=1`
2. Verify CPU is being used (look for "forcing CPU" messages in logs)
3. Check device consistency - all tensors must be on the same device
4. Fairseq requires the custom fork for macOS compatibility

### Performance Expectations
| Platform | Inference Speed | Notes |
|----------|-----------------|-------|
| NVIDIA GPU | Real-time | Best performance |
| Apple Silicon CPU | 2-5x slower | Functional but noticeable latency |
| Intel CPU | 5-10x slower | May not be real-time |
