# Brain - Complete A-to-Z Usage Guide 🧠

This comprehensive guide shows you how to use **every component** of the Brain system, from basic to advanced usage.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Architecture-by-Architecture Guide](#architecture-guide)
4. [Three Usage Modes](#three-modes)
5. [YAML Configuration](#yaml-configuration)
6. [Advanced Features](#advanced-features)
7. [Training & Evaluation](#training-evaluation)
8. [Deployment](#deployment)

---

## 1. Installation {#installation}

```bash
# Clone the repository
git clone https://github.com/JacquesGariepy/Brain.git
cd Brain

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(torch.__version__)"
```

### System Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU)
- 16GB+ RAM recommended

---

## 2. Quick Start {#quick-start}

### Example 1: Visual Question Answering (Automatic)

```python
from core.orchestrator import IntelligentOrchestrator, TaskSpecification, TaskType, ModalityType
import torch

# Initialize orchestrator
orchestrator = IntelligentOrchestrator()

# Define task
task = TaskSpecification(
    task_type=TaskType.VISUAL_QUESTION_ANSWERING,
    modalities=[ModalityType.TEXT, ModalityType.IMAGE],
    input_shape={'image': (3, 224, 224), 'text': (512,)}
)

# Automatic architecture selection
selection = orchestrator.select_architecture(task)
print(f"Selected: {selection.primary_architecture}")
print(f"Reasoning: {selection.reasoning}")

# Run inference
image = torch.randn(1, 3, 224, 224)
question = "What is in this image?"
answer = orchestrator.forward({'image': image, 'text': question}, task)
```

### Example 2: Object Detection

```python
from architectures.computer_vision.yolo import YOLOv8, YOLOv8Config

# Create model
config = YOLOv8Config(
    model_size='m',  # n, s, m, l, x
    num_classes=80,
    image_size=640
)
model = YOLOv8(config)

# Inference
images = torch.randn(2, 3, 640, 640)
outputs = model(images)

# Get predictions with NMS
detections = model.predict(images, conf_threshold=0.25)
for i, det in enumerate(detections):
    print(f"Image {i}: {len(det['boxes'])} objects detected")
```

---

## 3. Architecture-by-Architecture Guide {#architecture-guide}

### 🎨 Multimodal Models

#### CLIP - Contrastive Vision-Language Learning

```python
from architectures.multimodal.clip import CLIP, CLIPConfig

config = CLIPConfig(
    vision_model='ViT-B/16',
    text_model='transformer',
    embed_dim=512
)
model = CLIP(config)

# Zero-shot image classification
images = torch.randn(4, 3, 224, 224)
texts = ["a dog", "a cat", "a bird", "a fish"]

image_features, text_features, logit_scale = model(images, texts)
probs = model.zero_shot_classification(images, texts)
```

#### BLIP-2 - Q-Former with Frozen Encoders

```python
from architectures.multimodal.blip2 import BLIP2, BLIP2Config

config = BLIP2Config(
    vision_encoder='vit_large',
    llm_model='opt-2.7b',
    num_query_tokens=32
)
model = BLIP2(config)

# Image captioning
image = torch.randn(1, 3, 224, 224)
caption = model.generate(image, mode='caption')
print(f"Caption: {caption}")

# Visual question answering
question = "What color is the sky?"
answer = model.generate(image, prompt=question, mode='vqa')
```

#### LLaVA - Large Language and Vision Assistant

```python
from architectures.multimodal.llava import LLaVA, LLaVAConfig

config = LLaVAConfig(
    vision_encoder='clip_vit_large',
    llm_model='vicuna-7b',
    projection_dim=4096
)
model = LLaVA(config)

# Instruction following
image = torch.randn(1, 3, 336, 336)
prompt = "Describe this image in detail."
response = model.generate(image, prompt)
```

#### Flamingo - Few-Shot Visual Reasoning

```python
from architectures.multimodal.flamingo import Flamingo, FlamingoConfig

config = FlamingoConfig(
    vision_encoder='nfnet',
    num_latents=64,
    cross_attention_layers=[1, 4, 7, 10]
)
model = Flamingo(config)

# Few-shot learning
support_images = torch.randn(4, 3, 224, 224)  # 4 examples
support_labels = torch.tensor([0, 1, 0, 1])
query_image = torch.randn(1, 3, 224, 224)

prediction = model.few_shot_predict(
    support_images, support_labels, query_image, k=2
)
```

### 🎵 Audio Models

#### Whisper - Multilingual Speech Recognition

```python
from architectures.audio.whisper import Whisper, WhisperConfig

config = WhisperConfig(model_size='base')
model = Whisper(config)

# Transcribe audio
audio = torch.randn(1, 80, 3000)  # Mel spectrogram
transcription = model.generate(
    audio,
    task='transcribe',
    language='en'
)
print(f"Transcription: {transcription}")

# Translate to English
translation = model.generate(
    audio,
    task='translate',
    language='fr'
)
```

#### Encodec - Neural Audio Codec

```python
from architectures.audio.encodec import Encodec, EncodecConfig

config = EncodecConfig(
    sample_rate=24000,
    num_codebooks=4,
    codebook_size=1024
)
model = Encodec(config)

# Encode audio
audio = torch.randn(1, 1, 24000)  # 1 second
codes, commitment_loss = model.encode(audio)

# Decode
reconstructed = model.decode(codes)
```

#### MusicGen - Text-to-Music Generation

```python
from architectures.audio.musicgen import MusicGen, MusicGenConfig

config = MusicGenConfig(
    num_codebooks=4,
    vocab_size=2048
)
model = MusicGen(config)

# Generate music from text
text_descriptions = ["upbeat electronic dance music"]
music = model.generate(
    text=text_descriptions,
    duration=30.0,  # 30 seconds
    temperature=1.0
)
```

#### Wav2Vec 2.0 - Self-Supervised Speech Learning

```python
from architectures.audio.wav2vec2 import Wav2Vec2, Wav2Vec2Config

config = Wav2Vec2Config(
    mask_time_prob=0.05,
    mask_feature_prob=0.05
)
model = Wav2Vec2(config)

# Self-supervised pre-training
audio = torch.randn(2, 16000)  # Raw waveform
quantized, mask = model.forward_masked(audio)

# Fine-tuning for downstream tasks
features = model(audio)  # Extract features
```

### 👁️ Computer Vision

#### SAM - Segment Anything Model

```python
from architectures.computer_vision.sam import SAM, SAMConfig

config = SAMConfig(
    encoder_type='vit_h',
    image_size=1024
)
model = SAM(config)

# Segment with point prompts
images = torch.randn(1, 3, 1024, 1024)
points = torch.tensor([[[512, 512]]])  # Click location
labels = torch.tensor([[1]])  # Foreground

masks, iou_predictions = model(images, points=points, point_labels=labels)
```

#### YOLOv8 - Real-Time Object Detection

```python
from architectures.computer_vision.yolo import YOLOv8, YOLOv8Config

# All model sizes
for size in ['n', 's', 'm', 'l', 'x']:
    config = YOLOv8Config(model_size=size)
    model = YOLOv8(config)

    images = torch.randn(1, 3, 640, 640)
    detections = model.predict(images)
```

#### DETR - End-to-End Object Detection

```python
from architectures.computer_vision.detr import DETR, DETRConfig

config = DETRConfig(
    backbone='resnet50',
    num_queries=100,
    num_classes=80
)
model = DETR(config)

# No NMS needed!
images = torch.randn(1, 3, 640, 640)
predictions = model.predict(images, threshold=0.7)
```

#### DINOv2 - Self-Supervised Vision Learning

```python
from architectures.computer_vision.dino import DINOv2, DINOv2Config

config = DINOv2Config(
    model_size='base',
    image_size=518
)
model = DINOv2(config)

# Extract features
images = torch.randn(1, 3, 518, 518)
features = model.extract_features(images)

# Use for downstream tasks
```

### 📈 Time Series

#### N-BEATS - Neural Basis Expansion

```python
from architectures.time_series.nbeats import NBEATS, NBEATSConfig

config = NBEATSConfig(
    backcast_length=24,
    forecast_length=12,
    stack_types=['trend', 'seasonality']
)
model = NBEATS(config)

# Forecast
x = torch.randn(32, 24)  # Historical data
forecast, components = model(x, return_components=True)

print(f"Trend: {components['trend'].shape}")
print(f"Seasonality: {components['seasonality'].shape}")
```

#### TFT - Temporal Fusion Transformer

```python
from architectures.time_series.tft import TemporalFusionTransformer, TFTConfig

config = TFTConfig(
    static_input_size=4,
    temporal_observed_size=3,
    temporal_known_size=2,
    encoder_length=24,
    decoder_length=12,
    quantiles=[0.1, 0.5, 0.9]
)
model = TemporalFusionTransformer(config)

# Multi-horizon forecasting with uncertainty
static = torch.randn(8, 4)
historical = torch.randn(8, 24, 4)
future = torch.randn(8, 12, 2)

outputs = model(
    static_inputs=static,
    historical_inputs=historical,
    future_inputs=future,
    return_attention=True
)

# Get median prediction
median = model.predict(static, historical, future, quantile=0.5)
```

#### PatchTST - Patch Time Series Transformer

```python
from architectures.time_series.patchtst import PatchTST, PatchTSTConfig

config = PatchTSTConfig(
    num_variables=7,
    seq_len=336,
    pred_len=96,
    patch_len=16,
    stride=8
)
model = PatchTST(config)

# Efficient long-sequence forecasting
x = torch.randn(32, 7, 336)
forecast = model.predict(x)
```

### 🧠 Meta-Learning

#### MAML - Model-Agnostic Meta-Learning

```python
from architectures.meta_learning.maml import MAML, MAMLConfig

# Any PyTorch model
base_model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

config = MAMLConfig(
    inner_lr=0.01,
    inner_steps=5,
    meta_lr=0.001
)
maml = MAML(base_model, config)

# Few-shot learning
support_x = torch.randn(5, 784)  # 5-shot
support_y = torch.tensor([0, 1, 2, 3, 4])
query_x = torch.randn(10, 784)

# Adapt to new task
adapted_model = maml.adapt(support_x, support_y)
predictions = adapted_model(query_x)
```

---

## 4. Three Usage Modes {#three-modes}

### Mode 1: Unitaire (Standalone)

Each architecture works independently:

```python
# Use CLIP alone
from architectures.multimodal.clip import CLIP, CLIPConfig

clip = CLIP(CLIPConfig())
features = clip(images, texts)
```

### Mode 2: Combinatoire (Manual Combination)

Combine multiple architectures manually:

```python
# Combine CLIP + SAM for open-vocabulary segmentation
from architectures.multimodal.clip import CLIP
from architectures.computer_vision.sam import SAM

clip = CLIP(CLIPConfig())
sam = SAM(SAMConfig())

# CLIP finds objects
image_features = clip.encode_image(image)
text_features = clip.encode_text(["person", "car", "tree"])
similarities = image_features @ text_features.T

# SAM segments detected objects
masks = sam(image, boxes=detected_boxes)
```

### Mode 3: Symbiose (Automatic Orchestration)

Let the system automatically select and combine:

```python
from core.orchestrator import IntelligentOrchestrator

orchestrator = IntelligentOrchestrator()

# Automatically selects best architecture(s)
task = TaskSpecification(
    task_type=TaskType.VISUAL_QUESTION_ANSWERING,
    modalities=[ModalityType.IMAGE, ModalityType.TEXT]
)

selection = orchestrator.select_architecture(task)
# Might combine: CLIP + LLaVA + attention fusion
```

---

## 5. YAML Configuration {#yaml-configuration}

### Configure Any Architecture from YAML

```yaml
# configs/my_model.yaml
model:
  name: YOLOv8
  architecture: computer_vision

  model_size: m
  image_size: 640
  num_classes: 80

training:
  batch_size: 16
  epochs: 300
  learning_rate: 0.01

  optimizer:
    name: SGD
    momentum: 0.937
    weight_decay: 0.0005

device: cuda
```

Load and use:

```python
from config import load_config
from architectures.computer_vision.yolo import YOLOv8, YOLOv8Config

# Load config
config_dict = load_config('configs/my_model.yaml')

# Create model
config = YOLOv8Config(**config_dict['model'])
model = YOLOv8(config)
```

---

## 6. Advanced Features {#advanced-features}

### Universal Explainability

Works with **ANY** model:

```python
from architectures.explainability import UniversalExplainer, ExplanationType

# Any model
model = YOLOv8(config)

# Create explainer
explainer = UniversalExplainer(model)

# Explain predictions
image = torch.randn(1, 3, 640, 640)

# Multiple explanation methods
grad_explanation = explainer.explain(image, method=ExplanationType.GRADIENT)
ig_explanation = explainer.explain(image, method=ExplanationType.INTEGRATED_GRADIENT)
cam_explanation = explainer.explain(image, method=ExplanationType.GRADCAM)

# Visualize
visualization = explainer.visualize_attribution(grad_explanation, image)
```

### Continual Learning

Learn without forgetting:

```python
from architectures.continual_learning import ContinualLearner, ContinualMethod

model = YourModel()

# Choose method: EWC, iCaRL, LwF, GEM, A-GEM
learner = ContinualLearner(model, method=ContinualMethod.EWC)

# Train on Task 1
learner.train_task(task1_dataloader, optimizer, task_id=0)

# Train on Task 2 (without forgetting Task 1!)
learner.train_task(task2_dataloader, optimizer, task_id=1)
```

### Federated Learning

Privacy-preserving distributed training:

```python
from architectures.federated_learning import FederatedTrainer, FedAlgorithm

model = YourModel()

# Create federated trainer
trainer = FederatedTrainer(
    model,
    config=FederatedConfig(
        algorithm=FedAlgorithm.FEDAVG,
        num_rounds=100,
        clients_per_round=10
    )
)

# Add clients
for i, client_data in enumerate(client_datasets):
    trainer.add_client(i, client_data)

# Train federally (data never leaves clients!)
trainer.train(learning_rate=0.01)

# Get global model
global_model = trainer.get_global_model()
```

---

## 7. Training & Evaluation {#training-evaluation}

### Complete Training Pipeline

```python
import torch
from torch.utils.data import DataLoader
from config import load_config

# Load configuration
config = load_config('configs/training.yaml')

# Create model
model = create_model_from_config(config['model'])

# Create data loaders
train_loader = create_dataloader(config['data']['train'])
val_loader = create_dataloader(config['data']['val'])

# Training loop
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config['training']['learning_rate']
)

for epoch in range(config['training']['epochs']):
    # Train
    model.train()
    for batch in train_loader:
        images, targets = batch

        optimizer.zero_grad()
        outputs = model(images)
        loss = compute_loss(outputs, targets)
        loss.backward()
        optimizer.step()

    # Validate
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            images, targets = batch
            outputs = model(images)
            # Compute metrics
```

### Using Model Integrations (Local + Cloud)

```python
from architectures.model_integrations import UnifiedModelInterface, BackendType

interface = UnifiedModelInterface()

# Add local models
interface.add_local_model(
    name="local_llama",
    backend_type=BackendType.OLLAMA,
    model_name="llama2:7b"
)

# Add cloud models
interface.add_cloud_model(
    name="gpt4",
    backend_type=BackendType.OPENAI,
    model_name="gpt-4-turbo",
    api_key="your-key"
)

# Automatic backend selection
from architectures.model_integrations import InferenceRequest

request = InferenceRequest(prompt="Explain quantum computing")
response = await interface.generate(request)

print(f"Backend: {response.backend}")
print(f"Response: {response.text}")
print(f"Cost: ${response.cost_usd:.4f}")
```

---

## 8. Deployment {#deployment}

### Export for Production

```python
# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}}
)

# Export to TorchScript
scripted = torch.jit.script(model)
scripted.save("model.pt")
```

### Serve with API

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load model
model = load_model("checkpoints/best.pth")

class PredictionRequest(BaseModel):
    data: list

@app.post("/predict")
async def predict(request: PredictionRequest):
    inputs = preprocess(request.data)
    outputs = model(inputs)
    return {"predictions": postprocess(outputs)}
```

---

## 🎉 You're Ready!

You now know how to use **every component** of the Brain system:

- ✅ 19 complete SOTA architectures
- ✅ 3 universal frameworks (explainability, continual learning, federated learning)
- ✅ 3 usage modes (unitaire, combinatoire, symbiose)
- ✅ YAML configuration system
- ✅ Training, evaluation, and deployment

### Next Steps

1. **Browse Examples**: Check `examples/` directory for more examples
2. **Read Documentation**: See `IMPLEMENTATION_STATUS.md` for details
3. **Contribute**: We welcome contributions! See contribution guidelines
4. **Ask Questions**: Open an issue on GitHub

---

**Built with ❤️ for advancing AI research and development**

For questions or support, open an issue on GitHub: https://github.com/JacquesGariepy/Brain
