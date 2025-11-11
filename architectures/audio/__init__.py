"""Audio processing architectures for speech, music, and sound"""

from .whisper import Whisper, WhisperConfig, WhisperEncoder, WhisperDecoder
from .encodec import Encodec, EncodecConfig
from .audiogen import AudioGen, AudioGenConfig
from .musicgen import MusicGen, MusicGenConfig
from .wav2vec import Wav2Vec2, Wav2Vec2Config
from .audio_transformer import AudioTransformer, AudioTransformerConfig

__all__ = [
    'Whisper', 'WhisperConfig', 'WhisperEncoder', 'WhisperDecoder',
    'Encodec', 'EncodecConfig',
    'AudioGen', 'AudioGenConfig',
    'MusicGen', 'MusicGenConfig',
    'Wav2Vec2', 'Wav2Vec2Config',
    'AudioTransformer', 'AudioTransformerConfig'
]
