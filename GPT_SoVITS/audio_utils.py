import soundfile as sf
import torch

def load_audio_safe(filepath):
    """
    安全加载音频文件，兼容 torchaudio.load 的返回格式
    返回: (audio_tensor, sample_rate)
    audio_tensor shape: (channels, samples)
    """
    audio, sr = sf.read(filepath)
    audio = torch.from_numpy(audio).float()
    
    # 确保返回格式为 (channels, samples)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)  # (samples,) -> (1, samples)
    else:
        audio = audio.T  # (samples, channels) -> (channels, samples)
    
    return audio, sr