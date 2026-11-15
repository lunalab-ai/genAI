"""Generate and inspect classroom audio with shared notebook and app callbacks."""

from base64 import b64encode
from functools import lru_cache, wraps
from importlib.resources import files
from pathlib import Path
from tempfile import mkdtemp
import wave

import numpy as np

from .audio_core import stop_rule
from .audio_models import decode_speecht5, synthesize

MUSIC_MODEL = "facebook/musicgen-small"
MUSIC_REVISION = "4c8334b02c6ec4e8664a91979669a501ec497792"
SPEECH_TEXT = "We can hear the model."


def festival_tone(frequency=440, overtone=.25, sampling_rate=16000):
    """Make a three-second, gated two-sine logo with the same envelope for A/B."""
    frequency, overtone = float(frequency), float(overtone)
    if not 80 <= frequency <= 1800 or not 0 <= overtone <= 1:
        raise ValueError("Use 80–1800 Hz and an overtone weight of 0–1.")
    time = np.arange(3 * sampling_rate) / sampling_rate
    phase = 2 * np.pi * frequency * time
    envelope = np.where(time % 1 < .68, np.sin(np.pi * np.minimum(time % 1 / .68, 1)) ** 2, 0)
    signal = (np.sin(phase) + overtone * np.sin(2 * phase)) / (1 + overtone)
    return (0.60 * envelope * signal).astype(np.float32)


def classroom_stft(samples, rate, window=512, hop=128):
    """Center short Hann windows, then compute one-sided power at each frame."""
    if not 0 < hop <= window or window < 4 or len(samples) < window:
        raise ValueError("A window must fit the signal and cover its hop.")
    padded = np.pad(np.asarray(samples, dtype=np.float64), window // 2)
    frames = np.lib.stride_tricks.sliding_window_view(padded, window)[::hop]
    hann = np.hanning(window)
    spectrum = np.fft.rfft(frames * hann, axis=1) * (2 / hann.sum())
    power = np.abs(spectrum) ** 2
    times = np.arange(len(frames)) * hop / rate
    frequencies = np.fft.rfftfreq(window, d=1 / rate)
    return times, frequencies, power.T


def pcm16(samples):
    samples = np.asarray(samples)
    if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
        raise ValueError("Expected a finite, nonempty mono waveform.")
    return np.rint(np.clip(samples, -1, 1) * 32767).astype("<i2")


def write_wav(path, samples, rate):
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, int(rate), 0, "NONE", "not compressed"))
        output.writeframes(pcm16(samples).tobytes())
    return str(path)


def read_wav(path):
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("The lesson audio must be mono PCM16.")
        samples = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
        return samples.astype(np.float32) / 32767, source.getframerate()


def lesson_assets():
    return files("luna_genai").joinpath("assets/w12a")


def speech_clip(text=SPEECH_TEXT, threshold=.5, seed=7, *,
                decoder_fn=decode_speecht5, stop_fn=stop_rule):
    speaker = files("luna_genai").joinpath("assets/w11b/speaker.npy")
    with speaker.open("rb") as source:
        embedding = np.load(source, allow_pickle=False)
    return synthesize(text, embedding, threshold, seed, decoder_fn=decoder_fn, stop_fn=stop_fn)


def music_prompt(instrument):
    return f"A bright {instrument} melody at a steady tempo"


@lru_cache(maxsize=1)
def load_music():
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MUSIC_MODEL, revision=MUSIC_REVISION)
    model = MusicgenForConditionalGeneration.from_pretrained(MUSIC_MODEL, revision=MUSIC_REVISION)
    return processor, model.eval()


def music_clip(instrument, seed=7, max_tokens=100, *, prompt_fn=music_prompt):
    import torch

    if not 32 <= max_tokens <= 256:
        raise ValueError("Use a token budget of 32–256.")
    processor, model = load_music()
    prompt = prompt_fn(instrument)
    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    torch.manual_seed(seed)
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=True, guidance_scale=3,
                                   max_new_tokens=max_tokens)
    samples = generated[0, 0].cpu().numpy()
    if not np.all(np.isfinite(samples)) or np.max(np.abs(samples)) < 1e-4:
        raise RuntimeError("MusicGen did not return playable audio.")
    return samples, model.config.audio_encoder.sampling_rate, prompt


def sound_figure(clips, window=512, hop=128, *, analysis_fn=classroom_stft):
    """Show comparable waveform, STFT and Mel panels for each (label, y, rate)."""
    import librosa
    from matplotlib.figure import Figure

    end = max(len(samples) / rate for _, samples, rate in clips)
    common_hz = min(5000, *(rate / 2 for _, _, rate in clips))
    figure = Figure(figsize=(11, 2.65 * len(clips)), layout="constrained")
    axes = figure.subplots(len(clips), 3, squeeze=False)
    for (label, samples, rate), (waveform, stft, mel) in zip(clips, axes, strict=True):
        times, hz, power = analysis_fn(samples, rate, int(window), int(hop))
        spectrum = 10 * np.log10(np.maximum(power, 1e-8))
        filterbank = librosa.filters.mel(sr=rate, n_fft=window, n_mels=48,
                                         fmax=common_hz, norm=None)
        mel_power = 10 * np.log10(np.maximum(filterbank @ power, 1e-8))
        waveform.plot(np.arange(len(samples)) / rate, samples, color="#176f78", lw=.75)
        waveform.set(title=label, ylabel="Amplitude", ylim=(-1, 1))
        heat = stft.pcolormesh(times, hz, spectrum, cmap="magma", shading="auto",
                              vmin=-75, vmax=0)
        stft.set(title="STFT power", ylim=(0, common_hz), ylabel="Frequency (Hz)")
        mel.pcolormesh(times, np.arange(len(filterbank)), mel_power, cmap="magma",
                       shading="auto", vmin=-75, vmax=0)
        mel.set(title="Log-Mel power", ylabel="Mel band (0–47)")
        for axis in (waveform, stft, mel):
            axis.set(xlim=(0, end), xlabel="Time (s)")
    figure.colorbar(heat, ax=axes[:, 1:].ravel().tolist(), label="dB (fixed reference)")
    return figure


def build_studio(*, sound_fn=festival_tone, analysis_fn=classroom_stft,
                 decoder_fn=decode_speecht5, stop_fn=stop_rule, prompt_fn=music_prompt):
    import gradio as gr

    def inspect_clip(samples, rate, window, origin):
        path = Path(mkdtemp(prefix="w12a-studio-")) / "w12a-clip.wav"
        write_wav(path, samples, rate)
        played, rate = read_wav(path)
        figure = sound_figure([("Output", played, rate)], int(window),
                              analysis_fn=analysis_fn)
        return str(path), figure, str(path), f"{origin}  ·  {len(played) / rate:.2f} s  ·  {rate} Hz"

    def effect(frequency, overtone, window):
        return inspect_clip(sound_fn(frequency, overtone), 16000, window, "직접 합성")

    def speech(threshold, window):
        result = speech_clip(threshold=threshold, decoder_fn=decoder_fn, stop_fn=stop_fn)
        return inspect_clip(result["waveform"], result["sampling_rate"], window,
                            "지금 생성 · SpeechT5")

    def music(instrument, window):
        samples, rate, prompt = music_clip(instrument, prompt_fn=prompt_fn)
        return inspect_clip(samples, rate, window, f"지금 생성 · MusicGen · {prompt}")

    def with_player(callback):
        @wraps(callback)
        def render(*arguments):
            wav_path, figure, download, details = callback(*arguments)
            payload = b64encode(Path(wav_path).read_bytes()).decode("ascii")
            player = (
                '<p><strong>결과 듣기</strong></p>'
                f'<audio controls preload="metadata" aria-label="결과 듣기" src="data:audio/wav;base64,{payload}"></audio>'
            )
            return player, figure, download, details
        return render

    def outputs():
        return [gr.HTML(), gr.Plot(label="같은 WAV의 세 표현"),
                gr.File(label="WAV 저장"), gr.Textbox(label="생성 방식과 길이")]

    with gr.Blocks(title="축제 사운드 실험실") as app:
        gr.Markdown("# 축제 사운드 실험실\n한 조건을 바꾸고 새 소리와 그림을 비교하세요.")
        with gr.Tab("효과음"):
            frequency = gr.Slider(120, 1400, value=440, step=1, label="기본 주파수 (Hz)")
            harmonic = gr.Slider(0, 1, value=.25, step=.05, label="두 번째 배음")
            window = gr.Dropdown([256, 512, 1024], value=512, label="분석 창 (샘플)")
            gr.Button("새 효과음 만들기", variant="primary").click(
                with_player(effect), [frequency, harmonic, window], outputs(), api_name="effect")
        with gr.Tab("안내 음성"):
            threshold = gr.Slider(.2, 1.5, value=.5, step=.05, label="생성 종료 임계값")
            window = gr.Dropdown([256, 512, 1024], value=512, label="분석 창 (샘플)")
            gr.Button("안내 음성 지금 생성").click(
                with_player(speech), [threshold, window], outputs(), api_name="speech")
        with gr.Tab("음악"):
            instrument = gr.Dropdown(["piano", "acoustic guitar"], value="piano", label="악기")
            window = gr.Dropdown([512, 1024], value=1024, label="분석 창 (샘플)")
            gr.Button("MusicGen으로 지금 생성").click(
                with_player(music), [instrument, window], outputs(), api_name="music")
    app.w12a_callbacks = {"effect": effect, "speech": speech, "music": music}
    return app
