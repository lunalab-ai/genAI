"""W11B: inspectable audio representations, CTC decoding and evaluation.

No weights, downloads, training or Gradio launch occur on import. Worked
examples are distinct from the private exercise solutions. The notebook
executes definitions extracted from this file and passes them to the models.
"""
from __future__ import annotations

from importlib.resources import files
import json
import math
import re
import unicodedata
import numpy as np


def mono_float(signal):
    """PCM/float (N,) 또는 (N,C)를 유한 float32 mono (N,)로 변환한다.

    soundfile/Gradio의 마지막 축이 채널이다. 정수 PCM만 dtype 범위로 나누며
    float의 음량을 매번 최대값으로 정규화하지 않는다. 빈 배열/NaN은 ValueError.
    디스크·네트워크 부작용 없음. 예: int16 [0,16384] -> [0,.5].
    """
    x = np.asarray(signal)
    if x.ndim not in (1, 2) or not x.size or x.dtype.kind not in 'fiu':
        raise ValueError('Audio must be a nonempty (samples,) or (samples,channels) array')
    if x.dtype.kind in 'iu':
        bits = np.iinfo(x.dtype).bits
        midpoint = 2 ** (bits - 1)
        x = (x.astype(np.float32) - (midpoint if x.dtype.kind == 'u' else 0)) / midpoint
    else:
        x = x.astype(np.float32)
    if x.ndim == 2:
        if not 1 <= x.shape[1] <= 8:
            raise ValueError('Last axis must be channels, not time')
        x = x.mean(axis=1)
    if not np.isfinite(x).all():
        raise ValueError('Audio contains NaN or infinity')
    return x


def resample_audio(signal, source_rate, target_rate=16000):
    """실제 polyphase resampling. rate 라벨만 바꾸지 않는다.

    rate: 양의 정수 Hz, target 기본16000. 반환 float32 (N_new,),
    N_new=ceil(N*target/source). 저역통과 처리는 scipy.signal.resample_poly.
    업샘플링은 원본에 없던 고주파 정보를 복구하지 않는다. 외부 I/O 없음.
    """
    from scipy.signal import resample_poly
    if any(isinstance(r, (bool, np.bool_)) or int(r) != r or r <= 0 for r in (source_rate, target_rate)):
        raise ValueError('Rates must be positive integer Hz')
    x = mono_float(signal)
    divisor = math.gcd(int(source_rate), int(target_rate))
    return resample_poly(x, int(target_rate)//divisor, int(source_rate)//divisor).astype(np.float32)


def frame_signal(signal, frame_length=400, hop_length=160, center=False):
    """파형을 (frames, frame_length)로 복사한다. hop은 창의 이동 간격이다.

    center=False: 완전한 창만 사용, frames=1+(N-frame_length)//hop.
    True: 양끝 frame_length//2 reflect padding(너무 짧으면 constant).
    frame_length와 hop은 양의 정수. N이 창보다 짧으면 ValueError.
    """
    x = mono_float(signal)
    if any(isinstance(v, bool) or int(v) != v or v <= 0 for v in (frame_length, hop_length)):
        raise ValueError('Frame and hop lengths must be positive integers')
    frame_length, hop_length = int(frame_length), int(hop_length)
    if center:
        pad = frame_length // 2
        x = np.pad(x, (pad, pad), mode='reflect' if len(x) > pad else 'constant')
    if len(x) < frame_length:
        raise ValueError('Signal is shorter than one complete frame')
    starts = range(0, len(x)-frame_length+1, hop_length)
    return np.stack([x[start:start+frame_length] for start in starts])


def stft_power(signal, n_fft=400, hop_length=160, center=False):
    """Hann 창→rFFT→|X|². 반환 power (1+n_fft//2, frames).

    FFT는 numpy, 창·프레임·축 전환·power는 이 함수 본문에서 계산한다.
    FFT 정규화 없는 power이며 physical SPL/dBFS가 아니다. 외부 I/O 없음.
    """
    frames = frame_signal(signal, n_fft, hop_length, center)
    window = np.hanning(n_fft + 1)[:-1]  # periodic Hann, 같은 n_fft를 사용
    complex_frames = np.fft.rfft(frames * window, axis=1)
    return (np.abs(complex_frames) ** 2).T


def whisper_logmel(signal, sampling_rate=16000, n_mels=80, hop_length=160, chunk_seconds=30):
    """Whisper용 단일 입력을 설명 가능한 계산으로 만든다: (1,Mel,time).

    16kHz mono만 받는다. truncate/right-pad→Hann STFT→Mel 행렬곱→
    log10→dynamic-range clipping→Whisper scaling. 기본은 (1,80,3000).
    Mel 필터 구성만 Transformers 공식 helper를 사용한다. 임의 n_mels/hop
    탐구 결과가 사전학습 모델의 입력 계약과 맞는지는 별도 확인해야 한다.
    """
    from transformers.audio_utils import mel_filter_bank
    if sampling_rate != 16000 or not 1 <= n_mels <= 200 or hop_length < 1 or chunk_seconds < 1:
        raise ValueError('Use 16kHz audio and positive feature settings')
    length = int(sampling_rate * chunk_seconds)
    x = mono_float(signal)[:length]
    x = np.pad(x, (0, length-len(x)))
    power = stft_power(x, n_fft=400, hop_length=hop_length, center=True)[:, :-1]
    filters = mel_filter_bank(201, int(n_mels), 0., 8000., 16000, norm='slaney', mel_scale='slaney')
    mel = filters.T @ power
    logmel = np.log10(np.maximum(mel, 1e-10))
    logmel = np.maximum(logmel, logmel.max()-8.)
    return ((logmel+4.)/4.).astype(np.float32)[None, :, :]


def ctc_decode(token_ids, id_to_token, blank_id=0, word_delimiter='|'):
    """Greedy CTC 추론의 collapse: 연속 동일 ID 병합→blank 제거→단어 기호.

    token_ids: 1차원 정수 ID, id_to_token: dict[int,str]. 문자열 반환.
    blank_id 기본0은 모델의 실제 tokenizer.pad_token_id로 교체해야 한다.
    이것은 CTC 학습의 모든 정렬 경로 합산/beam search 구현이 아니다.
    예: IDs [1,1,0,1], {0:'_',1:'l'} -> 'll'.
    """
    output = []
    previous = None
    for raw in token_ids:
        token = int(raw)
        if token != previous and token != blank_id:
            piece = id_to_token[token]
            output.append(' ' if piece == word_delimiter else piece)
        previous = token  # blank도 이전 상태에 남겨야 반복 글자를 보존한다
    return ''.join(output).strip()


def normalize_text(text):
    """NFC+casefold, 구두점은 공백, 연속 공백은 하나로. 문자열 반환.

    같은 함수를 reference와 hypothesis 양쪽에 적용한다. 숫자/약어를
    의미적으로 확장하지 않는다. 한국어의 어절/문자 분절 기준도 별도 선택한다.
    """
    text = unicodedata.normalize('NFC', str(text)).casefold()
    text = ''.join(' ' if unicodedata.category(c).startswith('P') else c for c in text)
    return re.sub(r'\s+', ' ', text).strip()


def edit_alignment(reference, hypothesis):
    """두 토큰 list의 Levenshtein DP. 반환 distance, counts, alignment, table.

    insertion/deletion/substitution 비용1, match0. 동률은 대치→삭제→삽입
    순서로 결정하며 최적 정렬은 유일하지 않을 수 있다. table (N+1,M+1).
    counts 키 S,D,I,C; alignment는 (operation,reference,hypothesis) 목록.
    """
    a, b = list(reference), list(hypothesis)
    table = np.zeros((len(a)+1, len(b)+1), dtype=int)
    table[:, 0] = np.arange(len(a)+1)
    table[0, :] = np.arange(len(b)+1)
    for i, x in enumerate(a, 1):
        for j, y in enumerate(b, 1):
            table[i, j] = min(table[i-1,j]+1, table[i,j-1]+1, table[i-1,j-1]+(x != y))
    i, j = len(a), len(b)
    counts = dict(S=0, D=0, I=0, C=0)
    alignment = []
    while i or j:
        if i and j and table[i,j] == table[i-1,j-1]+(a[i-1] != b[j-1]):
            op = 'C' if a[i-1] == b[j-1] else 'S'
            alignment.append((op, a[i-1], b[j-1])); i -= 1; j -= 1
        elif i and table[i,j] == table[i-1,j]+1:
            op = 'D'; alignment.append((op, a[i-1], '')); i -= 1
        else:
            op = 'I'; alignment.append((op, '', b[j-1])); j -= 1
        counts[op] += 1
    return dict(distance=int(table[-1,-1]), counts=counts, alignment=alignment[::-1], table=table)


def text_error(reference, hypothesis, unit='word', normalizer=normalize_text):
    """WER/CER 상세. word=공백 단어/어절, char=정규화 후 공백 제외 문자.

    같은 normalizer를 양쪽에 적용한다. reference가 비면 분모가 없으므로
    ValueError. rate는 (S+D+I)/N이며 1을 넘을 수 있다. 퍼센트가 아닌 비율.
    반환 dict의 table은 numpy 배열, 나머지는 JSON으로 바꿀 수 있는 값이다.
    """
    ref, hyp = normalizer(reference), normalizer(hypothesis)
    if unit not in ('word', 'char'):
        raise ValueError('unit must be word or char')
    a, b = (ref.split(), hyp.split()) if unit == 'word' else (list(ref.replace(' ','')), list(hyp.replace(' ','')))
    if not a:
        raise ValueError('Empty reference: WER/CER denominator is undefined')
    result = edit_alignment(a, b)
    return dict(result, rate=result['distance']/len(a), denominator=len(a), reference=ref, hypothesis=hyp, unit=unit)


def residual_quantize(vector, codebooks):
    """소형 RVQ 원리 계산. 각 codebook (K,D)에서 잔차의 최근접 벡터 선택.

    vector (D,), codebooks list. 반환 codes/list, reconstructed(D,),
    residual(D,), mse(float), trace/list. 사전학습 EnCodec 전체 구현/학습은 아니다.
    """
    original = np.asarray(vector, dtype=float)
    if original.ndim != 1 or not original.size or not np.isfinite(original).all() or not codebooks:
        raise ValueError('Use a finite vector and at least one codebook')
    residual, reconstructed = original.copy(), np.zeros_like(original)
    codes, trace = [], []
    for stage, book in enumerate(codebooks):
        book = np.asarray(book, dtype=float)
        if book.ndim != 2 or book.shape[1] != len(original) or not len(book) or not np.isfinite(book).all():
            raise ValueError('Each codebook must have shape (entries,features)')
        distances = ((book-residual)**2).sum(axis=1)
        code = int(np.argmin(distances))
        reconstructed += book[code]
        residual -= book[code]
        codes.append(code)
        trace.append(dict(stage=stage, code=code, vector=book[code].tolist(), residual=residual.tolist()))
    return dict(codes=codes, reconstructed=reconstructed, residual=residual,
                mse=float(np.mean((original-reconstructed)**2)), trace=trace)


def stop_rule(score, threshold=.5):
    """현재 SpeechT5의 합산 stop score를 기준값과 비교한다. 확률 자체가 아님.

    score는 reduction_factor개 sigmoid 출력의 합, 기본 r=2이면0..2.
    비교는 >=. 이 함수를 실제 decode loop에 전달하여 수정이 결과에 반영된다.
    """
    return bool(float(score) >= float(threshold))


def asset_directory():
    """패키지에 포함된 공개 가능한 작은 오디오/기록 폴더 반환. 다운로드 없음."""
    return files('luna_genai').joinpath('assets/w11b')


def load_audio_sample():
    """출처가 명시된 LibriSpeech 짧은 음성/고정 화자 조건/manifest를 읽는다.

    반환 dict: waveform float32(N,), sampling_rate int, reference str,
    speaker_embedding float32(512,), manifest dict. 패키지 파일 읽기만 발생.
    개인 업로드나 외부 API 키가 필요하지 않다.
    """
    import soundfile as sf
    directory = asset_directory()
    manifest = json.loads(directory.joinpath('manifest.json').read_text(encoding='utf-8'))
    wave, rate = sf.read(str(directory.joinpath('speech.wav')), dtype='float32')
    speaker = np.load(str(directory.joinpath('speaker.npy')), allow_pickle=False)
    return dict(waveform=mono_float(wave), sampling_rate=int(rate),
                reference=manifest['speech']['reference'], speaker_embedding=speaker, manifest=manifest)

