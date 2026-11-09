"""Pinned, lazy audio models; W11B notebook supplies editable core callbacks.

The single-example SpeechT5 loop follows Hugging Face Transformers
modeling_speecht5._generate_speech (Apache-2.0). See assets/w11b/NOTICE.txt.
"""
from functools import lru_cache
import numpy as np
import torch
from .audio_core import resample_audio, ctc_decode, whisper_logmel, stop_rule

REVISIONS = {
 'openai/whisper-tiny':'169d4a4341b33bc18d8881c4b69c2e104e1cc0af',
 'facebook/wav2vec2-base-960h':'22aad52d435eb6dbaf354bdad9b0da84ce7d6156',
 'microsoft/speecht5_tts':'30fcde30f19b87502b8435427b5f5068e401d5f6',
 'microsoft/speecht5_hifigan':'bb6f429406e86a9992357a972c0698b22043307d',
}

@lru_cache(maxsize=3)
def load_asr(kind='whisper'):
    from transformers import AutoProcessor, AutoModelForCTC, WhisperForConditionalGeneration
    name='openai/whisper-tiny' if kind=='whisper' else 'facebook/wav2vec2-base-960h'
    processor=AutoProcessor.from_pretrained(name,revision=REVISIONS[name])
    cls=WhisperForConditionalGeneration if kind=='whisper' else AutoModelForCTC
    model=cls.from_pretrained(name,revision=REVISIONS[name]).eval()
    return processor,model

def recognize(signal, sampling_rate=16000, kind='whisper', feature_fn=whisper_logmel, decode_fn=ctc_decode, language='en'):
    """Actual CPU inference; edited features/CTC decoder are used, not displayed copies."""
    if kind not in ('whisper','ctc'): raise ValueError('kind must be whisper or ctc')
    wave=resample_audio(signal,sampling_rate,16000)
    processor,model=load_asr(kind)
    with torch.inference_mode():
        if kind=='whisper':
            features=feature_fn(wave,16000)
            if features.shape != (1,80,3000):
                raise ValueError('whisper-tiny requires (1,80,3000); use other settings only for shape experiments')
            ids=model.generate(torch.from_numpy(features),language=language,task='transcribe',max_new_tokens=64)
            text=processor.batch_decode(ids,skip_special_tokens=True)[0]
            return dict(text=text,input_shape=list(features.shape),token_ids=ids[0].tolist(),kind=kind)
        values=processor(wave,sampling_rate=16000,return_tensors='pt').input_values
        logits=model(values).logits
        ids=logits.argmax(-1)[0].tolist()
        vocabulary={v:k for k,v in processor.tokenizer.get_vocab().items()}
        text=decode_fn(ids,vocabulary,processor.tokenizer.pad_token_id,processor.tokenizer.word_delimiter_token)
        return dict(text=text,input_shape=list(values.shape),logit_shape=list(logits.shape),token_ids=ids,kind=kind)

@lru_cache(maxsize=1)
def load_tts():
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    name='microsoft/speecht5_tts'; voc='microsoft/speecht5_hifigan'
    return (SpeechT5Processor.from_pretrained(name,revision=REVISIONS[name]),
            SpeechT5ForTextToSpeech.from_pretrained(name,revision=REVISIONS[name]).eval(),
            SpeechT5HifiGan.from_pretrained(voc,revision=REVISIONS[voc]).eval())

def decode_speecht5(model, input_ids, speaker_embedding, threshold=.5, max_steps=240, stop_fn=stop_rule):
    """One-example actual autoregressive Mel loop, exposing every stop decision.

    Input IDs (1,L), speaker (1,512); output Mel (frames,80), trace/list.
    max_steps is decoder iterations, each emitting reduction_factor frames.
    SpeechT5 prenet uses dropout during inference; reset seed when comparing.
    stop score is SUM of r sigmoid outputs (r=2), not a single probability.
    This educational batch=1 path omits beam/attention/batched stopping.
    """
    if input_ids.ndim!=2 or input_ids.shape[0]!=1: raise ValueError('Single example only')
    if not 1<=int(max_steps)<=240: raise ValueError('max_steps must be 1..240')
    mask=(input_ids!=model.config.pad_token_id).int()
    hidden=model.speecht5.encoder(input_values=input_ids,attention_mask=mask,return_dict=True).last_hidden_state
    r=model.config.reduction_factor; m=model.config.num_mel_bins
    sequence=hidden.new_zeros(1,1,m); past=None; spectra=[]; trace=[]
    for step in range(1,int(max_steps)+1):
        pre=model.speecht5.decoder.prenet(sequence,speaker_embedding)
        out=model.speecht5.decoder.wrapped_decoder(hidden_states=pre[:,-1:],attention_mask=None,
            encoder_hidden_states=hidden,encoder_attention_mask=mask,
            past_key_values=past,use_cache=True,return_dict=True)
        last=out.last_hidden_state.squeeze(1); past=out.past_key_values
        spectrum=model.speech_decoder_postnet.feat_out(last).view(1,r,m)
        spectra.append(spectrum)
        sequence=torch.cat((sequence,spectrum[:,-1:,:]),dim=1)
        probabilities=torch.sigmoid(model.speech_decoder_postnet.prob_out(last))
        score=float(probabilities.sum())
        stopped=bool(stop_fn(score,threshold))
        trace.append(dict(step=step,frames=step*r,score=score,components=probabilities[0].tolist(),
                          reason='threshold' if stopped else ('limit' if step==int(max_steps) else 'continue')))
        if stopped: break
    mel=model.speech_decoder_postnet.postnet(torch.cat(spectra,dim=1))[0]
    return mel,trace

def synthesize(text, speaker_embedding, threshold=.5, seed=7, max_steps=240,
               decoder_fn=decode_speecht5,stop_fn=stop_rule):
    """Real SpeechT5->Mel->HiFiGAN. Downloads pinned weights once; no paid API."""
    if not str(text).strip() or len(str(text))>160: raise ValueError('Use 1..160 characters')
    if not np.isfinite(threshold) or not .01<=float(threshold)<=2: raise ValueError('threshold must be .01..2')
    processor,model,vocoder=load_tts()
    ids=processor(text=str(text),return_tensors='pt').input_ids
    speaker=torch.as_tensor(np.asarray(speaker_embedding),dtype=torch.float32).reshape(1,512)
    torch.manual_seed(int(seed))
    with torch.inference_mode():
        mel,trace=decoder_fn(model,ids,speaker,threshold,max_steps,stop_fn)
        wave=vocoder(mel).cpu().numpy()
    if not np.isfinite(wave).all() or not wave.size: raise RuntimeError('Invalid generated audio')
    return dict(waveform=wave,sampling_rate=16000,mel=mel.cpu().numpy(),trace=trace,
                text=str(text),seed=int(seed),threshold=float(threshold),mode='current-real-inference')
