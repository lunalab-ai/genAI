"""W11B actual ASR / text-model continuation / TTS with downloadable evidence."""
from pathlib import Path
import csv
import json
import tempfile
import soundfile as sf
from .audio_core import load_audio_sample, text_error, whisper_logmel, ctc_decode, stop_rule
from .audio_models import recognize, synthesize, decode_speecht5

def route_image_request(strength=.5):
    """Optional actual image branch, reusing W11A's frozen tiny denoiser.

    Fixed teaching image, CPU. This is not a fresh Stable Diffusion/text-to-image run.
    """
    from .diffusion_adaptation import TinyAdaptationLab,load_teaching_images
    from .image_editing_app import tiny_view
    lab=TinyAdaptationLab('frozen')
    return tiny_view(lab.model,load_teaching_images()['train'][:1],float(strength))

def build_audio_app(feature_fn=whisper_logmel,decode_fn=ctc_decode,error_fn=text_error,
                    decoder_fn=decode_speecht5,stop_fn=stop_rule,image_fn=route_image_request):
    import gradio as gr
    sample=load_audio_sample()
    def transcribe(audio,reference):
        if audio is None: rate,wave=sample['sampling_rate'],sample['waveform']
        else: rate,wave=audio
        result=recognize(wave,rate,'whisper',feature_fn=feature_fn)
        score=error_fn(reference,result['text'])
        return result['text'],dict(WER=score['rate'],counts=score['counts'],input_shape=result['input_shape'])
    def speak(text,threshold,seed):
        result=synthesize(text,sample['speaker_embedding'],threshold,int(seed),decoder_fn=decoder_fn,stop_fn=stop_fn)
        folder=Path(tempfile.mkdtemp(prefix='w11b-'))
        wav=folder/'speech.wav'; sf.write(wav,result['waveform'],16000)
        report=folder/'settings.json'; report.write_text(json.dumps({k:v for k,v in result.items() if k not in ('waveform','mel')},ensure_ascii=False,indent=2),encoding='utf-8')
        trace=folder/'stop-trace.csv'
        with trace.open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=['step','frames','score','components','reason']);writer.writeheader();writer.writerows(result['trace'])
        return (16000,result['waveform']),str(wav),str(report),str(trace)
    def respond(audio):
        if audio is None: rate,wave=sample['sampling_rate'],sample['waveform']
        else: rate,wave=audio
        recognized=recognize(wave,rate,'whisper',feature_fn=feature_fn)['text']
        from luna_genai.model import LanguageModelLab
        continuation=LanguageModelLab(device='cpu').generate('Question: '+recognized+'\nAnswer:',strategy='greedy',max_new_tokens=20)
        spoken=continuation.strip()[:160]
        if not spoken: raise RuntimeError('Text model returned empty continuation')
        outputs=speak(spoken,.5,7)
        return recognized,spoken,*outputs
    with gr.Blocks(title='W11B 오디오 원리 실험실') as app:
        gr.Markdown('# 듣기 → 문자 → 새 응답 → 말하기\n영어 소형 모델 실험입니다. 빈 입력은 공개 샘플을 사용합니다. 개인 음성 업로드는 선택이며 공유 앱에는 민감한 음성을 넣지 마세요.')
        audio=gr.Audio(type='numpy',label='입력 음성 (비우면 공개 샘플)')
        reference=gr.Textbox(value=sample['reference'],label='평가 기준 문장')
        asr_button=gr.Button('음성 인식·오류율 계산'); transcript=gr.Textbox(label='인식 결과'); metric=gr.JSON(label='실제 계산')
        asr_button.click(transcribe,[audio,reference],[transcript,metric],api_name='transcribe')
        text=gr.Textbox(value='We can hear the model.',label='합성할 영어 문장')
        threshold=gr.Slider(.1,1.5,value=.5,step=.1,label='합산 종료점수 기준'); seed=gr.Number(value=7,precision=0,label='seed')
        speech_button=gr.Button('Mel 생성 → 파형 듣기'); generated=gr.Audio(label='실제 생성 음성')
        wav=gr.File(label='WAV'); report=gr.File(label='설정 JSON'); trace=gr.File(label='종료점수 CSV')
        speech_button.click(speak,[text,threshold,seed],[generated,wav,report,trace],api_name='synthesize')
        response_button=gr.Button('실제 문자 모델 응답까지 연결'); answer=gr.Textbox(label='문자 모델의 새 continuation (정확한 답변 보장 없음)')
        response_button.click(respond,[audio],[transcript,answer,generated,wav,report,trace],api_name='respond')
        gr.Markdown('## 이전 이미지 기능 연결 · 선택 확장\n고정 수업 이미지에 실제 소형 denoiser를 실행합니다. 새 SD 텍스트 이미지 생성이 아닙니다.')
        strength=gr.Slider(0,1,value=.5,step=.25,label='이미지 편집 strength')
        image_button=gr.Button('이전 이미지 모듈 분기 실행');image=gr.Image(label='실제 소형 편집 이미지')
        image_trace=gr.Dataframe(label='denoising 기록');image_settings=gr.JSON(label='이미지 실행 조건')
        image_button.click(image_fn,[strength],[image,image_trace,image_settings],api_name='image_branch')
    app.w11b_callbacks=dict(transcribe=transcribe,speak=speak,respond=respond,image_branch=image_fn)
    return app
