"""W10A real small-model LoRA/SFT, with observable freeze/save/reload evidence.

Importing downloads nothing. Install luna-genai[adaptation] for training.
Short runs demonstrate mechanics, not improved general instruction following.
"""
from __future__ import annotations
from contextlib import nullcontext
from importlib.resources import files
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np

BASE_ID='HuggingFaceTB/SmolLM-135M'
BASE_REVISION='1d461723eec654e65efdc40cf49301c89c0c92f4'
INSTRUCT_ID='HuggingFaceTB/SmolLM-135M-Instruct'
INSTRUCT_REVISION='b5f4b394abe60fe0e4d666139e57bfce520302a9'
EMBED_ID='sentence-transformers/all-MiniLM-L6-v2'
EMBED_REVISION='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'


def pinned_model_directory(model_id: str, revision: str, *, embedding: bool=False) -> Path:
    """Download ONLY declared files at an immutable revision; reuse full cache.

    No authentication/upload. Sets existing course Hub timeouts. Missing local
    files trigger an official download (two workers); failures raise a clear
    error, never a fake model. First call writes HF cache; subsequent complete
    calls need no online metadata. embedding=True selects BERT vocab.txt.
    """
    from .download import configure_hub
    configure_hub()
    from huggingface_hub import snapshot_download
    names=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json']
    names+=['vocab.txt'] if embedding else ['generation_config.json','merges.txt']
    try:
        p=Path(snapshot_download(model_id,revision=revision,allow_patterns=names,local_files_only=True,token=False))
        if all((p/name).is_file() for name in names): return p
    except OSError:
        pass
    try:
        p=Path(snapshot_download(model_id,revision=revision,allow_patterns=names,token=False,max_workers=2))
    except Exception as exc:
        raise RuntimeError(f'모델 준비 실패: {model_id} ({type(exc).__name__}). 인터넷·디스크를 확인하고 같은 준비 셀을 다시 실행하세요. 받은 캐시는 유지됩니다.') from exc
    if not all((p/name).is_file() for name in names):
        raise RuntimeError('선택한 필수 모델 파일이 완전하지 않습니다.')
    return p


def load_headline_data() -> dict:
    """Packaged 24 self-authored fictional source documents -> split row lists.

    No external data download. 16/4/4 source-document split precedes formatting;
    no source_id or normalized input crosses splits. Returns a new dictionary.
    English input/response; Korean teaching explanations are in the notebook.
    """
    data=json.loads(files('luna_genai').joinpath('datasets/w10a-headlines.json').read_text(encoding='utf-8'))
    result={s:[r for r in data['rows'] if r['split']==s] for s in ('train','validation','test')}
    ids=[r['source_id'] for rows in result.values() for r in rows]
    texts=[' '.join(r['input'].lower().split()) for rows in result.values() for r in rows]
    if len(ids)!=len(set(ids)) or len(texts)!=len(set(texts)):
        raise ValueError('원문 분할 중복이 있습니다.')
    result['provenance']={k:v for k,v in data.items() if k!='rows'}
    return result


def headline_prompt(text: str) -> str:
    """Nonempty English source -> custom BASE prompt, not a native chat template."""
    if not isinstance(text,str) or not text.strip(): raise ValueError('원문이 비었습니다.')
    return f'Return a short headline.\nText: {text.strip()}\nHeadline:'


def response_batch(tokenizer, rows: list[dict], max_length: int=64) -> dict:
    """Build [B,L] response-only labels for an inspectable causal-LM batch.

    Uses separate prompt/response tokenization to make the boundary explicit.
    The same pretokenized rows go to SFTTrainer (skip_prepare_dataset=True).
    Prompt and padding labels=-100, real response EOS retained, no manual
    shift: the model loss aligns once. Truncating away every response raises.
    Does not change model/tokenizer. Returns fresh CPU tensors.
    """
    import torch
    if not rows or type(max_length) is not int or not 8<=max_length<=256:
        raise ValueError('행 목록과 max_length=8..256을 확인하세요.')
    encoded=[]
    for r in rows:
        prompt=tokenizer.encode(headline_prompt(r['input']),add_special_tokens=False)
        response=tokenizer.encode(' '+r['response'].strip(),add_special_tokens=False)+[tokenizer.eos_token_id]
        if not r['response'].strip() or len(prompt)+1>=max_length:
            raise ValueError('응답 정답이 남도록 짧은 원문과 충분한 길이를 사용하세요.')
        # Keep the true EOS even if the completion itself needs truncation.
        response=response[:max_length-len(prompt)-1]+[tokenizer.eos_token_id] if len(prompt)+len(response)>max_length else response
        encoded.append((prompt+response,[-100]*len(prompt)+response))
    width=max(len(ids) for ids,_ in encoded)
    return {'input_ids':torch.tensor([ids+[tokenizer.pad_token_id]*(width-len(ids)) for ids,_ in encoded]),
            'attention_mask':torch.tensor([[1]*len(ids)+[0]*(width-len(ids)) for ids,_ in encoded]),
            'labels':torch.tensor([labels+[-100]*(width-len(labels)) for _,labels in encoded])}


def parameter_digest(model, *, adapters: bool=False) -> str:
    """SHA-256 of ALL selected named parameter bytes, no persistent copies.

    adapters=False selects every non-LoRA parameter, including frozen norms;
    True selects lora_A/B. Quantization metadata buffers are not parameters
    and are outside this digest. Temporary per-tensor CPU byte views may copy.
    No model update, file save or network request.
    """
    import torch
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if ('lora_' in name)!=adapters: continue
        h.update(name.encode()); h.update(str(tuple(p.shape)).encode())
        h.update(p.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class AdapterLab:
    """Pinned 135M base + q_proj/v_proj LoRA, optionally real NF4 base storage.

    Constructor downloads/loads one model and mutates its trainability.
    device: 'cpu' or available 'cuda'; rank4/alpha8/dropout0 defaults. quantized
    False is the required float CPU path. True is a separately validated NF4
    path; CPU computes float32, supported CUDA chooses BF16 else FP16. No
    device_map='auto', Hub uploads, hidden fallback or automatic long training.
    """
    def __init__(self, *, device: str='cpu', rank: int=4, alpha: int=8,
                 max_length: int=64, seed: int=42, quantized: bool=False):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
        from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
        if device not in ('cpu','cuda') or (device=='cuda' and not torch.cuda.is_available()):
            raise ValueError('사용 가능한 cpu/cuda 장치를 선택하세요.')
        if type(rank) is not int or not 1<=rank<=32 or type(alpha) is not int or alpha<1:
            raise ValueError('rank=1..32와 양의 정수 alpha가 필요합니다.')
        if type(max_length) is not int or not 32<=max_length<=256:
            raise ValueError('max_length=32..256이 필요합니다.')
        torch.manual_seed(seed)
        self.device,self.rank,self.alpha,self.seed,self.max_length=device,rank,alpha,seed,max_length
        self.quantized=quantized; self.steps=0; self.history=[]
        self.directory=pinned_model_directory(BASE_ID,BASE_REVISION)
        self.tokenizer=AutoTokenizer.from_pretrained(self.directory,local_files_only=True)
        self.tokenizer.pad_token=self.tokenizer.eos_token
        self.compute_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device=='cuda' else torch.float32
        options={'local_files_only':True,'dtype':torch.float32,'attn_implementation':'eager'}
        if quantized:
            options['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=self.compute_dtype)
            options['device_map']={'':device}
        model=AutoModelForCausalLM.from_pretrained(self.directory,**options)
        if quantized:
            model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
        else: model=model.to(device)
        cfg=LoraConfig(r=rank,lora_alpha=alpha,lora_dropout=0.,target_modules=['q_proj','v_proj'],
                       bias='none',task_type='CAUSAL_LM',use_dora=False)
        self.model=get_peft_model(model,cfg).eval()
        self.model.config.pad_token_id=self.tokenizer.pad_token_id
        self.initial_base_digest=parameter_digest(self.model)
        self.initial_adapter_digest=parameter_digest(self.model,adapters=True)

    def state(self) -> dict:
        """Return actual counts/dtypes/module configuration; no update/download."""
        base=[p for n,p in self.model.named_parameters() if 'lora_' not in n]
        adapters=[p for n,p in self.model.named_parameters() if 'lora_' in n]
        return {'model':BASE_ID,'revision':BASE_REVISION,'device':self.device,'rank':self.rank,
                'alpha':self.alpha,'scale':self.alpha/self.rank,'dropout':0.,'target_modules':['q_proj','v_proj'],
                'trainable_base':sum(p.numel() for p in base if p.requires_grad),
                'trainable_adapter':sum(p.numel() for p in adapters if p.requires_grad),
                'base_parameter_elements':sum(p.numel() for p in base),
                'quantized_4bit_layers':sum(type(m).__name__=='Linear4bit' for m in self.model.modules()),
                'adapter_dtypes':sorted({str(p.dtype) for p in adapters}),
                'compute_dtype':str(self.compute_dtype),'steps':self.steps,
                'mode':'실제 NF4 기본 모델 + LoRA' if self.quantized else '실제 float 기본 모델 + LoRA',
                'count_note':'4bit packed parameter numel은 원래 논리적 모수 수와 다름'}

    def batch(self, rows: list[dict]) -> dict:
        """Response-only tensors [B,L] moved to this device; no parameter update."""
        return {k:v.to(self.device) for k,v in response_batch(self.tokenizer,rows,self.max_length).items()}

    def evaluate(self, rows: list[dict]) -> dict:
        """Token-weighted response NLL/PPL on caller-selected held-out rows."""
        import torch
        from .text_evaluation import perplexity_from_losses
        losses=[];counts=[];self.model.eval()
        with torch.inference_mode():
            for row in rows:
                batch=self.batch([row]); n=int((batch['labels'][:,1:]!=-100).sum())
                losses.append(float(self.model(**batch).loss.cpu()));counts.append(n)
        return {**perplexity_from_losses(losses,counts),'scope':'동일 토크나이저/길이의 응답 토큰만; 일반 전체문장 PPL과 별도'}

    def generate(self, text: str, *, base_only: bool=False, max_new_tokens: int=16) -> dict:
        """Custom prompt -> actual greedy continuation, reusing the loaded model.

        base_only temporarily disables the adapter. It neither loads a second
        model nor trains. Calls must be serialized when toggling adapters.
        """
        import torch
        if type(max_new_tokens) is not int or not 1<=max_new_tokens<=64: raise ValueError('새 토큰 수=1..64')
        batch=self.tokenizer(headline_prompt(text),return_tensors='pt').to(self.device)
        if batch.input_ids.shape[1]>256: raise ValueError('앱 입력을 짧게 줄여 주세요.')
        self.model.eval()
        with (self.model.disable_adapter() if base_only else nullcontext()),torch.inference_mode():
            output=self.model.generate(**batch,max_new_tokens=max_new_tokens,do_sample=False,
                                       pad_token_id=self.tokenizer.pad_token_id)
        new=output[0,batch.input_ids.shape[1]:].cpu().tolist()
        return {'text':self.tokenizer.decode(new,skip_special_tokens=True),'new_token_ids':new,
                'base_only':base_only,'state':self.state(),'prompt':headline_prompt(text)}

    def fit(self, rows: list[dict], *, steps: int=4, learning_rate: float=5e-4, output_dir: str|Path='.build/w10a-sft') -> dict:
        """REAL TRL SFTTrainer, response labels=-100 on prompt/PAD, AdamW steps.

        steps1..16, batch1, packing=False, no checkpoints/report/Hub. Continues
        current adapter with a new optimizer per call. Records first A/B grads,
        ALL non-LoRA parameter hashes and adapter changes. Writes trainer state
        only under supplied local output_dir. No quality-improvement assertion.
        """
        import torch
        from datasets import Dataset
        from transformers import TrainerCallback
        from trl import SFTConfig,SFTTrainer
        if type(steps) is not int or not 1<=steps<=16 or not math.isfinite(learning_rate) or not 0<learning_rate<=.01:
            raise ValueError('steps=1..16, learning_rate=(0,.01]을 확인하세요.')
        # Supply explicit labels, not an unsupported assistant-only template.
        items=[]
        for row in rows:
            batch=response_batch(self.tokenizer,[row],self.max_length)
            items.append({k:v[0].tolist() for k,v in batch.items()})
        ds=Dataset.from_list(items)
        before_base=parameter_digest(self.model)
        before={n:p.detach().cpu().clone() for n,p in self.model.named_parameters() if 'lora_' in n}
        grads=[]
        class Observe(TrainerCallback):
            def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
                a=b=0.
                for n,p in model.named_parameters():
                    if p.grad is not None and 'lora_' in n:
                        value=float(p.grad.float().norm().cpu())
                        if 'lora_A' in n: a+=value
                        if 'lora_B' in n: b+=value
                grads.append({'step':int(state.global_step)+1,'A_gradient_norm_sum':a,'B_gradient_norm_sum':b})
        cfg=SFTConfig(output_dir=str(output_dir),max_steps=steps,per_device_train_batch_size=1,
            learning_rate=learning_rate,max_length=self.max_length,packing=False,gradient_checkpointing=False,
            completion_only_loss=False,assistant_only_loss=False,dataset_kwargs={'skip_prepare_dataset':True},
            use_cpu=self.device=='cpu',bf16=False,fp16=False,report_to='none',save_strategy='no',
            logging_steps=1,logging_strategy='steps',disable_tqdm=True,dataloader_num_workers=0,
            dataloader_pin_memory=False,optim='adamw_torch',weight_decay=0.,seed=self.seed,data_seed=self.seed)
        self.model.config.use_cache=False
        trainer=SFTTrainer(model=self.model,args=cfg,train_dataset=ds,processing_class=self.tokenizer,callbacks=[Observe()])
        # TRL 1.17.1 unconditionally casts quantized-model trainables to BF16.
        # Keep adapters FP32 explicitly, also on devices without native BF16.
        # This happens before Trainer creates its optimizer; the NF4 storage
        # and bitsandbytes compute_dtype are unchanged.
        for name,p in self.model.named_parameters():
            if p.requires_grad and 'lora_' in name: p.data=p.data.to(torch.float32)
        start=time.perf_counter()
        try: result=trainer.train()
        finally: self.model.eval();self.model.config.use_cache=True
        self.steps+=int(trainer.state.global_step)
        history=[dict(r) for r in trainer.state.log_history if 'loss' in r]
        if not history or any(not math.isfinite(r['loss']) for r in history): raise RuntimeError('유한 학습 loss가 없습니다.')
        changes={n:float((p.detach().cpu()-before[n]).norm()) for n,p in self.model.named_parameters() if n in before}
        after_base=parameter_digest(self.model)
        if before_base!=after_base or not any(v>0 for v in changes.values()):
            raise RuntimeError('기본 모델 동결 또는 어댑터 갱신 검사를 통과하지 못했습니다.')
        self.history.extend(history)
        return {'history':history,'gradients':grads,'adapter_l2_changes':changes,
                'base_digest_before':before_base,'base_digest_after':after_base,'base_unchanged':before_base==after_base,
                'seconds':time.perf_counter()-start,'train_metrics':result.metrics,'state':self.state(),
                'loss_contract':'사전 토큰화 labels 사용; prompt/PAD=-100, response/EOS 유지; 모델에서 한 번 shift',
                'adapter_dtype_policy':'TRL 초기화 후 학습 어댑터 FP32 유지; 기본 NF4/compute_dtype와 별도',
                'quality_claim':'없음: 짧은 실제 갱신 관찰'}

    def save_reload_check(self, directory: str|Path, row: dict) -> dict:
        """Save adapter only and compare eval logits after SAME-revision reload.

        Creates an adapter directory and a temporary second base model (~RAM).
        Float path only; quantized reload/merge needs separate validation.
        After comparison discards the second model. No Hub uploads.
        """
        import torch
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        if self.quantized: raise ValueError('이 검사는 float 어댑터 전용입니다. 4bit 재로딩 검증과 구별하세요.')
        path=Path(directory);path.mkdir(parents=True,exist_ok=True)
        self.model.save_pretrained(path)
        batch=self.batch([row]);inputs={k:v for k,v in batch.items() if k!='labels'}
        with torch.inference_mode(): expected=self.model(**inputs).logits.float().cpu()
        clean=AutoModelForCausalLM.from_pretrained(self.directory,local_files_only=True,dtype=torch.float32,attn_implementation='eager').to(self.device)
        loaded=PeftModel.from_pretrained(clean,path,is_trainable=False).eval()
        with torch.inference_mode(): actual=loaded(**inputs).logits.float().cpu()
        delta=float((expected-actual).abs().max())
        if not torch.allclose(expected,actual,atol=2e-5,rtol=2e-5): raise RuntimeError('어댑터 재로딩 logits 불일치')
        merged=loaded.merge_and_unload().eval()
        with torch.inference_mode(): merged_logits=merged(**inputs).logits.float().cpu()
        merge_delta=float((expected-merged_logits).abs().max())
        if not torch.allclose(expected,merged_logits,atol=1e-4,rtol=1e-4): raise RuntimeError('float 병합 logits 허용오차 초과')
        result={'max_abs_logits_error':delta,'base_revision':BASE_REVISION,
                'adapter_bytes':sum(p.stat().st_size for p in path.iterdir() if p.is_file()),'passed':True,
                'merge':{'max_abs_logits_error':merge_delta,'atol':1e-4,'rtol':1e-4,'passed':True,
                         'scope':'재로딩한 별도 float 모델의 merge_and_unload; 수업 모델 유지'}}
        del loaded,clean,merged
        return result

    def merge_check(self, row: dict) -> dict:
        """Float ONLY: save/reload a temporary adapter, merge the second model.

        This model stays untouched (unmerging can add floating roundoff). Uses
        a temporary adapter directory plus a second base model, then removes
        temporary files through TemporaryDirectory. Never persists base weights.
        """
        import tempfile
        if self.quantized: raise ValueError('4bit 병합은 별도 경로입니다. float 검사를 복사하지 마세요.')
        with tempfile.TemporaryDirectory(prefix='w10a-adapter-') as folder:
            result=self.save_reload_check(folder,row)['merge']
        return result
