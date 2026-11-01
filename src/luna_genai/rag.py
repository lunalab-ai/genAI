"""W10A: real MiniLM embeddings -> exact cosine index -> frozen small LLM.

Eight packaged fictional documents, no API key, external vector service or
fine-tuning. Never substitute lexical search or hardcoded model answers.
"""
from __future__ import annotations
from importlib.resources import files
import hashlib
import json
import numpy as np
from .peft_lab import (pinned_model_directory,INSTRUCT_ID,INSTRUCT_REVISION,
                       EMBED_ID,EMBED_REVISION,parameter_digest)


def load_documents() -> list[dict]:
    """Fresh 8 self-authored fictional English lab docs, packaged in the wheel.

    id/title/text/ko strings. CC0 teaching data, NOT actual university policies.
    No Internet or teacher file path. Model weights download separately.
    """
    return json.loads(files('luna_genai').joinpath('datasets/w10a-documents.json').read_text(encoding='utf-8'))['documents']


class TextEmbedder:
    """Pinned all-MiniLM-L6-v2, mask-aware mean pool + unit normalization.

    Constructor downloads ~91MB weights and loads frozen model. encode(strings)
    returns new finite float32 [N,384] unit vectors. CPU, max_length128 default;
    truncation is explicit. No training, lexical fallback or remote inference.
    """
    def __init__(self, *, max_length: int=128):
        from transformers import AutoModel,AutoTokenizer
        if type(max_length) is not int or not 8<=max_length<=256: raise ValueError('max_length=8..256')
        self.max_length=max_length
        path=pinned_model_directory(EMBED_ID,EMBED_REVISION,embedding=True)
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.model=AutoModel.from_pretrained(path,local_files_only=True).eval()
        self.model.requires_grad_(False)

    def encode(self, texts: list[str]) -> np.ndarray:
        """English strings -> [N,384] normalized dense vectors; no cache/update."""
        import torch
        if not texts or any(not isinstance(t,str) or not t.strip() for t in texts): raise ValueError('텍스트 목록이 비었습니다.')
        result=[]
        with torch.inference_mode():
            for start in range(0,len(texts),8):
                batch=self.tokenizer(texts[start:start+8],padding=True,truncation=True,max_length=self.max_length,return_tensors='pt')
                hidden=self.model(**batch).last_hidden_state
                mask=batch['attention_mask'].unsqueeze(-1).to(hidden.dtype)
                pooled=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
                pooled=torch.nn.functional.normalize(pooled,p=2,dim=1)
                result.append(pooled.float().cpu().numpy())
        matrix=np.concatenate(result)
        if matrix.shape!=(len(texts),384) or not np.isfinite(matrix).all(): raise RuntimeError('실제 임베딩 shape/유한값 오류')
        return matrix


class DenseIndex:
    """Small exact cosine index; injected encode callable enables toy tests.

    documents: nonempty unique id/title/text dictionaries. Constructor embeds
    all docs. search returns NEW id/title/text/score rows, not generated answers.
    replace_document rebuilds embeddings and content hash; no stale vector reuse.
    documents property returns copies; internal vectors are read-only.
    """
    def __init__(self, documents: list[dict], encode):
        self.encode=encode
        self._replace(documents)

    def _replace(self, documents):
        if not documents or any(not all(isinstance(d.get(k),str) and d[k].strip() for k in ('id','title','text')) for d in documents):
            raise ValueError('id/title/text가 있는 문서 목록이 필요합니다.')
        if len({d['id'] for d in documents})!=len(documents): raise ValueError('문서ID 중복')
        docs=[dict(d) for d in documents]
        vectors=np.asarray(self.encode([d['title']+'. '+d['text'] for d in docs]),dtype=float)
        if vectors.ndim!=2 or vectors.shape[0]!=len(docs) or not np.isfinite(vectors).all(): raise ValueError('임베딩 행렬 오류')
        norms=np.linalg.norm(vectors,axis=1,keepdims=True)
        if (norms==0).any(): raise ValueError('영벡터 cosine은 정의하지 않습니다.')
        self.vectors=vectors/norms;self.vectors.setflags(write=False)
        self._documents=docs
        self.revision=hashlib.sha256(json.dumps(docs,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

    @property
    def documents(self) -> list[dict]:
        """Copies: editing the returned list never silently mutates the index."""
        return [dict(d) for d in self._documents]

    def search(self, query: str, k: int=2) -> list[dict]:
        """Query text,k1..N -> score-descending exact normalized cosine rows.

        score is similarity, not probability/confidence of answer correctness.
        Stable document order breaks exact ties. No generation or weight update.
        """
        if not isinstance(query,str) or not query.strip() or type(k) is not int or not 1<=k<=len(self._documents):
            raise ValueError('질문과 k=1..문서 수를 확인하세요.')
        q=np.asarray(self.encode([query]),dtype=float)
        if q.shape!=(1,self.vectors.shape[1]) or not np.isfinite(q).all() or np.linalg.norm(q)==0:
            raise ValueError('질문 임베딩 오류')
        scores=self.vectors@(q[0]/np.linalg.norm(q[0]))
        order=np.argsort(-scores,kind='stable')[:k]
        return [{**self._documents[int(i)],'score':float(scores[i]),'index_revision':self.revision} for i in order]

    def replace_document(self, document_id: str, text: str) -> str:
        """Replace ONE document, re-embed all, return new content revision.

        Unknown ID or empty text raises before changing state. Original caller
        lists remain unchanged. No generator training/parameter update.
        """
        if not isinstance(text,str) or not text.strip(): raise ValueError('새 문서가 비었습니다.')
        if document_id not in {d['id'] for d in self._documents}: raise ValueError('없는 문서ID')
        docs=self.documents
        for d in docs:
            if d['id']==document_id:d['text']=text.strip()
        self._replace(docs)
        return self.revision


class RAGLab:
    """Independent B runtime: real dense index and frozen Instruct generator.

    Constructor prepares both models; does not need A adapters/files. CPU only
    by default. English model/docs with Korean explanations. No train method.
    A 135M generator can ignore good evidence; output failures stay observable.
    """
    def __init__(self, *, documents: list[dict]|None=None, seed: int=42, max_input_tokens: int=384):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        if type(max_input_tokens) is not int or not 128<=max_input_tokens<=1024: raise ValueError('입력 예산128..1024')
        torch.manual_seed(seed);self.seed=seed;self.max_input_tokens=max_input_tokens
        self.embedder=TextEmbedder()
        self.index=DenseIndex(load_documents() if documents is None else documents,self.embedder.encode)
        path=pinned_model_directory(INSTRUCT_ID,INSTRUCT_REVISION)
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        if not self.tokenizer.chat_template: raise RuntimeError('Instruct 모델 chat template 없음')
        self.model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval()
        self.model.requires_grad_(False)
        self.tokenizer.pad_token=self.tokenizer.eos_token
        self.initial_generator_digest=parameter_digest(self.model)

    def state(self) -> dict:
        """Actual immutable revisions/index state, no model update/download."""
        return {'generator':INSTRUCT_ID,'generator_revision':INSTRUCT_REVISION,'embedding':EMBED_ID,
                'embedding_revision':EMBED_REVISION,'embedding_shape':list(self.index.vectors.shape),
                'index_revision':self.index.revision,'documents':len(self.index.documents),
                'generator_trainable':sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'generator_training_steps':0,'max_input_tokens':self.max_input_tokens,'device':'cpu',
                'mode':'실제 임베딩·정확 cosine 검색·실제 LLM 생성; 문서는 가상'}

    def _messages(self, query, context):
        return [{'role':'system','content':'Use only the supplied fictional lab documents. If they do not answer the question, say I do not know. Answer briefly and cite the document ID.'},
                {'role':'user','content':f'Documents:\n{context or "(No documents supplied.)"}\n\nQuestion: {query}'}]

    def run(self, query: str, *, k: int=2, mode: str='retrieved', max_new_tokens: int=40) -> dict:
        """Query -> real retrieval, budgeted chat context, actual greedy answer.

        mode: retrieved / none / low_rank (forced lowest-score doc, relevance
        still judged by humans). k1..N; output cap1..64. If context exceeds
        token budget, drop lowest-ranked whole documents; never silently cut
        the question/control tokens. Records all retrieved vs actually used IDs.
        No retraining/download on calls; serialize calls with document updates.
        """
        import torch
        if mode not in ('retrieved','none','low_rank') or type(max_new_tokens) is not int or not 1<=max_new_tokens<=64:
            raise ValueError('모드 또는 새 토큰 수 오류')
        retrieved=self.index.search(query,k)
        selected=(retrieved if mode=='retrieved' else [] if mode=='none' else self.index.search(query,len(self.index.documents))[-1:])
        used=[dict(d) for d in selected];dropped=[]
        while True:
            context='\n'.join(f"[{d['id']}] {d['title']}: {d['text']}" for d in used)
            messages=self._messages(query,context)
            ids=self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)
            if len(ids)<=self.max_input_tokens:break
            if not used: raise ValueError('질문만으로도 입력 토큰 예산을 넘습니다. 질문을 줄여 주세요.')
            dropped.append(used.pop()['id'])
        tensor=torch.tensor([ids]);attention=torch.ones_like(tensor)
        with torch.inference_mode():
            output=self.model.generate(input_ids=tensor,attention_mask=attention,max_new_tokens=max_new_tokens,
                                       do_sample=False,pad_token_id=self.tokenizer.pad_token_id)
        new=output[0,len(ids):].cpu().tolist()
        answer=self.tokenizer.decode(new,skip_special_tokens=True)
        return {'query':query,'mode':mode,'k':k,'retrieved':retrieved,'used_document_ids':[d['id'] for d in used],
                'dropped_document_ids':dropped,'context':context,'messages':messages,'input_tokens':len(ids),
                'answer':answer,'new_token_ids':new,'generation':{'do_sample':False,'max_new_tokens':max_new_tokens},
                'state':self.state(),'evaluation':'검색 hit@k와 근거 일치·지시 이행·형식은 각각 사람이 판정'}

    def replace_document(self, document_id: str, text: str) -> dict:
        """Re-embed changed document set; returns index revisions, trains no LLM."""
        before=self.index.revision
        after=self.index.replace_document(document_id,text)
        return {'before':before,'after':after,'generator_training_steps':0,'reembedded':len(self.index.documents)}
