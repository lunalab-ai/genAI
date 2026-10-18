"""Inspectable CLIP computations using the real, fixed W05A image collection."""
from __future__ import annotations
import pandas as pd
import torch
from torch.nn import functional as F
from .semantic_search import CLIPSearch,unit_rows,search_view


def contrastive_table(image_vectors:torch.Tensor,text_vectors:torch.Tensor,*,scale:float=10.)->dict:
    """Compute paired N-by-N cosine/logits/probabilities and symmetric CE loss.

    Finite, nonzero [N,d] matrices; matching diagonal pairs, N>=2. scale=10 is
    an explicit toy setting; use model.logit_scale.exp() for a real CLIP batch.
    Normalizes rows; returns cosine, logits, image_to_text, text_to_image, loss.
    text_to_image rows are texts (transpose logits). Preserves autograd; does
    not train. Example: contrastive_table(torch.eye(3),torch.eye(3),scale=2).
    """
    if image_vectors.shape!=text_vectors.shape or image_vectors.ndim!=2 or len(image_vectors)<2:
        raise ValueError('Matching [N,d] matrices with N>=2 required')
    if not 0<float(scale)<1000: raise ValueError('scale must lie in (0,1000)')
    cosine=unit_rows(image_vectors)@unit_rows(text_vectors).T
    logits=cosine*scale; labels=torch.arange(len(cosine),device=cosine.device)
    loss=(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))/2
    return dict(cosine=cosine,logits=logits,image_to_text=logits.softmax(dim=1),
                text_to_image=logits.T.softmax(dim=1),loss=loss)


def compare_search(query_a:str,query_b:str,k:int|float,lab:CLIPSearch)->tuple:
    """Two independent real searches -> two galleries, combined table, message.

    Inputs: two nonempty strings, integral k, already indexed CLIPSearch.
    Returns (gallery_a,gallery_b,DataFrame,status). No training/download occurs;
    table has query,rank,id,cosine,category,source. Gallery keeps attribution.
    Example: compare_search('a cat in the snow','a cat indoors',3,lab).
    """
    ga,ta,_=search_view(query_a,k,lab); gb,tb,_=search_view(query_b,k,lab)
    ta=ta.assign(query=query_a); tb=tb.assign(query=query_b)
    shared=sorted(set(ta.id)&set(tb.id))
    return ga,gb,pd.concat([ta,tb],ignore_index=True),f'공통 ID: {", ".join(shared) or "없음"}. 대상·속성·관계를 직접 확인하세요. 점수는 정답 확률이 아닙니다.'
