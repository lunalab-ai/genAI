"""Small inspectable evaluation calculations; no model-quality guarantees."""
from __future__ import annotations
from collections import Counter
import math
import re


def perplexity_from_losses(mean_losses, token_counts) -> dict:
    """Batch mean NLLs and valid NEXT-token counts -> weighted NLL and exp(NLL).

    Nonempty equal-length sequences; losses finite >=0, counts positive ints.
    Padding/prompt ignored by the caller must not enter counts. Comparable PPL
    also needs the same tokenizer, documents, context and loss-mask convention.
    """
    losses, counts = list(mean_losses), list(token_counts)
    if (not losses or len(losses) != len(counts) or
        any(not math.isfinite(x) or x < 0 for x in losses) or
        any(type(n) is not int or n < 1 for n in counts)):
        raise ValueError('유한한 비음수 NLL과 양의 정수 토큰 수가 필요합니다.')
    nll = sum(x*n for x,n in zip(losses,counts))/sum(counts)
    return {'nll': nll, 'target_tokens': sum(counts),
            'perplexity': math.exp(nll) if nll < 709 else float('inf')}


def _tokens(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError('비어 있지 않은 문장이 필요합니다.')
    tokens = re.findall(r"\w+|[^\w\s]", text.casefold())
    if not tokens: raise ValueError('토큰이 없습니다.')
    return tokens


def overlap_metrics(candidate: str, reference: str, max_order: int = 2) -> dict:
    """Single-reference educational BLEU-N (no smoothing), ROUGE-1 and L.

    Lowercase word/punctuation tokenization, max_order 1..4; NOT an official
    benchmark pipeline. BLEU uses clipped precision, geometric mean and BP;
    any missing order scores zero. ROUGE-L uses dynamic-programming LCS.
    Returns intermediate counts so denominators can be inspected. No fitting.
    """
    if type(max_order) is not int or not 1 <= max_order <= 4:
        raise ValueError('max_order=1..4가 필요합니다.')
    c,r = _tokens(candidate),_tokens(reference)
    rows=[]
    for n in range(1,max_order+1):
        cc=Counter(tuple(c[i:i+n]) for i in range(len(c)-n+1))
        rr=Counter(tuple(r[i:i+n]) for i in range(len(r)-n+1))
        matched=sum((cc & rr).values()); total=sum(cc.values())
        rows.append({'order':n,'clipped_matches':matched,'candidate_ngrams':total,
                     'precision':matched/total if total else 0.})
    bp=min(1.,math.exp(1-len(r)/len(c)))
    bleu=bp*math.exp(sum(math.log(x['precision']) for x in rows)/max_order) if all(x['precision']>0 for x in rows) else 0.
    overlap=sum((Counter(c)&Counter(r)).values())
    previous=[0]*(len(r)+1)
    for word in c:
        current=[0]
        for j,other in enumerate(r,1):
            current.append(previous[j-1]+1 if word==other else max(previous[j],current[-1]))
        previous=current
    lcs=previous[-1]
    return {'bleu':bleu,'bleu_order':max_order,'brevity_penalty':bp,'ngram_counts':rows,
            'rouge1_precision':overlap/len(c),'rouge1_recall':overlap/len(r),
            'rougeL_precision':lcs/len(c),'rougeL_recall':lcs/len(r),
            'candidate_tokens':c,'reference_tokens':r,'lcs_length':lcs,
            'scope':'교육용 단일 참조·소문자 단어/구두점; 사실성 점수 아님'}
