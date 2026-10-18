"""Real CLIP image/text embeddings and small-gallery semantic retrieval.

Example: lab=CLIPSearch(); lab.index_images(); lab.search('a cat in the snow',3)
No model download occurs on import. Image names/categories are display metadata,
never inputs to the image encoder. Fixed pretrained model; no CLIP fine-tuning.
"""
from __future__ import annotations
import hashlib,json,os,time
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
import torch
from PIL import Image
from .representation import ASSETS

CLIP_ID='openai/clip-vit-base-patch32'
CLIP_REVISION='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
CLIP_FILES=('config.json','merges.txt','preprocessor_config.json','pytorch_model.bin',
            'special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.json')


def unit_rows(values: torch.Tensor) -> torch.Tensor:
    """Normalize finite nonzero [N,d] rows by L2 norm; return same-shape tensor.

    Reject zero vectors instead of pretending their cosine is defined. Preserves
    device, dtype and autograd. Example: unit_rows(tensor([[3.,4.]])) -> [[.6,.8]].
    """
    if values.ndim!=2 or not values.is_floating_point() or not torch.isfinite(values).all():
        raise ValueError('Expected finite floating [N,d] matrix')
    norms=values.norm(dim=1,keepdim=True)
    if len(values)==0 or values.shape[1]==0 or (norms<=0).any(): raise ValueError('Cosine requires nonzero vectors')
    return values/norms


def load_gallery() -> pd.DataFrame:
    """Verify and return the 40 bundled Commons photos in fixed P01..P40 order.

    Columns include id,category,title,source,author,license,license_url,sha256,path.
    Photos arrive with the fixed package install; no login/upload required. Paths
    are local filenames usable by PIL/Gradio. Raises on missing/corrupted images.
    Attribution lives in gallery.json and GALLERY-CREDITS.md; do not strip it.
    """
    rows=json.loads((ASSETS/'gallery.json').read_text(encoding='utf8'))
    for row in rows:
        path=ASSETS/(row['id']+'.jpg')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise RuntimeError('Photo integrity failure: '+row['id']+'; reinstall the fixed course package.')
        row['path']=str(path)
    return pd.DataFrame(rows)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 3) -> float:
    """Count relevant unique IDs among first k results / k; missing ranks count 0.

    k=3 positive integer; duplicate IDs are invalid. Relevance must be defined
    before inspecting scores. An absent-target query can have empty relevance.
    Example: precision_at_k(['A','B','C'],{'A','C'},3) == 2/3.
    """
    if isinstance(k,bool) or not isinstance(k,int) or k<1: raise ValueError('k must be a positive integer')
    if len(set(retrieved_ids))!=len(retrieved_ids): raise ValueError('duplicate retrieval IDs')
    return sum(x in relevant_ids for x in retrieved_ids[:k])/k


class CLIPSearch:
    """Load fixed CLIP on CPU, then separately index pixels and query by text.

    gallery=None loads checked classroom photos; cache_dir=None uses
    ~/.cache/luna-genai/clip-w05a. local_files_only=False permits initial model
    download (~605 MB); True requires existing complete HF cache. First call
    loads the model, index_images builds/reuses the feature cache. search never
    trains or re-encodes stored images. English is the primary query language.
    """
    def __init__(self, gallery: pd.DataFrame | None = None, cache_dir: str | Path | None = None,
                 *, local_files_only: bool = False):
        from .download import configure_hub
        configure_hub()
        try:
            from transformers import CLIPModel,CLIPProcessor
        except (ImportError,RuntimeError,OSError) as exc:
            raise RuntimeError('CLIP import failed. Restart a fresh runtime and run the fixed installation cell. '+str(exc)) from exc
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        self.gallery=(load_gallery() if gallery is None else gallery.copy()).reset_index(drop=True)
        required={'id','path','sha256','source','author','license','license_url','category'}
        if not required.issubset(self.gallery) or len(self.gallery)==0 or self.gallery.id.duplicated().any():
            raise ValueError('Gallery requires unique IDs and image/attribution metadata')
        self.cache_dir=Path(cache_dir or os.environ.get('LUNA_CLIP_CACHE',Path.home()/'.cache/luna-genai/clip-w05a'))
        self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.model_downloads=[]
        for filename in CLIP_FILES:
            kwargs=dict(repo_id=CLIP_ID,revision=CLIP_REVISION,filename=filename)
            try: path=hf_hub_download(**kwargs,local_files_only=True)
            except LocalEntryNotFoundError:
                if local_files_only: raise RuntimeError('Missing offline CLIP file: '+filename+'. Connect and prepare once.') from None
                try: path=hf_hub_download(**kwargs,etag_timeout=30)
                except Exception as exc:
                    raise RuntimeError('CLIP download failed for '+filename+'. Check network/free disk, then rerun; keep completed cache files.') from exc
                self.model_downloads.append(filename)
        directory=str(Path(path).parent)
        self.model=CLIPModel.from_pretrained(directory,local_files_only=True).eval()
        self.processor=CLIPProcessor.from_pretrained(directory,local_files_only=True)
        self.image_embeddings=None
        self.cache_reused=False

    def encode_images(self, paths: list[str], batch_size: int = 8) -> torch.Tensor:
        """Read RGB images; return normalized CPU [N,512], preserving path order.

        batch_size=8 limits memory. CLIP processor resizes/crops to 224x224 and
        applies its pretrained pixel normalization. Image filenames never enter
        the model. Transformers 5.15.1 returns projected features in pooler_output.
        """
        if not paths or batch_size<1: raise ValueError('nonempty paths and positive batch_size required')
        rows=[]
        with torch.inference_mode():
            for start in range(0,len(paths),batch_size):
                images=[]
                for path in paths[start:start+batch_size]:
                    with Image.open(path) as pic: images.append(pic.convert('RGB'))
                inputs=self.processor(images=images,return_tensors='pt')
                features=self.model.get_image_features(**inputs).pooler_output
                rows.append(unit_rows(features).cpu())
        return torch.cat(rows)

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Return normalized [N,512] for nonempty strings, without training.

        Tokenization pads a batch and truncates at the model's 77-token limit.
        Short English descriptions are recommended; Korean is exploratory.
        """
        if not texts or any(not isinstance(x,str) or not x.strip() for x in texts):
            raise ValueError('Enter a nonempty text query')
        inputs=self.processor(text=texts,padding=True,truncation=True,max_length=77,return_tensors='pt')
        with torch.inference_mode():
            return unit_rows(self.model.get_text_features(**inputs).pooler_output).cpu()

    def index_images(self, *, force: bool = False) -> torch.Tensor:
        """Compute/reuse normalized [N,512] pixel embeddings; return the matrix.

        force=False reuses matching model revision, library version, ordered
        image hashes and IDs. A corrupt cache raises; force=True rebuilds that
        derived file. Writes one local NPZ atomically, never changes weights.
        """
        payload={'model':CLIP_REVISION,'transformers':version('transformers'),
                 'images':self.gallery[['id','sha256']].to_dict('records')}
        fingerprint=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        path=self.cache_dir/(fingerprint+'.npz')
        self.cache_reused=path.exists() and not force
        if self.cache_reused:
            with np.load(path,allow_pickle=False) as data: matrix=torch.from_numpy(data['embeddings'].copy())
        else:
            matrix=self.encode_images(self.gallery.path.tolist())
            temporary=path.with_suffix('.tmp.npz');np.savez_compressed(temporary,embeddings=matrix.numpy());temporary.replace(path)
        if matrix.shape!=(len(self.gallery),512) or not torch.isfinite(matrix).all() or not torch.allclose(matrix.norm(dim=1),torch.ones(len(matrix)),atol=1e-5):
            raise RuntimeError('Invalid feature cache; call index_images(force=True) to rebuild derived embeddings.')
        self.image_embeddings=matrix
        return matrix

    def search(self, query: str, k: int = 3) -> pd.DataFrame:
        """Return descending cosine top-k rows with rank,id,cosine and metadata.

        Requires index_images first. k=3, range 1..gallery size. Ties preserve
        gallery order. Cosine is not a calibrated probability. A result is always
        returned even when the requested object is absent from the gallery.
        """
        if self.image_embeddings is None: raise RuntimeError('Call index_images() first')
        if isinstance(k,bool) or not isinstance(k,int) or not 1<=k<=len(self.gallery): raise ValueError('k outside gallery range')
        vector=self.encode_texts([query])[0]
        scores=(self.image_embeddings@vector).numpy()
        order=np.argsort(-scores,kind='stable')[:k]
        result=self.gallery.iloc[order].copy().reset_index(drop=True)
        result.insert(0,'rank',np.arange(1,k+1));result.insert(2,'cosine',scores[order])
        return result


def search_view(query: str, k: int | float, lab: CLIPSearch) -> tuple[list,pd.DataFrame,str]:
    """Gradio callback: (query,k,prepared lab) -> (gallery,score table,status).

    Gallery captions preserve author/license/source URLs; table includes IDs and
    cosine. k may be a whole-number slider value. No new model/index is created.
    """
    if isinstance(k,bool) or not float(k).is_integer(): raise ValueError('k must be integral')
    started=time.perf_counter();result=lab.search(query,int(k))
    gallery=[(r.path,f'{r.id} · cosine={r.cosine:.3f}\n{r.author} · {r.license}\n{r.source}\n{r.license_url}') for r in result.itertuples()]
    table=result[['rank','id','cosine','category','source']]
    status=f'{len(lab.gallery)}장 중 {len(result)}장 · {time.perf_counter()-started:.2f}초 · 코사인 점수는 확률이 아닙니다. 대상이 없어도 결과는 반환됩니다.'
    return gallery,table,status
