"""W10B optional SDXL project recipe, pinned to a reviewed official script.

Default functions prepare data and an argument list only. SDXL training is NOT
executed by making a plan. The explicit run function requires an available GPU.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import torch
from .diffusion_adaptation import _integer, load_teaching_images

SDXL_ID='stabilityai/stable-diffusion-xl-base-1.0'
SDXL_REVISION='462165984030d82259a11f4367a4eed129e94a7b'
SCRIPT_URL='https://raw.githubusercontent.com/huggingface/diffusers/v0.40.0/examples/dreambooth/train_dreambooth_lora_sdxl.py'
SCRIPT_SHA256='88ffa83ccf7f45fb79fb7fc7fd23bf49ba87a5ba8b011ae295dd0b57fdd5ddfa'


def prepare_project_files(directory: str | Path) -> dict:
    """Write six instance PNGs and two held-out PNGs to a NEW project directory.

    The instance folder contains images only: the official local-folder loader
    must not be asked to open a CSV as an image. Metadata lives one level above.
    No model, training, upload, personal picture, or credential is involved.
    """
    root=Path(directory)
    if root.exists():raise FileExistsError('Choose a new SDXL project run directory')
    data=load_teaching_images()
    (root/'instance').mkdir(parents=True);(root/'validation').mkdir();(root/'class').mkdir()
    rows=[]
    for image,entry in zip(data['rgb'],data['manifest']['images']):
        folder='instance' if entry['split']=='train' else 'validation'
        filename=entry['id']+'.png';image.save(root/folder/filename)
        rows.append({'file_name':folder+'/'+filename,'text':entry['caption'],'group_id':entry['group_id'],'split':entry['split']})
    with (root/'metadata.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result={'directory':str(root),'train_images':6,'validation_images':2,
        'instance_prompt':data['manifest']['instance_prompt'],'class_prompt':data['manifest']['class_prompt'],
        'mode':'prepared files only; SDXL training/generation not executed',
        'dataset_provenance':data['manifest']['provenance']}
    (root/'project.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def sdxl_command(directory: str | Path, script: str | Path, *, rank: int = 4,
                 steps: int = 100, resolution: int = 512, seed: int = 42) -> list[str]:
    """Build a single-process Accelerate argv, without launching it.

    Uses instance/class separation, prior weight1, and frozen text encoders.
    No Hub upload or paid service. More steps/rank/resolution have resource and
    generalization effects; these are starting settings, not quality promises.
    """
    rank=_integer(rank,'rank',1,16);steps=_integer(steps,'steps',1,1000)
    resolution=_integer(resolution,'resolution',256,1024);seed=_integer(seed,'seed',0,2**31-1)
    if resolution%8:raise ValueError('Use a multiple-of8 resolution')
    root=Path(directory);meta=json.loads((root/'project.json').read_text(encoding='utf-8'))
    if len(list((root/'instance').glob('*.png')))!=6 or len(list((root/'validation').glob('*.png')))!=2:
        raise ValueError('Prepared train/held-out images are incomplete')
    return [sys.executable,'-m','accelerate.commands.launch','--num_processes','1',str(Path(script)),
        '--pretrained_model_name_or_path',SDXL_ID,'--revision',SDXL_REVISION,
        '--instance_data_dir',str(root/'instance'),'--instance_prompt',meta['instance_prompt'],
        '--class_data_dir',str(root/'class'),'--class_prompt',meta['class_prompt'],
        '--with_prior_preservation','--prior_loss_weight','1.0','--num_class_images','3',
        '--resolution',str(resolution),'--train_batch_size','1','--gradient_accumulation_steps','1',
        '--rank',str(rank),'--learning_rate','1e-4','--max_train_steps',str(steps),
        '--gradient_checkpointing','--mixed_precision','fp16',
        '--validation_prompt',meta['instance_prompt']+' on a wooden table beside a window',
        '--validation_epochs','10','--num_validation_images','1',
        '--report_to','none','--seed',str(seed),'--output_dir',str(root/'adapter')]


def download_reviewed_script(cache_dir: str | Path | None = None) -> Path:
    """Fetch official Diffusers0.40.0 script with timeout and exact SHA-256.

    Reuses only a matching cache; a changed/corrupt cached file raises and is
    preserved. Does not install packages or execute the downloaded script.
    """
    root=Path(cache_dir or Path.home()/'.cache/luna-genai/w10b-diffusers040')
    root.mkdir(parents=True,exist_ok=True);path=root/'train_dreambooth_lora_sdxl.py'
    if path.exists():
        payload=path.read_bytes()
    else:
        with urllib.request.urlopen(SCRIPT_URL,timeout=60) as response:payload=response.read(1024*1024)
        if hashlib.sha256(payload).hexdigest()!=SCRIPT_SHA256:raise RuntimeError('Official script hash changed')
        path.write_bytes(payload)
    if hashlib.sha256(payload).hexdigest()!=SCRIPT_SHA256:raise RuntimeError('Cached script checksum mismatch')
    return path


def run_optional_sdxl(directory: str | Path, *, steps: int = 100, rank: int = 4,
                      resolution: int = 512) -> None:
    """Explicit opt-in actual GPU training; failures are not replaced by records.

    Requires a CUDA GPU and a compatible torchvision installation. It downloads
    several GB of SDXL weights and may exceed a T4's memory. This optional path
    is marked unverified until actually executed on that device. No upload flag.
    """
    if not torch.cuda.is_available():raise RuntimeError('SDXL optional training requires an explicitly selected supported GPU; CPU record analysis does not execute it')
    try:import torchvision  # noqa: F401
    except ImportError as exc:raise RuntimeError('Install the torchvision build matching the current torch in this isolated runtime, then restart') from exc
    script=download_reviewed_script()
    command=sdxl_command(directory,script,steps=steps,rank=rank,resolution=resolution)
    subprocess.run(command,check=True)
