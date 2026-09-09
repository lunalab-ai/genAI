"""Typeset LaTeX and package flat, script-free HTML/PNG for Notion import.

Canonical Markdown stays editable. Export images are local (no login/CDN),
double-resolution, and share the HTML folder. The manifest is outside the ZIP.
Local Chromium verification does not certify Notion's actual importer.
"""
from __future__ import annotations
import argparse, base64, hashlib, html, json, mimetypes, re, tempfile, zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse
from markdown_it import MarkdownIt

VERSION = 'notion-html-png-v2'
CODE = re.compile(r'(?m)^ {0,3}(`{3,}|~{3,})[^\n]*\n[\s\S]*?^ {0,3}\1[ \t]*(?:\n|$)|(`+)[^\n]*?\2')
MATH = re.compile(r'\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\([^\n]*?\\\)|(?<![\\\w])\$(?!\s)[^$\n]+?(?<![\\\s])\$(?![A-Za-z0-9_$])')
STYLE = """body{font-family:'Malgun Gothic',Arial,sans-serif;font-size:17px;line-height:1.7;color:#172333;background:white;max-width:1000px;margin:30px auto;padding:20px}img{max-width:100%;height:auto}img.formula{vertical-align:middle}div.equation{overflow-x:auto;margin:20px 0;text-align:center}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccd5df;padding:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f7;padding:12px}a{color:#185c99}h1,h2,h3{line-height:1.3}mjx-container{overflow:visible!important}"""

def protect_math(text):
    """Return Markdown with opaque math placeholders plus (TeX, display) records.

    Fenced code and inline code are literal; R's $ operator is not a delimiter.
    Inputs are Markdown text; no formula is evaluated as program code.
    """
    formulas=[]
    def replace(m):
        value=m[0]; display=value.startswith(('$$',r'\['))
        trim=2 if display or value.startswith(r'\(') else 1
        if not value[trim:-trim].strip():
            raise ValueError('Empty LaTeX expression')
        formulas.append((value[trim:-trim],display))
        return f'MATHPLACEHOLDER{len(formulas)-1}END'
    result=[]; last=0
    for match in CODE.finditer(text):
        result.extend((MATH.sub(replace,text[last:match.start()]),match[0]));last=match.end()
    result.append(MATH.sub(replace,text[last:]))
    return ''.join(result),formulas

def math_html(text):
    """Render Markdown, protecting LaTeX underscores/backslashes from Markdown."""
    protected, formulas=protect_math(text)
    body=MarkdownIt('commonmark',{'html':False}).enable('table').render(protected)
    for i,(tex,display) in enumerate(formulas):
        opening,closing=(r'\[',r'\]') if display else (r'\(',r'\)')
        body=body.replace(f'MATHPLACEHOLDER{i}END',opening+html.escape(tex)+closing)
    return body,formulas

def math_script(vendor):
    return '<script>MathJax={startup:{typeset:true},svg:{fontCache:"none"},options:{enableMenu:false}};</script><script>'+vendor.read_text(encoding='utf-8')+'</script>'

def verify_math(tab):
    tab.evaluate('async()=>{if(window.MathJax?.startup)await MathJax.startup.promise;await document.fonts.ready}')
    if tab.locator('[data-mjx-error],mjx-merror').count():
        raise ValueError('LaTeX rendering failed: '+tab.locator('[data-mjx-error],mjx-merror').first.inner_text())

def export_bundle(source: Path, output: Path, pages=None, vendor=None, repository=None):
    """Package selected public Markdown pages (default all under source).

    source: course/notion directory; output: new ZIP path; pages: local Path list.
    Returns manifest with exact source/export hashes and rendered image counts.
    Rejects remote/missing images and path traversal. Does not publish anything.
    """
    from playwright.sync_api import sync_playwright
    source=source.resolve(); output=output.resolve()
    vendor=vendor or Path(__file__).resolve().parent/'vendor/tex-svg-full.js'
    pages=sorted(p.resolve() for p in (pages or source.rglob('*.md')))
    for p in pages:p.relative_to(source)
    mapping={p:p.stem+'-'+hashlib.sha256(p.relative_to(source).as_posix().encode()).hexdigest()[:8]+'.html' for p in pages}
    entries={};records=[]
    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as pw:
        browser=pw.chromium.launch()
        tab=browser.new_page(viewport={'width':1400,'height':1000},device_scale_factor=2)
        tab.route('http://**/*',lambda route:route.abort())
        tab.route('https://**/*',lambda route:route.abort())
        for p in pages:
            text=p.read_text(encoding='utf-8');body,formulas=math_html(text)
            document='<html><head><meta charset="utf-8"><title>'+html.escape(text.splitlines()[0].lstrip('# '))+'</title><style>'+STYLE+'</style></head><body>'+body+math_script(vendor)+'</body></html>'
            tab.set_content(document,wait_until='load')
            # Resolve links before loading local image bytes; never request the network.
            for link in tab.locator('a[href]').all():
                href=link.get_attribute('href')
                if not href or urlparse(href).scheme or href.startswith('#'):continue
                target=(p.parent/unquote(href.split('#')[0])).resolve()
                if target in mapping:link.evaluate('(e,v)=>e.setAttribute("href",v)',mapping[target])
                elif repository and target.is_relative_to(source.parent.parent):
                    link.evaluate('(e,v)=>e.setAttribute("href",v)','https://github.com/'+repository+'/blob/main/'+target.relative_to(source.parent.parent).as_posix())
            image_count=0
            for img in tab.locator('img').all():
                href=img.get_attribute('src') or ''
                if urlparse(href).scheme:raise ValueError('Use local images: '+href)
                asset=(p.parent/unquote(href)).resolve();asset.relative_to(source)
                data=asset.read_bytes();mime=mimetypes.guess_type(asset.name)[0]
                if mime not in ('image/png','image/jpeg','image/gif','image/webp','image/svg+xml'):raise ValueError('Unsupported image: '+str(asset))
                img.evaluate('(e,v)=>e.src=v','data:'+mime+';base64,'+base64.b64encode(data).decode())
                img.evaluate('async e=>{await e.decode()}')
                if asset.suffix.lower()=='.svg':
                    data=img.screenshot(type='png');extension='.png'
                else:extension=asset.suffix.lower()
                name='image-'+hashlib.sha256(data).hexdigest()[:20]+extension
                entries[name]=data
                img.evaluate('(e,v)=>{e.dataset.exportSrc=v}',name);image_count+=1
            verify_math(tab)
            nodes=tab.locator('mjx-container').all()
            if len(nodes)!=len(formulas):raise ValueError(f'Formula count mismatch: {p}: {len(nodes)} != {len(formulas)}')
            for tex,display in formulas:
                node=tab.locator('mjx-container').first
                svg=node.locator('svg').first
                box=svg.bounding_box()
                if not box or box['width']<=0 or box['height']<=0:raise ValueError('Invisible formula: '+tex)
                data=svg.screenshot(type='png')
                name='equation-'+hashlib.sha256(data).hexdigest()[:20]+'.png';entries[name]=data
                node.evaluate('''(e,v)=>{const i=document.createElement('img');i.src=v.src;i.dataset.exportSrc=v.name;i.alt=v.tex;i.className='formula '+(v.display?'display':'inline');i.width=Math.ceil(v.width);i.height=Math.ceil(v.height);if(v.display){const d=document.createElement('div');d.className='equation';d.append(i);e.replaceWith(d)}else{e.replaceWith(i)}}''',{'src':'data:image/png;base64,'+base64.b64encode(data).decode(),'name':name,'tex':tex,'width':box['width'],'height':box['height'],'display':display})
            # Notion rich text has equations, but no inline image type. Keep
            # complete prose/table layout together and retain clickable citations.
            tab.evaluate('''()=>{const cs=Array.from(document.querySelectorAll('table,ul,ol,blockquote,p,h1,h2,h3,h4,h5,h6')).filter(e=>!e.querySelector('pre')&&(e.querySelector('img.formula.inline')||(e.tagName==='TABLE'&&e.querySelector('img.formula'))));cs.filter(e=>!cs.some(a=>a!==e&&a.contains(e))).forEach(e=>e.dataset.typesetBlock='yes')}''')
            grouped=0
            while tab.locator('[data-typeset-block]').count():
                block=tab.locator('[data-typeset-block]').first
                block.evaluate('e=>{e.style.width="720px";e.style.maxWidth="100%";e.style.boxSizing="border-box";e.style.backgroundColor="white";e.style.padding="4px 8px"}')
                alt=block.evaluate('''e=>{const c=e.cloneNode(true);c.querySelectorAll('img').forEach(i=>i.replaceWith(document.createTextNode(i.alt)));return c.textContent}''')
                links=block.locator('a[href]').evaluate_all('(xs)=>xs.map(x=>({href:x.getAttribute("href"),text:x.textContent}))')
                data=block.screenshot(type='png');box=block.bounding_box()
                name='typeset-'+hashlib.sha256(data).hexdigest()[:20]+'.png';entries[name]=data
                block.evaluate('''(e,v)=>{const p=document.createElement('p'),i=document.createElement('img');i.src=v.src;i.dataset.exportSrc=v.name;i.alt=v.alt;i.width=Math.ceil(v.width);i.className='typeset-block';p.append(i);for(const link of v.links){const a=document.createElement('a');a.href=link.href;a.textContent=link.text;p.append(document.createElement('br'),a)}e.replaceWith(p)}''',{'src':'data:image/png;base64,'+base64.b64encode(data).decode(),'name':name,'alt':alt,'width':box['width'],'links':links})
                grouped+=1
            tab.evaluate('()=>{document.querySelectorAll("script,mjx-assistive-mml").forEach(e=>e.remove());document.querySelectorAll("img[data-export-src]").forEach(e=>{e.src=e.dataset.exportSrc;delete e.dataset.exportSrc})}')
            exported=tab.content()
            if '<script' in exported or '<svg' in exported:raise ValueError('Dynamic/vector content survived export')
            entries[mapping[p]]=exported.encode('utf-8')
            records.append({'source':p.relative_to(source).as_posix(),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'html':mapping[p],'formulas':len(formulas),'images':image_count,'typeset_blocks':grouped})
        browser.close()
    referenced=set()
    for name,data in entries.items():
        if name.endswith('.html'):referenced.update(re.findall(r'<img[^>]+src="([^"]+)"',data.decode('utf-8')))
    entries={name:data for name,data in entries.items() if name.endswith('.html') or name in referenced}
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(entries.items()):
            info=zipfile.ZipInfo('notion/'+name,(2026,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)
    # Reopen the exact packaged HTML with local files, network blocked.
    with tempfile.TemporaryDirectory() as tmp,sync_playwright() as pw:
        with zipfile.ZipFile(output) as z:
            if z.testzip():raise ValueError('Corrupt ZIP')
            z.extractall(tmp)
        browser=pw.chromium.launch();tab=browser.new_page()
        tab.route('https://**/*',lambda r:r.abort());tab.route('http://**/*',lambda r:r.abort())
        for record in records:
            tab.goto((Path(tmp)/'notion'/record['html']).as_uri())
            if not tab.evaluate('Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)'):raise ValueError('Broken packaged image: '+record['html'])
        browser.close()
    manifest={'exporter':VERSION,'zip':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'pages':records,'assets':len(entries)-len(records),'local_render_verified':True,'notion_import_verified':False}
    output.with_suffix('.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--pages',nargs='*',type=Path);parser.add_argument('--repository')
    args=parser.parse_args();print(json.dumps(export_bundle(args.source,args.output,args.pages,repository=args.repository),ensure_ascii=False,indent=2))
