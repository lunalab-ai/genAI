"""Verify every LaTeX expression in generated student/instructor notebooks.

No code execution, outputs, uploads, or secrets are involved. This is a rendering
check in addition to (not a replacement for) fresh-kernel runtime validation.
"""
import json
from pathlib import Path
from notion_export import math_html, math_script, verify_math, STYLE

def embed_course_images(notebook, root):
    """Embed this course's public figures as notebook PNG/JPEG attachments.

    Only URLs under this repository's course/notion tree are resolved locally.
    Public source links are retained. Raw/private paths are never embedded.
    This makes preview/Colab figures independent of an unpublished Git tag.
    """
    import base64,hashlib,mimetypes,re,yaml
    from playwright.sync_api import sync_playwright
    config=yaml.safe_load((root/'course/course.yml').read_text(encoding='utf-8'))
    repository=config.get('repository')
    if not repository and (root/'course/hub.yml').exists():
        repository=yaml.safe_load((root/'course/hub.yml').read_text(encoding='utf-8')).get('repository')
    if not repository:return
    pattern=re.compile(r'!\[([^\]]*)\]\((https://raw\.githubusercontent\.com/'+re.escape(repository)+r'/[^/]+/(course/notion/[^)]+))\)')
    for cell in notebook.cells:
        if cell.cell_type!='markdown':continue
        links=[]
        def replace(match):
            asset=(root/match[3]).resolve();asset.relative_to((root/'course/notion').resolve())
            data=asset.read_bytes();mime=mimetypes.guess_type(asset.name)[0]
            if mime=='image/svg+xml':
                with sync_playwright() as pw:
                    browser=pw.chromium.launch();tab=browser.new_page(viewport={'width':1300,'height':900},device_scale_factor=2)
                    tab.set_content('<body style="margin:0;background:white"><img style="max-width:1200px" src="data:image/svg+xml;base64,'+base64.b64encode(data).decode()+'">')
                    img=tab.locator('img');img.evaluate('async e=>await e.decode()');data=img.screenshot(type='png');browser.close()
                mime='image/png'
            if mime not in ('image/png','image/jpeg','image/gif','image/webp'):raise ValueError('Unsupported notebook figure')
            name='figure-'+hashlib.sha256(data).hexdigest()[:16]+('.png' if mime=='image/png' else asset.suffix)
            cell.setdefault('attachments',{})[name]={mime:base64.b64encode(data).decode()}
            links.append('[Figure source]('+match[2]+')')
            return '!['+match[1]+'](attachment:'+name+')'
        cell.source=pattern.sub(replace,cell.source)
        if links:cell.source+='\n\n'+' · '.join(links)+'\n'

def verify_notebook_math(paths):
    from playwright.sync_api import sync_playwright
    bodies=[];expected=0
    for path in paths:
        nb=json.loads(Path(path).read_text(encoding='utf-8'))
        for cell in nb['cells']:
            if cell['cell_type']!='markdown':continue
            body,formulas=math_html(''.join(cell['source']))
            # Only math is rendered: notebook pictures may intentionally be remote.
            if formulas:
                for tex,display in formulas:
                    import html
                    bodies.append((r'\[' if display else r'\(')+html.escape(tex)+(r'\]' if display else r'\)'))
                expected+=len(formulas)
    if not expected:return {'notebooks':len(paths),'formulas':0}
    vendor=Path(__file__).resolve().parent/'vendor/tex-svg-full.js'
    with sync_playwright() as pw:
        browser=pw.chromium.launch();tab=browser.new_page()
        tab.set_content('<meta charset="utf-8"><style>'+STYLE+'</style>'+''.join('<p>'+b+'</p>' for b in bodies)+math_script(vendor))
        verify_math(tab)
        if tab.locator('mjx-container').count()!=expected:raise ValueError('Notebook math count mismatch')
        if not tab.locator('mjx-container svg').evaluate_all('(xs)=>xs.every(x=>x.getBoundingClientRect().width>0&&x.getBoundingClientRect().height>0)'):raise ValueError('Invisible notebook formula')
        browser.close()
    return {'notebooks':len(paths),'formulas':expected}
