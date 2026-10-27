"""Explicit, approved PDF-only lessons and English exam-part writing pages."""
from pathlib import Path
import hashlib
import json
import re

MODE = 'board-review-pdf'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(root, relative):
    p = Path(relative)
    if not relative or p.is_absolute() or '..' in p.parts or '.git' in p.parts:
        raise ValueError('Unsafe PDF review path')
    target = (root / p).resolve()
    target.relative_to(root.resolve())
    return target


def kit_pdf_only(root, session, *, require_approval=False):
    mode = session.get('material_mode')
    if mode is None:
        return False
    if mode != MODE:
        raise ValueError('Unknown lesson material_mode')
    if (session.get('labs') or session.get('quiz') or session.get('colab_url')
            or session.get('assignment', {}).get('requested') is not False):
        raise ValueError('PDF-only review must have no lab, quiz or assignment')
    if len(session.get('pages', [])) != 1:
        raise ValueError('PDF-only review requires one canonical student note')
    if require_approval or (root / '.teaching.json').is_file():
        folder = root / 'briefs' / session['id']
        state = json.loads((folder / 'lesson-plan-state.json').read_text(encoding='utf-8'))
        agreed = state.get('approval', {})
        if (state.get('status') != 'approved' or not agreed.get('response')
                or state.get('proposal_sha256') != sha(folder / 'lesson-plan.yml')
                or agreed.get('sha256') != sha(folder / 'lesson-plan.yml')
                or state.get('rendered_sha256') != sha(folder / 'lesson-plan.md')):
            raise ValueError('PDF-only review needs an actually approved unchanged proposal')
    return True


def render_parts(body, *, question_count, writing_mm=100):
    if type(question_count) is not int or not 1 <= question_count <= 99:
        raise ValueError('Explicit exam question count required')
    if type(writing_mm) is not int or not 80 <= writing_mm <= 140:
        raise ValueError('Writing height must be 80 to 140 mm')
    pattern = r'<h2>Q(\d{1,2})\(([abc])\)\s*·'
    ids = [(int(q), p) for q, p in re.findall(pattern, body)]
    expected = [(q, p) for q in range(1, question_count + 1) for p in 'abc']
    if ids != expected:
        raise ValueError('Exam parts must appear once each in original Q1(a) order')
    label = '<h3>Board notes and calculations</h3>'

    def part(match):
        fragment = match.group(1)
        if fragment.count(label) != 1:
            raise ValueError('Each exam part needs exactly one writing heading')
        q, p = re.search(pattern, fragment).groups()
        name = f'Q{int(q)}({p})'
        box = (f'<div class="board-writing-space" data-board-part="{name}" '
               f'style="height:{writing_mm}mm"><span class="board-writing-label">'
               f'{name} · Board notes and calculations</span></div>')
        return '<section class="pdf-question">' + fragment.replace(label, box) + '</section>'

    body = re.sub(r'(<h2>Q\d{1,2}\([abc]\)\s*·[\s\S]*?)(?=<h2>|$)', part, body)
    return body.replace('<h2>After-class review notes</h2>',
                        '<h2 class="board-reflection">After-class review notes</h2>')


def require_pdf_review(root, session):
    """Require current source/PDF/every-image hashes, not a visual-check waiver."""
    if not kit_pdf_only(root, session, require_approval=True):
        raise ValueError('This is not an approved PDF-only review')
    sid = session['id']
    proof = json.loads((root / 'briefs' / sid / 'verification/pdf-render.json').read_text(encoding='utf-8'))
    if proof.get('status') != 'passed' or proof.get('coverage') != 'all-pages':
        raise ValueError('Actual all-page PDF visual review required')
    note = session['pages'][0]
    expected = {
        'student': (note, 'course/handouts/' + Path(note).stem + '.pdf'),
        'guide': (f'instructor/{sid}/teacher-foundations-ko.md',
                  f'instructor/{sid}/teacher-foundations-ko.pdf'),
        'script_ko': (f'instructor/{sid}/speaker-script-ko.md',
                      f'instructor/{sid}/speaker-script-ko.pdf'),
    }
    for role, (source, pdf) in expected.items():
        item = proof['files'][role]
        if item['source'] != source or item['path'] != pdf:
            raise ValueError('Wrong reviewed PDF/source for ' + role)
        if (sha(inside(root, source)) != item['source_sha256']
                or sha(inside(root, pdf)) != item['sha256']):
            raise ValueError('PDF/source changed after actual visual review: ' + role)
        images = item['page_images']
        if item['pages'] < 1 or [x['page'] for x in images] != list(range(1, item['pages'] + 1)):
            raise ValueError('Missing reviewed PDF pages: ' + role)
        if any(sha(inside(root, x['path'])) != x['sha256'] for x in images):
            raise ValueError('Reviewed page image changed: ' + role)
    extras = proof.get('aligned_sources', {})
    for rel in (f'instructor/{sid}/speaker-script-en.md', f'instructor/{sid}/run-sheet.md',
                f'instructor/{sid}/exercise-map.md'):
        if extras.get(rel) != sha(inside(root, rel)):
            raise ValueError('Aligned instructor source changed: ' + rel)
    return proof
