#!/usr/bin/env python3
"""
Build search_index.json from all webapp HTML docs.
Also injects id= attributes on <h2> tags for deep-link anchors.
Run: python3 webapp/build_index.py
"""

import os
import re
import json
from html import unescape

WEBAPP_DIR = os.path.dirname(os.path.abspath(__file__))
EXCLUDE = {'index.html', 'build_index.py', 'deploy.sh', 'nav.js'}


def slugify(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_]+', '-', text)
    text = re.sub(r'-+', '-', text)
    return text.strip('-')


def strip_tags(html):
    return re.sub(r'<[^>]+>', '', html)


def get_text_snippet(html, max_chars=200):
    text = unescape(strip_tags(html))
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:max_chars]


def parse_doc(filepath):
    with open(filepath, encoding='utf-8') as f:
        content = f.read()

    # Extract h1 title — strip <small> subtitle and decode HTML entities
    m = re.search(r'<h1[^>]*>(.*?)</h1>', content, re.IGNORECASE | re.DOTALL)
    if m:
        h1_inner = re.sub(r'<small[^>]*>.*?</small>', '', m.group(1), flags=re.DOTALL | re.IGNORECASE)
        h1_inner = re.sub(r'<br[^>]*>', ' ', h1_inner, flags=re.IGNORECASE)
        title = unescape(strip_tags(h1_inner)).strip()
    else:
        title = os.path.basename(filepath)
    title = re.sub(r'\s+', ' ', title)

    # Find all h2 sections
    # Split content on <h2 ...>...</h2> boundaries
    sections = []
    h2_pattern = re.compile(r'<h2([^>]*)>(.*?)</h2>', re.IGNORECASE | re.DOTALL)
    positions = [(m.start(), m.end(), m.group(1), m.group(2)) for m in h2_pattern.finditer(content)]

    for i, (start, end, attrs, heading_html) in enumerate(positions):
        heading_text = unescape(strip_tags(heading_html)).strip()
        heading_text = re.sub(r'\s+', ' ', heading_text)
        anchor = slugify(heading_text)

        # Text between this h2 and the next h2 (or end)
        next_start = positions[i + 1][0] if i + 1 < len(positions) else len(content)
        between = content[end:next_start]
        snippet = get_text_snippet(between)

        sections.append({'heading': heading_text, 'anchor': anchor, 'snippet': snippet})

    return title, sections


def inject_h2_ids(filepath, sections):
    """Add id= attributes to <h2> tags that don't already have one."""
    with open(filepath, encoding='utf-8') as f:
        content = f.read()

    modified = False
    for sec in sections:
        anchor = sec['anchor']
        # Find matching h2 without an id already set
        pattern = re.compile(
            r'<h2([^>]*)>(' + re.escape(sec['heading']) + r'(.*?))</h2>',
            re.IGNORECASE | re.DOTALL
        )

        def replacer(m, anchor=anchor):
            attrs = m.group(1)
            if 'id=' in attrs.lower():
                return m.group(0)
            return f'<h2{attrs} id="{anchor}">{m.group(2)}</h2>'

        new_content, n = pattern.subn(replacer, content)
        if n:
            content = new_content
            modified = True

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)


def main():
    index = []
    html_files = sorted([
        f for f in os.listdir(WEBAPP_DIR)
        if f.endswith('.html') and f not in EXCLUDE
    ])

    for filename in html_files:
        filepath = os.path.join(WEBAPP_DIR, filename)
        title, sections = parse_doc(filepath)
        inject_h2_ids(filepath, sections)
        index.append({'file': filename, 'title': title, 'sections': sections})
        print(f'  indexed: {filename} ({len(sections)} sections)')

    out_path = os.path.join(WEBAPP_DIR, 'search_index.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, separators=(',', ':'))

    print(f'\nWrote {out_path} ({len(index)} docs)')


if __name__ == '__main__':
    main()
