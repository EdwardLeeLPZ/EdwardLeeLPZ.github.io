#!/usr/bin/env python3
"""Guard against MathJax swallowing prose that merely contains dollar amounts.

assets/js/mathjax-setup.js deliberately drops the single-dollar inline
delimiter, because kramdown already turns this site's $$...$$ source into
\\(...\\). This script fails if that delimiter comes back while any rendered
page still carries two bare dollar signs inside one text block, which is the
exact combination that renders a paragraph as one unwrapped formula.

Run after `tools/jekyll-docker.sh build`:

    python3 tools/check-math-delimiters.py
"""

import pathlib
import re
import sys
from html.parser import HTMLParser

REPO = pathlib.Path(__file__).resolve().parent.parent
SETUP = REPO / "assets/js/mathjax-setup.js"
SITE = REPO / "_site"

# MathJax does not scan inside these, so dollars there are harmless.
SKIP_TAGS = {"script", "noscript", "style", "textarea", "pre", "code", "annotation", "annotation-xml"}
BLOCK_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "td", "th", "figcaption", "dd", "blockquote"}


class BlockText(HTMLParser):
    """Collect the visible text of each block element, the way MathJax sees it."""

    def __init__(self):
        super().__init__()
        self.open_tags = []
        self.buffer = []
        self.blocks = []

    def handle_starttag(self, tag, attrs):
        self.open_tags.append(tag)
        if tag in BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag):
        if tag in BLOCK_TAGS:
            self._flush()
        if tag in self.open_tags:
            del self.open_tags[len(self.open_tags) - 1 - self.open_tags[::-1].index(tag) :]

    def handle_data(self, data):
        if not any(tag in SKIP_TAGS for tag in self.open_tags):
            self.buffer.append(data)

    def _flush(self):
        text = re.sub(r"\s+", " ", "".join(self.buffer)).strip()
        if text:
            self.blocks.append(text)
        self.buffer = []


def main():
    if not SITE.is_dir():
        sys.exit("_site is missing; run tools/jekyll-docker.sh build first")

    # Strip line comments first: the file explains the removed delimiter in prose,
    # and that explanation must not read as the delimiter still being configured.
    setup = re.sub(r"//[^\n]*", "", SETUP.read_text(encoding="utf-8"))
    inline_math = re.search(r"inlineMath:\s*(\[.*?\]\s*\])", setup, re.S)
    if not inline_math:
        sys.exit("could not find inlineMath in assets/js/mathjax-setup.js")
    single_dollar_enabled = bool(re.search(r'\[\s*"\$"\s*,\s*"\$"\s*\]', inline_math.group(1)))

    offenders = []
    for page in sorted(SITE.rglob("*.html")):
        parser = BlockText()
        parser.feed(page.read_text(encoding="utf-8", errors="replace"))
        parser._flush()
        for block in parser.blocks:
            if block.count("$") >= 2:
                offenders.append((page.relative_to(SITE), block))

    print(f"single-dollar inline delimiter enabled: {single_dollar_enabled}")
    print(f"rendered blocks holding two or more bare dollar signs: {len(offenders)}")
    for path, block in offenders:
        print(f"  {path}: {block[:120]}")

    if single_dollar_enabled and offenders:
        sys.exit("FAIL: these blocks would render as unwrapped formulas")
    print("PASS")


if __name__ == "__main__":
    main()
