"""Shared legal heading recognition for chunking and validation."""
import re

ARTICLE_RE = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:[·º\-][ \t]*)?Điều[ \t]+(\d+[a-zđ]?)[ \t]*[.:][ \t]*(\S[^\n]*)$"
)


def is_inside_quote(text: str, pos: int) -> bool:
    prefix = text[:pos]
    if text.count("“") != text.count("”") or text.count('"') % 2:
        # Some PDF text layers drop a closing mark or mix straight/curly marks.
        # Recover scope at the next explicit closing quote instead of letting
        # one damaged quotation swallow every subsequent article. The document
        # article-count gate still has to pass; clauses with damaged quotes are
        # kept as whole articles with a parse warning.
        inside = False
        for mark in re.findall('[“”"]', prefix):
            inside = True if mark == "“" else False if mark == "”" else not inside
        return inside
    return prefix.count("“") > prefix.count("”") or prefix.count('"') % 2 == 1


def article_matches(text: str) -> list:
    return [match for match in ARTICLE_RE.finditer(text) if not is_inside_quote(text, match.start())]
