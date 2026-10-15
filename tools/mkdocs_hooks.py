"""mkdocs hook: links to files the site does not carry point to the repository.

The site publishes the pages of docs/ and their images (mkdocs.yml). A relative link from a page to anything else,
in docs/ or outside it (code, logs, data, a directory), is rewritten to its GitHub URL, on the branch the public
repository is built from. Links that resolve to nothing are left alone, so `mkdocs build --strict` still fails on
them."""
import posixpath
import re
from pathlib import Path

BRANCH = "master"
ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"(\[[^\]]*\]\()([^)\s]+)(\))")
FENCE = re.compile(r"^(```|~~~)")


def _rewrite(target, page_dir, docs, built, repo):
    if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith(("#", "/")):
        return target
    path, _, anchor = target.partition("#")
    src = posixpath.normpath(posixpath.join(page_dir, path))
    if src in built:
        return target
    full = (docs / src).resolve()
    if not full.exists() or ROOT not in full.parents:
        return target
    kind = "tree" if full.is_dir() else "blob"
    url = f"{repo}/{kind}/{BRANCH}/{full.relative_to(ROOT).as_posix()}"
    return url + ("#" + anchor if anchor else "")


def on_page_markdown(markdown, page, config, files):
    docs = Path(config["docs_dir"]).resolve()
    built = {f.src_uri for f in files if not f.inclusion.is_excluded()}
    page_dir = posixpath.dirname(page.file.src_uri)
    repo = config["repo_url"].rstrip("/")
    out, fenced = [], False
    for line in markdown.split("\n"):
        if FENCE.match(line.lstrip()):
            fenced = not fenced
        elif not fenced:
            line = LINK.sub(lambda m: m[1] + _rewrite(m[2], page_dir, docs, built, repo) + m[3], line)
        out.append(line)
    return "\n".join(out)
