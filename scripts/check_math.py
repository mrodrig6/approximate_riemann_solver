#!/usr/bin/env python3
r"""Check the Markdown maths against GitHub's renderer constraints.

Run from the repository root::

    python scripts/check_math.py

This structural checker enforces the portable syntax documented in
``docs/reference/documentation.md``. It does not invoke TeX or validate browser
rendering. Unsupported constructs can render as literal text without raising
an error, so keep renderer-independent restrictions explicit.

Each rule below exists because of something that does not render:

* display maths must be flush left, with blank lines around the block; either
  use a self-contained ``$$ ... $$`` line or separate ``$$`` delimiter lines;
* no multi-line environments (``aligned``, ``cases``, ``array``, ``bmatrix``),
  because they need ``\\`` row separators and the Markdown parser consumes
  backslashes first;
* no ``\tag`` or ``\label``: equation numbering is unavailable;
* no cosmetic spacing macros (``\,``, ``\;``, ``\!``), each of which is
  another backslash for the parser to eat;
* no literal ``*`` inside maths: GitHub resolves emphasis before the maths
  span reaches the renderer, and ``*`` -- unlike ``_`` -- pairs intraword, so
  two asterisks in a paragraph wrap everything between them in ``<em>`` and
  the equation is destroyed.  Write ``\ast`` instead;
* balanced ``$`` delimiters on every line.

Exits non-zero if anything fails.
"""

from __future__ import annotations

import glob
import os
import re

#: Constructs that do not survive GitHub's Markdown-then-MathJax pipeline,
#: mapped to the reason, for the error message.
FORBIDDEN = {
    r"\begin{": "multi-line environments need \\\\ row separators",
    r"\end{": "multi-line environments need \\\\ row separators",
    r"\tag{": "equation numbering is not available",
    r"\label{": "equation numbering is not available",
    r"\,": "cosmetic spacing macro; use a plain space",
    r"\;": "cosmetic spacing macro; use a plain space",
    r"\!": "cosmetic spacing macro; delete it",
    "*": "GitHub pairs '*' as emphasis before the maths is rendered; use \\ast",
}

MATH = re.compile(r"\$\$([\s\S]+?)\$\$|(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)")


def math_spans(text):
    """Yield ``(line_number, body)`` for every maths span outside code fences."""
    lines = text.split("\n")
    incode = False
    masked = []
    for line in lines:
        if line.strip().startswith("```"):
            incode = not incode
            masked.append("")
            continue
        masked.append("" if incode else line)
    clean = "\n".join(masked)
    # Blank out inline code spans too: a backticked ``$$`` in prose about these
    # rules is not maths, and neither is a ``$`` in a shell snippet.
    clean = re.sub(r"`[^`\n]*`", lambda m: " " * len(m.group(0)), clean)
    for m in MATH.finditer(clean):
        body = m.group(1) if m.group(1) is not None else m.group(2)
        yield clean[: m.start()].count("\n") + 1, body


def check(path):
    """Return a list of problem strings for one Markdown file."""
    problems = []
    name = os.path.relpath(path)
    lines = open(path).read().split("\n")

    incode = False
    display = False
    for i, line in enumerate(lines, 1):
        if line.strip().startswith("```"):
            incode = not incode
            continue
        if incode:
            continue
        stripped = line.strip()
        if stripped == "$$":
            if line != stripped:
                problems.append(f"{name}:{i}: display maths is indented")
            if not display and i > 1 and lines[i - 2].strip():
                problems.append(f"{name}:{i}: no blank line before display maths")
            if display and i < len(lines) and lines[i].strip():
                problems.append(f"{name}:{i}: no blank line after display maths")
            display = not display
            continue
        if display:
            continue
        if stripped.startswith("$$"):
            if line != stripped:
                problems.append(
                    f"{name}:{i}: display maths is indented; GitHub will not "
                    f"treat it as maths"
                )
            if not (stripped.endswith("$$") and len(stripped) > 4):
                problems.append(
                    f"{name}:{i}: display maths must open and close on one line"
                )
            if i > 1 and lines[i - 2].strip() != "":
                problems.append(f"{name}:{i}: no blank line before display maths")
            if i < len(lines) and lines[i].strip() != "":
                problems.append(f"{name}:{i}: no blank line after display maths")
            continue
        if line.count("$") % 2:
            problems.append(f"{name}:{i}: unbalanced $ delimiters")

    if display:
        problems.append(f"{name}: unclosed display maths")

    text = open(path).read()
    for line_no, body in math_spans(text):
        for bad, why in FORBIDDEN.items():
            if bad in body:
                problems.append(f"{name}:{line_no}: {bad!r} -- {why}")
        if "\\\\" in body:
            problems.append(
                f"{name}:{line_no}: '\\\\' row separator -- the Markdown parser "
                f"eats it before the maths renderer sees it"
            )
    return problems


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    targets = sorted(glob.glob(os.path.join(root, "docs", "**", "*.md"), recursive=True))
    targets.append(os.path.join(root, "README.md"))

    problems = []
    spans = 0
    for path in targets:
        problems.extend(check(path))
        spans += sum(1 for _ in math_spans(open(path).read()))

    for p in problems:
        print(p)
    print(
        f"\n{spans} maths spans in {len(targets)} files; "
        f"{len(problems)} problem(s)"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
