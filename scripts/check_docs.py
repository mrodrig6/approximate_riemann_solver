"""Execute complete documentation examples and validate repository links."""

import contextlib
import doctest
import importlib
import io
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FENCES = re.compile(r"^```([^\n]*)\n(.*?)^```", re.M | re.S)


def check_links(path, markdown):
    """Repository-file existence is checked here; MkDocs checks site anchors."""
    for _, target in re.findall(r"\[([^\]]*)\]\(([^)\s]+)\)", markdown):
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        destination = (path.parent / unquote(parsed.path)).resolve()
        if not destination.is_relative_to(ROOT) or not destination.exists():
            raise ValueError(f"{path.relative_to(ROOT)}: missing local target {target}")


def main():
    python_count = shell_count = output_count = 0
    paths = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    os.environ["MPLBACKEND"] = "Agg"
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="riemann-docs-") as directory:
        os.environ["MPLCONFIGDIR"] = str(Path(directory) / "matplotlib")
        os.environ["XDG_CACHE_HOME"] = str(Path(directory) / "cache")
        os.chdir(directory)
        try:
            for path in paths:
                markdown = path.read_text()
                check_links(path, markdown)
                for alt, target in re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", markdown):
                    if not alt.strip():
                        raise ValueError(f"{path}: image needs descriptive alt text")
                    if path != ROOT / "README.md" and "figures/" in target:
                        if "make_figures.py" not in markdown or "## Figure source" not in markdown:
                            raise ValueError(f"{path}: missing figure source and rebuild caption")
                blocks = list(FENCES.finditer(markdown))
                for index, block in enumerate(blocks):
                    language, source = block.groups()
                    if language.strip() == "python":
                        capture = io.StringIO()
                        with contextlib.redirect_stdout(capture):
                            exec(compile(source, str(path), "exec"), {"__name__": "__main__"})
                        python_count += 1
                        if index + 1 < len(blocks) and blocks[index + 1].group(1).strip() == "text":
                            expected = blocks[index + 1].group(2).strip()
                            if capture.getvalue().strip() != expected:
                                raise AssertionError(f"{path}: recorded Python output differs")
                            output_count += 1
                    elif language.strip() == "bash":
                        for line in source.splitlines():
                            words = shlex.split(line, comments=True)
                            if not words:
                                continue
                            # Setup/check commands are run by their dedicated workflows.
                            # Execute all solver invocations here; example files run separately.
                            if words[:3] == ["python", "-m", "src"]:
                                result = subprocess.run(
                                    [sys.executable, *words[1:]], capture_output=True, text=True
                                )
                                if result.returncode:
                                    raise RuntimeError(f"{path}: {line}\n{result.stderr}")
                                shell_count += 1
                                if index + 1 < len(blocks) and blocks[index + 1].group(1).strip() == "text":
                                    expected = blocks[index + 1].group(2)
                                    # Backend version varies; numerical output must match.
                                    def normalize(s):
                                        return re.sub(r"backend\s*:.*", "backend: installed backend", s).strip()
                                    if normalize(result.stdout) != normalize(expected):
                                        raise AssertionError(f"{path}: recorded CLI output differs")
                                    output_count += 1
            for name in ("src", "src.shallow_water"):
                result = doctest.testmod(importlib.import_module(name))
                if result.failed:
                    raise AssertionError(f"{name}: doctest failure")
                print(f"PASS {name}: {result.attempted} doctest examples")
        finally:
            os.chdir(original_cwd)
    print(f"PASS {len(paths)} documents; {python_count} Python blocks; "
          f"{shell_count} solver commands; {output_count} recorded outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
