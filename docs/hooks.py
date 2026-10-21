"""Use README as the site home and adapt repository links for the site."""

import re
from pathlib import Path
from urllib.parse import urlsplit

from mkdocs.plugins import event_priority
from mkdocs.structure.files import File


def on_files(files, config):
    """Create a virtual home page; do not write a second README to disk."""
    existing = files.get_file_from_path("index.md")
    if existing is not None:
        files.remove(existing)
    files.append(
        File.generated(config, "index.md", content=(Path(config.config_file_path).parent / "README.md").read_text())
    )
    return files


@event_priority(100)
def on_page_markdown(markdown, page, config, files):
    """Make README links and repository-file links work in built HTML."""
    repo = config["repo_url"].rstrip("/")
    root = Path(config.config_file_path).parent.resolve()
    docs = (root / config["docs_dir"]).resolve()
    home = page.file.src_uri == "index.md"

    def rewrite(match):
        target = match.group(1)
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            return match.group(0)
        path = parsed.path
        if home and path.startswith("docs/"):
            path = path[5:]
        elif home:
            path = f"{repo}/blob/master/{path}"
        else:
            resolved = (docs / Path(page.file.src_uri).parent / path).resolve()
            if resolved.is_relative_to(root) and (
                not resolved.is_relative_to(docs) or resolved.suffix == ".py"
            ):
                path = f"{repo}/blob/master/{resolved.relative_to(root).as_posix()}"
        return path + (f"#{parsed.fragment}" if parsed.fragment else "")

    return re.sub(r"(?<=\]\()([^)\s]+)(?=\))", rewrite, markdown)
