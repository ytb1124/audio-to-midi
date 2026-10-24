"""Bundle an arm64 ffmpeg and its Homebrew dylib closure for Trackform.

The resulting directory is consumed as a Tauri resource.  System frameworks
and Apple-provided libraries are intentionally not copied; Homebrew dylibs are
copied and rewritten to use @loader_path-relative references.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "src-tauri" / "resources" / "ffmpeg"
FFMPEG = Path(os.environ.get("TRACKFORM_FFMPEG", "/opt/homebrew/bin/ffmpeg"))


def deps(path: Path) -> list[str]:
    output = subprocess.check_output(["otool", "-L", str(path)], text=True)
    return [line.strip().split(" ", 1)[0] for line in output.splitlines()[1:]]


def resolve_dep(dep: str, current: Path) -> Path | None:
    if dep.startswith("/opt/homebrew/"):
        path = Path(dep)
        return path.resolve() if path.exists() else None
    if dep.startswith("@rpath/"):
        name = Path(dep).name
        candidates = [
            current.parent / name,
            Path("/opt/homebrew/lib") / name,
        ]
        candidates.extend(Path("/opt/homebrew/opt").glob(f"*/lib/{name}"))
        candidates.extend(Path("/opt/homebrew/Cellar").glob(f"*/*/lib/{name}"))
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
    return None


def main() -> None:
    if not FFMPEG.exists():
        raise SystemExit(f"ffmpeg not found: {FFMPEG}")
    if subprocess.check_output(["file", str(FFMPEG)], text=True).find("arm64") < 0:
        raise SystemExit(f"ffmpeg is not arm64: {FFMPEG}")

    bin_dir = DEST / "bin"
    lib_dir = DEST / "lib"
    bin_dir.mkdir(parents=True, exist_ok=True)
    lib_dir.mkdir(parents=True, exist_ok=True)

    bundled_ffmpeg = bin_dir / "ffmpeg"
    # copyfile avoids carrying Homebrew's provenance xattr into the app.
    shutil.copyfile(FFMPEG.resolve(), bundled_ffmpeg)
    bundled_ffmpeg.chmod(0o755)

    queue = [FFMPEG.resolve()]
    seen: set[Path] = set()
    copied: dict[str, Path] = {}
    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        for dep in deps(current):
            source = resolve_dep(dep, current)
            if source is None:
                continue
            name = source.name
            target = lib_dir / name
            if not target.exists():
                shutil.copyfile(source, target)
            copied[dep] = target
            if source not in seen:
                queue.append(source)

    # Rewrite every copied Homebrew reference.  The executable is one level
    # below lib/, while dylibs sit next to each other in lib/.
    files = [bundled_ffmpeg, *sorted(lib_dir.glob("*.dylib"))]
    for current in files:
        for dep in deps(current):
            source = resolve_dep(dep, current)
            if source is None:
                continue
            target = lib_dir / source.name
            if not target.exists():
                continue
            replacement = f"@loader_path/../lib/{target.name}" if current == bundled_ffmpeg else f"@loader_path/{target.name}"
            subprocess.run(["install_name_tool", "-change", dep, replacement, str(current)], check=True)
        if current != bundled_ffmpeg:
            subprocess.run(["install_name_tool", "-id", f"@loader_path/{current.name}", str(current)], check=True)

    # install_name_tool invalidates the Homebrew ad-hoc signatures. Re-sign
    # locally so macOS can execute the resource before the whole app receives
    # its eventual Developer ID signature in the release pipeline.
    for library in sorted(lib_dir.glob("*.dylib")):
        subprocess.run(["xattr", "-d", "com.apple.provenance", str(library)], check=False)
        subprocess.run(["codesign", "--force", "--sign", "-", str(library)], check=True)
    subprocess.run(["codesign", "--force", "--sign", "-", str(bundled_ffmpeg)], check=True)
    # macOS may re-attach provenance while signing copied files; remove it
    # after signing as well so the unsigned local resource is executable.
    for library in sorted(lib_dir.glob("*.dylib")):
        subprocess.run(["xattr", "-d", "com.apple.provenance", str(library)], check=False)
    subprocess.run(["xattr", "-d", "com.apple.provenance", str(bundled_ffmpeg)], check=False)

    print(f"Bundled {bundled_ffmpeg}")
    print(f"Bundled {len(list(lib_dir.glob('*.dylib')))} dylibs")


if __name__ == "__main__":
    main()
