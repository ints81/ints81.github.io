#!/usr/bin/env python3
"""
블로그 포스트 동기화 스크립트

/mnt/c/Users/ints/Documents/blog_posts 의 마크다운 파일을 감시하여
블로그 저장소(src/data/blog)로 동기화한다.

- 새 파일: 복사 + 프론트매터 자동 생성 + 이미지 처리
- 기존 파일: 본문만 갱신 (프론트매터 보존)
- 변경이 있으면 git commit & push
"""

import logging
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

SOURCE_DIR = Path("/mnt/c/Users/ints/Documents/blog_posts")
REPO_ROOT = Path(__file__).resolve().parent.parent
BLOG_DIR = REPO_ROOT / "src" / "data" / "blog"
ASSETS_DIR = REPO_ROOT / "src" / "assets" / "images"

KST = timezone(timedelta(hours=9))
MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
HTML_IMAGE_RE = re.compile(r'<img\s[^>]*src=["\']([^"\']+)["\'][^>]*/?\s*>', re.IGNORECASE)
FRONTMATTER_RE = re.compile(r"^---[ \t]*\n(.*?\n)---[ \t]*\n", re.DOTALL)

LOG_FILE = REPO_ROOT / "scripts" / "sync_posts.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────


def title_from_filename(name: str) -> str:
    return Path(name).stem.replace("-", " ").replace("_", " ")


def make_frontmatter(title: str) -> str:
    pub = datetime.now(KST).strftime("%Y-%m-%dT00:00:00+09:00")
    return (
        "---\n"
        f'title: "{title}"\n'
        f"pubDatetime: {pub}\n"
        "draft: false\n"
        "tags:\n"
        "  - 일반\n"
        'description: ""\n'
        "---\n"
    )


def split_frontmatter(text: str) -> tuple[str | None, str]:
    """프론트매터와 본문을 분리. 프론트매터가 없으면 (None, 전체 텍스트)."""
    m = FRONTMATTER_RE.match(text)
    if m:
        return text[: m.end()].rstrip("\n"), text[m.end() :]
    return None, text


WIN_ABS_RE = re.compile(r"^[A-Za-z]:[/\\]")


def _win_to_wsl(path: str) -> str:
    """Windows 절대 경로를 WSL 경로로 변환. (C:\\Users\\... -> /mnt/c/Users/...)"""
    path = path.replace("\\", "/")
    drive = path[0].lower()
    return f"/mnt/{drive}{path[2:]}"


def _resolve_image(img_path: str, source_file: Path) -> Path | None:
    """이미지 경로를 WSL 파일시스템 경로로 변환."""
    if img_path.startswith(("http://", "https://", "//")):
        return None

    if WIN_ABS_RE.match(img_path):
        return Path(_win_to_wsl(img_path))

    return (source_file.parent / img_path.replace("\\", "/")).resolve()


def _copy_image(src_img: Path, source_file: Path) -> str | None:
    """이미지를 assets로 복사하고 새 경로를 반환. 실패 시 None."""
    if not src_img.is_file():
        log.warning("이미지 누락: %s (in %s)", src_img, source_file.name)
        return None

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    dest_img = ASSETS_DIR / src_img.name
    shutil.copy2(src_img, dest_img)
    log.info("이미지 복사: %s -> %s", src_img.name, dest_img.relative_to(REPO_ROOT))
    return f"../../assets/images/{src_img.name}"


def process_images(content: str, source_file: Path) -> str:
    """마크다운 및 HTML 이미지를 찾아 assets로 복사하고 경로를 갱신."""

    def _replace_md(match: re.Match) -> str:
        alt, img_path = match.group(1), match.group(2)
        src_img = _resolve_image(img_path, source_file)
        if src_img is None:
            return match.group(0)
        new_path = _copy_image(src_img, source_file)
        if new_path is None:
            return match.group(0)
        return f"![{alt}]({new_path})"

    def _replace_html(match: re.Match) -> str:
        img_path = match.group(1)
        src_img = _resolve_image(img_path, source_file)
        if src_img is None:
            return match.group(0)
        new_path = _copy_image(src_img, source_file)
        if new_path is None:
            return match.group(0)
        return match.group(0).replace(img_path, new_path)

    content = MD_IMAGE_RE.sub(_replace_md, content)
    content = HTML_IMAGE_RE.sub(_replace_html, content)
    return content


# ── git ──────────────────────────────────────────────────────────────


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def commit_and_push() -> bool:
    now = datetime.now(KST)
    msg = f"{now.strftime('%Y-%m-%d %H:%M')} New post added"

    git("add", ".")

    diff = git("diff", "--cached", "--quiet")
    if diff.returncode == 0:
        log.info("스테이징된 변경 없음, 커밋 생략")
        return False

    result = git("commit", "-m", msg)
    if result.returncode != 0:
        log.error("커밋 실패:\n%s", result.stderr)
        return False

    result = git("push")
    if result.returncode != 0:
        log.error("푸시 실패:\n%s", result.stderr)
        return False

    log.info("커밋 & 푸시 완료: %s", msg)
    return True


# ── sync ─────────────────────────────────────────────────────────────


def sync():
    if not SOURCE_DIR.exists():
        log.info("소스 디렉토리 없음: %s", SOURCE_DIR)
        return

    BLOG_DIR.mkdir(parents=True, exist_ok=True)
    changed = False

    for src_file in sorted(SOURCE_DIR.glob("*.md")):
        raw = src_file.read_text(encoding="utf-8")
        processed = process_images(raw, src_file)
        dest_file = BLOG_DIR / src_file.name

        if dest_file.exists():
            changed |= _update_existing(src_file, dest_file, processed)
        else:
            changed |= _create_new(src_file, dest_file, processed)

    if changed:
        commit_and_push()
    else:
        log.info("변경 사항 없음")


def _create_new(src_file: Path, dest_file: Path, processed: str) -> bool:
    _, body = split_frontmatter(processed)
    title = title_from_filename(src_file.name)
    fm = make_frontmatter(title)
    dest_file.write_text(fm + "\n" + body.lstrip("\n"), encoding="utf-8")
    log.info("새 글 추가: %s", src_file.name)
    return True


def _update_existing(src_file: Path, dest_file: Path, processed: str) -> bool:
    existing = dest_file.read_text(encoding="utf-8")
    existing_fm, existing_body = split_frontmatter(existing)
    _, new_body = split_frontmatter(processed)

    if existing_fm:
        updated = existing_fm + "\n\n" + new_body.lstrip("\n")
    else:
        updated = processed

    if updated == existing:
        return False

    dest_file.write_text(updated, encoding="utf-8")
    log.info("본문 갱신: %s", src_file.name)
    return True


if __name__ == "__main__":
    sync()
