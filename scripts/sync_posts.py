#!/usr/bin/env python3
"""
블로그 포스트 동기화 스크립트

지정한 디렉토리의 마크다운 파일을 블로그 저장소(src/data/blog)로 동기화한다.

- 새 파일: 복사 + 프론트매터 자동 생성 + 이미지 처리
- 기존 파일: 본문만 갱신 (프론트매터 보존)
- 변경이 있으면 git commit & push

사용법: python3 sync_posts.py <마크다운 파일이 있는 디렉토리>
"""

import argparse
import logging
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
BLOG_DIR = REPO_ROOT / "src" / "data" / "blog"
IMAGES_DIR = REPO_ROOT / "public" / "images"

KST = timezone(timedelta(hours=9))
MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
HTML_IMAGE_RE = re.compile(r'<img\s[^>]*src=["\']([^"\']+)["\'][^>]*/?\s*>', re.IGNORECASE)
FRONTMATTER_RE = re.compile(r"^---[ \t]*\n(.*?\n)---[ \t]*\n", re.DOTALL)

LOG_DIR = Path.home() / ".logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "sync_posts.log"
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


def category_from_path(src_file: Path, source_dir: Path) -> tuple[str, str]:
    """
    소스 파일의 디렉토리 경로에서 category (parent, child) 추출.
    - 경로가 parent/child/file.md → (parent, child)
    - 경로가 parent/file.md → (parent, 기타)
    - 경로가 file.md (루트) → (일상, 기타)
    """
    try:
        rel = src_file.parent.relative_to(source_dir)
        parts = [p for p in rel.parts if not p.startswith("_")]
        if len(parts) >= 2:
            return parts[0], parts[1]
        if len(parts) == 1:
            return parts[0], "기타"
    except ValueError:
        pass
    return "일상", "기타"


TAGS_RE = re.compile(r"^tags:\s*$", re.MULTILINE)
TAGS_ITEM_RE = re.compile(r"^[\s-]+(.+)$", re.MULTILINE)
SERIES_NAME_RE = re.compile(r"^series:.*?^[\s]*name:\s*['\"]?([^'\"]+)['\"]?", re.MULTILINE | re.DOTALL)
SERIES_ORDER_RE = re.compile(r"^series:.*?^[\s]*order:\s*(\d+)", re.MULTILINE | re.DOTALL)


def parse_source_frontmatter(fm_text: str | None) -> tuple[list[str] | None, dict | None]:
    """
    소스 frontmatter에서 tags, series 파싱.
    반환: (tags 리스트 또는 None, series 딕셔너리 또는 None)
    """
    if not fm_text:
        return None, None
    tags = None
    series = None
    try:
        import yaml
        data = yaml.safe_load(fm_text)
        if data:
            t = data.get("tags")
            if isinstance(t, list):
                tags = [str(x).strip() for x in t if x]
            elif isinstance(t, str) and t.strip():
                tags = [x.strip() for x in t.split(",") if x.strip()]
            s = data.get("series")
            if isinstance(s, dict) and s.get("name") and "order" in s:
                series = {"name": str(s["name"]).strip(), "order": int(s["order"])}
    except ImportError:
        pass
    except Exception:
        pass
    if tags is None:
        m = TAGS_RE.search(fm_text)
        if m:
            rest = fm_text[m.end() :].split("\n")
            found = []
            for line in rest[:20]:
                if line.strip() and (line.startswith("-") or line.startswith("  -")):
                    tag = line.lstrip("- ").strip().strip("'\"").split("#")[0].strip()
                    if tag:
                        found.append(tag)
                elif line.strip() and not line.startswith(" ") and not line.startswith("-"):
                    break
            if found:
                tags = found
    if series is None:
        mn = SERIES_NAME_RE.search(fm_text)
        mo = SERIES_ORDER_RE.search(fm_text)
        if mn and mo:
            series = {"name": mn.group(1).strip(), "order": int(mo.group(1))}
    return tags, series


def make_frontmatter(
    title: str,
    category_parent: str,
    category_child: str,
    tags: list[str] | None = None,
    series: dict | None = None,
) -> str:
    pub = datetime.now(KST).strftime("%Y-%m-%dT%H:%M:%S+09:00")
    tag_list = tags if tags else ["일반"]
    lines = [
        "---",
        f'title: "{title}"',
        f"pubDatetime: {pub}",
        "draft: false",
        "tags:",
        *[f"  - {t}" for t in tag_list],
        "category:",
        f"  parent: {category_parent}",
        f"  child: {category_child}",
        'description: ""',
    ]
    if series:
        lines.extend([
            "series:",
            f"  name: {series['name']}",
            f"  order: {series['order']}",
        ])
    return "\n".join(lines) + "\n---\n"


def split_frontmatter(text: str) -> tuple[str | None, str]:
    """프론트매터와 본문을 분리. 프론트매터가 없으면 (None, 전체 텍스트)."""
    m = FRONTMATTER_RE.match(text)
    if m:
        return text[: m.end()].rstrip("\n"), text[m.end() :]
    return None, text


# tags: 블록 - list item 줄만 매칭 (- 로 시작하는 줄), 다음 키 직전까지
TAGS_BLOCK_RE = re.compile(
    r"^tags:\s*\n(?:(?!\n?\s*(?:category|description|series|ogImage|draft|title|pubDatetime)\s*:).+\n)*(?=\n\s*(?:category|description|series|ogImage|draft|title|pubDatetime)\s*:|\n---|\Z)",
    re.MULTILINE,
)
# tags: [a,b] 플로우 형태 (한 줄)
TAGS_FLOW_RE = re.compile(r"^tags:\s*\[[^\]]*\]\s*", re.MULTILINE)
# series: name: x\n  order: N 형태의 블록
SERIES_BLOCK_RE = re.compile(
    r"^series:\s*\n(?:\s+name:\s*.+\n)?(?:\s+order:\s*.+\n)?\s*",
    re.MULTILINE,
)


def merge_tags_series_into_frontmatter(
    existing_fm: str,
    source_tags: list[str] | None,
    source_series: dict | None,
) -> str:
    """
    기존 frontmatter에 소스의 tags, series를 반영한 새 frontmatter 문자열 반환.
    source_tags/source_series가 None이면 해당 필드는 기존값 유지.
    yaml load/dump 대신 regex로 치환해 datetime 등 직렬화 이슈를 피함.
    """
    result = existing_fm
    if source_tags is not None:
        new_tags_block = "tags:\n" + "\n".join(f"  - {t}" for t in source_tags) + "\n"
        if TAGS_BLOCK_RE.search(result):
            result = TAGS_BLOCK_RE.sub(new_tags_block, result, count=1)
        else:
            # tags가 없으면 description 앞에 삽입 (없으면 맨 뒤)
            if "\ndescription:" in result:
                result = result.replace("\ndescription:", "\n" + new_tags_block + "description:", 1)
            else:
                result = result.rstrip()
                if not result.endswith("\n"):
                    result += "\n"
                result += new_tags_block
    if source_series is not None:
        new_series_block = (
            "series:\n"
            f"  name: {source_series['name']}\n"
            f"  order: {source_series['order']}\n"
        )
        if SERIES_BLOCK_RE.search(result):
            result = SERIES_BLOCK_RE.sub(new_series_block, result, count=1)
        else:
            if "\ndescription:" in result:
                result = result.replace("\ndescription:", "\n" + new_series_block + "description:", 1)
            else:
                result = result.rstrip()
                if not result.endswith("\n"):
                    result += "\n"
                result += "\n" + new_series_block
    return result


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
    """이미지를 public/images/로 복사하고 새 경로를 반환. 실패 시 None."""
    if not src_img.is_file():
        log.warning("이미지 누락: %s (in %s)", src_img, source_file.name)
        return None

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    dest_img = IMAGES_DIR / src_img.name
    shutil.copy2(src_img, dest_img)
    log.info("이미지 복사: %s -> %s", src_img.name, dest_img.relative_to(REPO_ROOT))
    return f"/images/{src_img.name}"


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


def sync(source_dir: Path):
    if not source_dir.exists():
        log.info("소스 디렉토리 없음: %s", source_dir)
        return

    BLOG_DIR.mkdir(parents=True, exist_ok=True)
    changed = False

    for src_file in sorted(source_dir.rglob("*.md")):
        raw = src_file.read_text(encoding="utf-8")
        processed = process_images(raw, src_file)
        try:
            rel_path = src_file.relative_to(source_dir)
        except ValueError:
            rel_path = src_file.name
        dest_file = BLOG_DIR / rel_path

        if dest_file.exists():
            changed |= _update_existing(src_file, dest_file, processed)
        else:
            changed |= _create_new(src_file, dest_file, processed, source_dir)

    if changed:
        commit_and_push()
    else:
        log.info("변경 사항 없음")


def _create_new(src_file: Path, dest_file: Path, processed: str, source_dir: Path) -> bool:
    fm_raw, body = split_frontmatter(processed)
    title = title_from_filename(src_file.name)
    parent, child = category_from_path(src_file, source_dir)
    tags, series = parse_source_frontmatter(fm_raw)
    fm = make_frontmatter(title, parent, child, tags=tags, series=series)
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    dest_file.write_text(fm + "\n" + body.lstrip("\n"), encoding="utf-8")
    log.info("새 글 추가: %s", src_file.name)
    return True


def _update_existing(src_file: Path, dest_file: Path, processed: str) -> bool:
    existing = dest_file.read_text(encoding="utf-8")
    existing_fm, existing_body = split_frontmatter(existing)
    source_fm, new_body = split_frontmatter(processed)

    # 소스의 tags, series를 기존 frontmatter에 반영
    if existing_fm:
        tags, series = parse_source_frontmatter(source_fm)
        updated_fm = merge_tags_series_into_frontmatter(existing_fm, tags, series)
        # 기존 dest 형식 유지: frontmatter와 body 사이 개행 수
        sep = "\n\n" if "\n\n" in existing[len(existing_fm) : len(existing_fm) + 3] else "\n"
        updated = updated_fm + sep + new_body.lstrip("\n")
    else:
        updated = processed

    if updated == existing:
        return False

    dest_file.write_text(updated, encoding="utf-8")
    log.info("본문 갱신: %s", src_file.name)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="블로그 포스트 동기화")
    parser.add_argument("source_dir", type=Path, help="마크다운 파일이 있는 디렉토리 경로")
    args = parser.parse_args()
    sync(args.source_dir)
