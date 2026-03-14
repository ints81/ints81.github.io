#!/usr/bin/env python3
"""
사용자 지정 디렉토리를 감시하여, 새 마크다운 파일이 생성되면 frontmatter를 자동으로 추가한다.

- frontmatter가 없는 .md 파일만 처리 (기존 frontmatter는 유지)
- sync_posts.py와 동일한 규칙: 경로→category, tags 기본값, series 비움
- systemd user service로 등록하여 백그라운드 실행 가능

사용법: python3 watch_and_add_frontmatter.py <감시할 디렉토리 경로>
의존성: pip install watchdog
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# sync_posts 모듈 import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sync_posts import (
    category_from_path,
    make_frontmatter,
    parse_source_frontmatter,
    split_frontmatter,
    title_from_filename,
)


def add_frontmatter_if_needed(file_path: Path, source_dir: Path) -> bool:
    """
    파일에 frontmatter가 없으면 추가한다.
    반환: 실제로 추가했으면 True
    """
    if not file_path.is_file() or not file_path.suffix.lower() == ".md":
        return False

    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception:
        return False

    fm_raw, body = split_frontmatter(text)

    # 이미 frontmatter가 있으면 (--- 로 시작하는 완전한 블록) 건너뛴다
    if fm_raw is not None and fm_raw.strip():
        return False

    # body가 없거나 --- 로 시작하는 불완전한 frontmatter 의심 시에도 보수적으로 처리
    content = body.lstrip("\n")
    if text.strip().startswith("---") and "---" not in text[3:100]:
        # 불완전한 frontmatter일 수 있음 - 건너뛰기
        return False

    parent, child = category_from_path(file_path, source_dir)
    title = title_from_filename(file_path.name)
    tags, series = parse_source_frontmatter(fm_raw)
    fm = make_frontmatter(title, parent, child, tags=tags, series=series)

    new_content = fm + "\n" + content
    file_path.write_text(new_content, encoding="utf-8")
    return True


def run_watcher(source_dir: Path):
    import logging

    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    LOG_DIR = Path.home() / ".logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(LOG_DIR / "watch_frontmatter.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger(__name__)

    class Handler(FileSystemEventHandler):
        def _process(self, path: str):
            p = Path(path)
            if not p.suffix.lower() == ".md":
                return
            if add_frontmatter_if_needed(p, source_dir):
                log.info("frontmatter 추가: %s", p.relative_to(source_dir))

        def on_created(self, event):
            if event.is_directory:
                return
            self._process(event.src_path)

        def on_modified(self, event):
            if event.is_directory:
                return
            self._process(event.src_path)

    if not source_dir.exists():
        log.error("감시 디렉토리가 없습니다: %s", source_dir)
        sys.exit(1)

    log.info("감시 시작: %s", source_dir)
    observer = Observer()
    observer.schedule(Handler(), str(source_dir), recursive=True)
    observer.start()

    try:
        while observer.is_alive():
            observer.join(timeout=1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="사용자 지정 디렉토리 감시 및 frontmatter 자동 추가")
    parser.add_argument(
        "source_dir",
        type=Path,
        help="감시할 마크다운 디렉토리 경로",
    )
    args = parser.parse_args()
    run_watcher(args.source_dir.resolve())
