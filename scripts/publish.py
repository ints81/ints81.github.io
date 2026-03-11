#!/usr/bin/env python3
"""변경된 블로그 글을 모아서 commit & push하는 스크립트"""

import subprocess
import sys
from pathlib import Path

BLOG_DIR = "src/data/blog"
REPO_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, **kwargs)


def get_changed_posts() -> tuple[list[str], list[str], list[str]]:
    """변경된 블로그 파일을 (새 파일, 수정 파일, 삭제 파일)로 분류"""
    result = run(["git", "status", "--porcelain", "--", BLOG_DIR])
    if result.returncode != 0:
        print(f"❌ git status 실패: {result.stderr}")
        sys.exit(1)

    new, modified, deleted = [], [], []
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        status = line[:2].strip()
        file_path = line[3:].strip().strip('"')
        if status in ("??", "A"):
            new.append(file_path)
        elif status in ("M", "AM", "MM"):
            modified.append(file_path)
        elif status == "D":
            deleted.append(file_path)

    return new, modified, deleted


def main():
    print("\n📦 블로그 글 발행\n")

    new, modified, deleted = get_changed_posts()
    total = len(new) + len(modified) + len(deleted)

    if total == 0:
        print("변경된 글이 없습니다.")
        return

    if new:
        print(f"  📝 새 글 ({len(new)}개):")
        for f in new:
            print(f"     + {f}")
    if modified:
        print(f"  ✏️  수정 ({len(modified)}개):")
        for f in modified:
            print(f"     ~ {f}")
    if deleted:
        print(f"  🗑️  삭제 ({len(deleted)}개):")
        for f in deleted:
            print(f"     - {f}")

    print(f"\n  총 {total}개 파일")
    confirm = input("\n커밋하고 push할까요? (y/n) (n): ").strip()
    if confirm.lower() != "y":
        print("취소했습니다.")
        return

    parts = []
    if new:
        titles = [Path(f).stem.replace("-", " ") for f in new]
        parts.append("add: " + ", ".join(titles))
    if modified:
        titles = [Path(f).stem.replace("-", " ") for f in modified]
        parts.append("update: " + ", ".join(titles))
    if deleted:
        titles = [Path(f).stem.replace("-", " ") for f in deleted]
        parts.append("delete: " + ", ".join(titles))
    commit_msg = "post: " + "; ".join(parts)

    custom = input(f"\n커밋 메시지 ({commit_msg}): ").strip()
    if custom:
        commit_msg = custom

    try:
        all_files = new + modified + deleted
        run(["git", "add", "--"] + all_files, check=True)
        run(["git", "commit", "-m", commit_msg], check=True)
        run(["git", "push"], check=True)
        print("\n🚀 push 완료!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ git 명령 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
