#!/usr/bin/env python3
"""블로그 새 글 생성 스크립트"""

import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BLOG_DIR = Path(__file__).resolve().parent.parent / "src" / "data" / "blog"


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s가-힣-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return re.sub(r"-+", "-", text).strip("-")


def ask(prompt: str, default: str = "") -> str:
    suffix = f" ({default})" if default else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    return answer or default


def main():
    print("\n📝 새 블로그 포스트 생성\n")

    title = ask("제목")
    if not title:
        print("제목은 필수입니다.")
        sys.exit(1)

    slug = ask("슬러그 (URL 경로)", slugify(title))
    description = ask("설명 (한 줄 요약)")
    tags_input = ask("태그 (쉼표 구분)", "others")
    draft_input = ask("초안 여부 (y/n)", "y")

    tags = [t.strip() for t in tags_input.split(",") if t.strip()]
    is_draft = draft_input.lower() == "y"
    pub_datetime = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    tags_yaml = "\n".join(f"  - {t}" for t in tags)

    frontmatter = f"""---
title: "{title}"
pubDatetime: {pub_datetime}
draft: {str(is_draft).lower()}
tags:
{tags_yaml}
description: "{description}"
---

여기에 글을 작성하세요.
"""

    file_path = BLOG_DIR / f"{slug}.md"

    if file_path.exists():
        print(f"\n⚠️  이미 존재하는 파일입니다: {file_path}")
        sys.exit(1)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(frontmatter, encoding="utf-8")

    rel_path = file_path.relative_to(Path.cwd())
    print(f"\n✅ 생성 완료: {rel_path}")
    if is_draft:
        print("   (초안 상태입니다. 발행하려면 draft: false로 변경하세요)")

    push_input = ask("\n바로 커밋하고 push할까요? (y/n)", "n")
    if push_input.lower() == "y":
        git_push(rel_path, title)


def git_push(rel_path: Path, title: str):
    repo_root = Path(__file__).resolve().parent.parent
    try:
        subprocess.run(["git", "add", str(rel_path)], cwd=repo_root, check=True)
        subprocess.run(
            ["git", "commit", "-m", f"post: {title}"],
            cwd=repo_root,
            check=True,
        )
        subprocess.run(["git", "push"], cwd=repo_root, check=True)
        print("\n🚀 push 완료!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ git 명령 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
