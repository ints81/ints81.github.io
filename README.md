# Ints의 블로그

[AstroPaper](https://github.com/satnaing/astro-paper) 기반 개인 블로그.

## 설치

```bash
bun install
```

## 실행

```bash
# 로컬 개발 서버
bun run dev

# 프로덕션 빌드
bun run build

# 빌드 미리보기
bun run preview
```

## 글 작성

`/mnt/c/Users/ints/Documents/blog_posts/` 디렉토리에 마크다운 파일을 작성하면, 동기화 스크립트가 자동으로 블로그에 반영합니다.

동기화 스크립트가 처리하는 작업:

1. 마크다운 파일을 `src/data/blog/`로 복사
2. 프론트매터 자동 생성 (title, pubDatetime 등)
3. 마크다운 내 로컬 이미지를 `src/assets/images/`로 복사 및 경로 갱신
4. 이미 존재하는 파일은 프론트매터를 보존하고 본문만 갱신
5. 변경 사항이 있으면 자동으로 git commit & push

프론트매터의 `tags`와 `description`은 빈칸으로 생성되므로, `src/data/blog/`에 복사된 파일에서 직접 채워 넣으면 됩니다.

## Git 인증 설정

crontab에서 자동으로 `git push`가 동작하려면 credential 저장이 필요합니다.

```bash
git config --global credential.helper store
```

설정 후 한 번 수동으로 push하면 토큰이 `~/.git-credentials`에 저장됩니다.

```bash
git push
# Username: <GitHub 아이디>
# Password: <Personal Access Token>
```

Personal Access Token은 [GitHub Settings > Tokens](https://github.com/settings/tokens)에서 `repo` 권한으로 생성합니다.

## Crontab 등록

동기화 스크립트를 5분 간격으로 자동 실행하려면:

```bash
# 등록
./scripts/install-cron.sh

# 상태 확인
./scripts/install-cron.sh status

# 해제
./scripts/install-cron.sh uninstall
```

수동 실행도 가능합니다:

```bash
python3 scripts/sync_posts.py
```

로그는 `scripts/sync_posts.log`에 기록됩니다.
