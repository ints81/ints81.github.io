# Ints의 블로그

[AstroPaper](https://github.com/satnaing/astro-paper) 기반 개인 블로그.

## 설치

```bash
git clone https://github.com/ints81/ints81.github.io.git blog
cd blog
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

원하는 디렉토리에 마크다운 파일을 작성한 뒤, `sync_posts.py`에 해당 디렉토리 경로를 인자로 전달하면 블로그에 동기화됩니다.

```bash
python3 scripts/sync_posts.py /path/to/your/posts
```

동기화 스크립트가 처리하는 작업:

1. 지정한 디렉토리의 마크다운 파일을 `src/data/blog/`로 복사
2. 프론트매터 자동 생성 (title, pubDatetime 등)
3. 마크다운 내 로컬 이미지를 `public/images/`로 복사 및 경로 갱신
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
# 등록 (디렉토리 경로 필수)
./scripts/install-cron.sh install /path/to/your/posts

# 상태 확인
./scripts/install-cron.sh status

# 해제
./scripts/install-cron.sh uninstall
```

로그는 `~/.logs/sync_posts.log`에 기록됩니다.

## Frontmatter 자동 추가 (watch 스크립트)

`watch_and_add_frontmatter.py`는 지정한 디렉토리를 감시하여, **frontmatter가 없는** 새 마크다운 파일이 생성되면 자동으로 frontmatter를 추가합니다.

- 경로에서 category 추출 (sync_posts.py와 동일 규칙)
- tags 기본값 `["일반"]`, series는 비움
- 기존 frontmatter가 있는 파일은 건드리지 않음

### 수동 실행

```bash
python3 scripts/watch_and_add_frontmatter.py /path/to/your/posts
```

### systemd 서비스 등록 (백그라운드 자동 실행)

로그인 시 자동으로 감시를 시작하려면:

```bash
# 서비스 설치 (감시할 디렉토리 경로 필수)
./scripts/install-watch-service.sh /path/to/your/posts

# 상태 확인
systemctl --user status watch-blog-frontmatter

# 중지
systemctl --user stop watch-blog-frontmatter

# 해제 (자동 시작 비활성화)
systemctl --user disable watch-blog-frontmatter
```

**요구사항**

- `pip install watchdog` (설치 스크립트가 자동으로 시도)
- WSL에서 systemd 사용 시 `/etc/wsl.conf`에 `[boot] systemd=true` 설정

로그는 `~/.logs/watch_frontmatter.log`에 기록됩니다.
