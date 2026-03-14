#!/usr/bin/env bash
# watch-blog-frontmatter systemd user service 설치
#
# 사용법: ./install-watch-service.sh <감시할 디렉토리 경로>
# 예: ./install-watch-service.sh /path/to/your/posts

if [ -z "$1" ]; then
    echo "사용법: $0 <감시할 디렉토리 경로>"
    echo "예: $0 /path/to/your/posts"
    exit 1
fi

BLOG_POSTS_DIR="$1"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVICE_NAME="watch-blog-frontmatter.service"
SERVICE_FILE="$SCRIPT_DIR/$SERVICE_NAME"
USER_SYSTEMD="$HOME/.config/systemd/user"

# watchdog 의존성 확인
if ! python3 -c "import watchdog" 2>/dev/null; then
    echo "watchdog 미설치. pip install watchdog 실행"
    pip install watchdog || { echo "실패: pip install watchdog 후 다시 시도하세요."; exit 1; }
fi

mkdir -p "$USER_SYSTEMD"
sed -e "s|__REPO_ROOT__|$REPO_ROOT|g" -e "s|__BLOG_POSTS_DIR__|$BLOG_POSTS_DIR|g" "$SERVICE_FILE" > "$USER_SYSTEMD/$SERVICE_NAME"
echo "서비스 파일 복사: $USER_SYSTEMD/$SERVICE_NAME (감시 경로: $BLOG_POSTS_DIR)"

systemctl --user daemon-reload
systemctl --user enable "$SERVICE_NAME"
systemctl --user start "$SERVICE_NAME"

echo ""
echo "등록 완료. 상태 확인: systemctl --user status $SERVICE_NAME"
echo "중지: systemctl --user stop $SERVICE_NAME"
echo "해제: systemctl --user disable $SERVICE_NAME"
