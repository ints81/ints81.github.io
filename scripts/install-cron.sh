#!/usr/bin/env bash
# 블로그 포스트 동기화 crontab 등록/해제 스크립트

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$(which python3)"
SYNC_SCRIPT="$SCRIPT_DIR/sync_posts.py"
LOG_FILE="$HOME/.logs/sync_posts.log"
MARKER="sync_posts.py"

install() {
    local source_dir="$1"
    if [ -z "$source_dir" ]; then
        echo "사용법: $0 install <마크다운 디렉토리 경로>"
        exit 1
    fi

    if crontab -l 2>/dev/null | grep -q "$MARKER"; then
        echo "이미 등록되어 있습니다."
        crontab -l | grep "$MARKER"
        return
    fi

    local cron_job="*/5 * * * * $PYTHON $SYNC_SCRIPT $source_dir >> $LOG_FILE 2>&1"
    (crontab -l 2>/dev/null; echo "$cron_job") | crontab -
    echo "crontab 등록 완료 (5분 간격)"
    echo "  $cron_job"
}

uninstall() {
    if ! crontab -l 2>/dev/null | grep -q "$MARKER"; then
        echo "등록된 작업이 없습니다."
        return
    fi

    crontab -l | grep -v "$MARKER" | crontab -
    echo "crontab 해제 완료"
}

status() {
    if crontab -l 2>/dev/null | grep -q "$MARKER"; then
        echo "등록됨:"
        crontab -l | grep "$MARKER"
    else
        echo "등록되지 않음"
    fi
}

ACTION="${1:-install}"
shift 2>/dev/null || true

case "$ACTION" in
    install)   install "$1" ;;
    uninstall) uninstall ;;
    status)    status ;;
    *)
        echo "사용법: $0 {install <디렉토리>|uninstall|status}"
        exit 1
        ;;
esac
