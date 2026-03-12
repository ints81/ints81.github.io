#!/usr/bin/env bash
# 블로그 포스트 동기화 crontab 등록/해제 스크립트

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$(which python3)"
SYNC_SCRIPT="$SCRIPT_DIR/sync_posts.py"
LOG_FILE="$HOME/.logs/sync_posts.log"
CRON_JOB="*/5 * * * * $PYTHON $SYNC_SCRIPT >> $LOG_FILE 2>&1"
MARKER="sync_posts.py"

install() {
    if crontab -l 2>/dev/null | grep -q "$MARKER"; then
        echo "이미 등록되어 있습니다."
        crontab -l | grep "$MARKER"
        return
    fi

    (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
    echo "crontab 등록 완료 (5분 간격)"
    echo "  $CRON_JOB"
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

case "${1:-install}" in
    install)   install ;;
    uninstall) uninstall ;;
    status)    status ;;
    *)
        echo "사용법: $0 {install|uninstall|status}"
        exit 1
        ;;
esac
