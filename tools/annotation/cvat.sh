#!/usr/bin/env bash
# Bring the CVAT annotation stack up the same way every time, and get its data back out.
#
#   ./cvat.sh up                 pin, pull and start
#   ./cvat.sh admin paul         create the CVAT superuser (the one-time blocker)
#   ./cvat.sh backup             consistent DB + media snapshot, optionally to GCS
#   ./cvat.sh status | logs | down
#
# The instance was first stood up by hand from a clone of CVAT's `develop` branch on
# `cvat/server:dev`. That works right up until something restarts on a different commit.
# This script exists so the deployment is a checked-out tag plus an override file, and so
# the two cannot drift apart: `up` refuses to start a checkout that does not match
# CVAT_VERSION in cvat.env.
#
# Everything is plain docker compose underneath. Nothing here is CVAT-version-specific
# except the tag, which is the point.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${CVAT_SRC:-/opt/cvat/src}"          # upstream checkout, never edited
REPO="${CVAT_REPO:-https://github.com/cvat-ai/cvat.git}"
BACKUP_DIR="${CVAT_BACKUP_DIR:-$HERE/backups}"
GCS_PREFIX="${CVAT_BACKUP_GCS:-gs://syncai-hydranet/annotation-backups}"

# shellcheck source=cvat.env
set -a; . "$HERE/cvat.env"; set +a

DOCKER=(docker)
docker info >/dev/null 2>&1 || DOCKER=(sudo docker)

die() { echo "error: $*" >&2; exit 1; }

compose() {
  "${DOCKER[@]}" compose \
    --env-file "$HERE/cvat.env" \
    -f "$SRC/docker-compose.yml" \
    -f "$HERE/docker-compose.override.yml" "$@"
}

# ---------------------------------------------------------------- the pinned checkout

sync_source() {
  if [ ! -d "$SRC/.git" ]; then
    echo "cloning $REPO at $CVAT_VERSION into $SRC"
    sudo mkdir -p "$(dirname "$SRC")"
    sudo git clone --depth 1 --branch "$CVAT_VERSION" "$REPO" "$SRC"
    return
  fi
  local dirty
  dirty="$(sudo git -C "$SRC" status --porcelain)"
  [ -z "$dirty" ] || die "$SRC has local edits. Upstream's compose file is not ours to
  change -- put local policy in docker-compose.override.yml, then re-run.
$dirty"
  local at
  at="$(sudo git -C "$SRC" describe --tags --exact-match 2>/dev/null || echo none)"
  if [ "$at" != "$CVAT_VERSION" ]; then
    echo "checkout is at ${at}, cvat.env pins $CVAT_VERSION -- moving it"
    echo "back up first if this stack holds annotations: ./cvat.sh backup" >&2
    sudo git -C "$SRC" fetch --depth 1 origin "refs/tags/$CVAT_VERSION:refs/tags/$CVAT_VERSION"
    sudo git -C "$SRC" checkout --detach "$CVAT_VERSION"
  fi
}

# --------------------------------------------------------------------------- commands

cmd_up() {
  sync_source
  compose pull --quiet
  compose up -d "$@"
  echo
  echo "CVAT $CVAT_VERSION is starting; the API answers a minute or two after the containers do."
  echo "It listens on 127.0.0.1:8080 only. From your workstation:"
  echo "  gcloud compute ssh hydranet-annotation --zone=asia-east1-b \\"
  echo "      --tunnel-through-iap -- -L 8080:localhost:8080"
  echo "then open http://localhost:8080"
}

cmd_down()   { compose down "$@"; }
cmd_logs()   { compose logs --tail 100 -f "$@"; }
cmd_status() {
  compose ps
  echo
  echo "pinned:   $CVAT_VERSION"
  echo "checkout: $(sudo git -C "$SRC" describe --tags --exact-match 2>/dev/null || echo 'not a tag')"
}

# CVAT ships no admin account, and creating one interactively is the step every
# bring-up forgets. Annotator accounts are made from the admin UI afterwards.
cmd_admin() {
  local user="${1:-}" email="${2:-}"
  [ -n "$user" ] || die "usage: ./cvat.sh admin <username> [email]"
  local pass="${CVAT_ADMIN_PASSWORD:-}"
  if [ -z "$pass" ]; then
    read -rsp "password for CVAT user '$user': " pass; echo
    local again
    read -rsp "again: " again; echo
    [ "$pass" = "$again" ] || die "passwords differ"
  fi
  [ ${#pass} -ge 12 ] || die "use at least 12 characters; store it in a password manager"

  # Passed as environment, never on the command line: docker inspect and the process
  # table are both readable by anyone who can reach the daemon.
  compose exec -T \
    -e DJANGO_SUPERUSER_USERNAME="$user" \
    -e DJANGO_SUPERUSER_EMAIL="${email:-$user@syncrobotic.local}" \
    -e DJANGO_SUPERUSER_PASSWORD="$pass" \
    cvat_server bash -ic 'python3 ~/manage.py createsuperuser --noinput'
  echo "created superuser '$user'. Add annotators from http://localhost:8080/admin"
}

# A backup that runs while jobs are being saved can catch a task whose media and whose
# database row disagree. Stopping the app containers for the duration costs a minute and
# removes the question; the database stays up so pg_dump has something to talk to.
cmd_backup() {
  local online=0
  if [ "${1:-}" = "--online" ]; then online=1; fi
  local stamp dest
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  dest="$BACKUP_DIR/$stamp"
  mkdir -p "$dest"

  local app=(cvat_server cvat_ui cvat_worker_annotation cvat_worker_import
             cvat_worker_export cvat_worker_webhooks cvat_worker_quality_reports
             cvat_worker_chunks cvat_worker_consensus cvat_worker_utils)
  if [ "$online" = 0 ]; then
    echo "stopping the app containers for a consistent snapshot"
    compose stop "${app[@]}"
  else
    echo "WARNING: --online, so media and database rows may disagree for tasks being edited"
  fi

  echo "dumping the database"
  compose exec -T cvat_db pg_dump -U root -d cvat | gzip > "$dest/cvat_db.sql.gz"

  for vol in cvat_data cvat_keys; do
    echo "archiving volume $vol"
    "${DOCKER[@]}" run --rm -v "cvat_${vol}:/from:ro" -v "$dest:/to" alpine:3.20 \
      tar czf "/to/${vol}.tgz" -C /from .
  done

  {
    echo "cvat_version=$CVAT_VERSION"
    echo "taken=$stamp"
    echo "online=$online"
  } > "$dest/MANIFEST"

  if [ "$online" = 0 ]; then compose start "${app[@]}"; fi

  echo "wrote $dest ($(du -sh "$dest" | cut -f1))"
  if command -v gcloud >/dev/null 2>&1 && [ -n "$GCS_PREFIX" ]; then
    echo "uploading to $GCS_PREFIX/$stamp/"
    gcloud storage cp -r "$dest" "$GCS_PREFIX/$stamp/" \
      || echo "upload failed; the local copy in $dest is still good" >&2
  fi
  echo "restore procedure: git show b7457c2:docs/METHODOLOGY.md"
}

case "${1:-}" in
  up)      shift; cmd_up "$@" ;;
  down)    shift; cmd_down "$@" ;;
  logs)    shift; cmd_logs "$@" ;;
  status)  shift; cmd_status "$@" ;;
  admin)   shift; cmd_admin "$@" ;;
  backup)  shift; cmd_backup "$@" ;;
  *) sed -n '2,9p' "${BASH_SOURCE[0]}" | cut -c3-; exit 1 ;;
esac
