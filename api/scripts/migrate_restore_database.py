#!/usr/bin/env python3
"""Pin every migration connection to a verified disposable restore database."""

from __future__ import annotations

import os
import re
import sys

import psycopg
from sqlalchemy.engine import make_url


def migrate(target: str) -> None:
    if not re.fullmatch(r'xcelsior_restore_[0-9]{8}_[0-9]{6}_[0-9]+', target):
        raise ValueError('refusing a database outside the generated restore namespace')
    base = next((os.environ[key] for key in ('XCELSIOR_POSTGRES_DSN_MIGRATOR','XCELSIOR_POSTGRES_DSN','XCELSIOR_PG_DSN','DATABASE_URL') if os.environ.get(key)), None)
    if not base:
        raise ValueError('the application connection is required to locate the restore server')
    url = make_url(base)
    if url.get_backend_name() != 'postgresql':
        raise ValueError('restore migrations require PostgreSQL')
    # libpq accepts dbname in URI query parameters, overriding the URI path.
    # Remove that alias before constructing either verified connection.
    url = url.difference_update_query(['dbname','database']).set(database=target)
    dsn = url.set(drivername='postgresql').render_as_string(hide_password=False)
    os.environ['PYTHON_DOTENV_DISABLED'] = '1'
    os.environ['XCELSIOR_SKIP_DOTENV'] = '1'
    os.environ['XCELSIOR_BG_TASKS'] = 'false'
    for key in list(os.environ):
        if key.startswith('XCELSIOR_POSTGRES_DSN_'):
            os.environ[key] = dsn
    for key in ('XCELSIOR_POSTGRES_DSN','XCELSIOR_PG_DSN','DATABASE_URL'):
        os.environ[key] = dsn
    with psycopg.connect(dsn, connect_timeout=10) as conn:
        observed = conn.execute('SELECT current_database()').fetchone()[0]
        if observed != target:
            raise RuntimeError('connected database is not the isolated restore target')
    from alembic import command
    from alembic.config import Config

    command.upgrade(Config('alembic.ini'), 'head')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('one generated restore database name is required')
    migrate(sys.argv[1])
