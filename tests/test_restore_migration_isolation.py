"""Restore migrations cannot inherit a production Compose database override."""

from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock

import psycopg
from psycopg import sql
import pytest
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT/'api/scripts/migrate_restore_database.py'
spec = importlib.util.spec_from_file_location('restore_migration_guard', HELPER)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
TARGET = 'xcelsior_restore_20261020_200000_12345'


@pytest.mark.parametrize('target',['xcelsior','postgres','xcelsior_restore_prod',TARGET+';DROP DATABASE xcelsior'])
def test_restore_guard_rejects_unsafe_database_names_before_connecting(target, monkeypatch):
    connect=mock.Mock(side_effect=AssertionError('must not connect'))
    monkeypatch.setattr(guard.psycopg,'connect',connect)
    with pytest.raises(ValueError):
        guard.migrate(target)
    connect.assert_not_called()


def test_restore_guard_rewrites_all_dsns_and_refuses_an_unexpected_connected_database(monkeypatch):
    conn=mock.MagicMock()
    conn.__enter__.return_value.execute.return_value.fetchone.return_value=('xcelsior',)
    connect=mock.Mock(return_value=conn)
    monkeypatch.setattr(guard.psycopg,'connect',connect)
    from alembic import command
    upgrade=mock.Mock(side_effect=AssertionError('must not migrate production'))
    monkeypatch.setattr(command,'upgrade',upgrade)
    with mock.patch.dict(os.environ,{'XCELSIOR_POSTGRES_DSN':'postgresql://test:test@localhost/xcelsior?dbname=xcelsior','DATABASE_URL':'postgresql://test@localhost/xcelsior','XCELSIOR_POSTGRES_DSN_API':'postgresql://test@localhost/xcelsior'}):
        with pytest.raises(RuntimeError,match='isolated restore target'):
            guard.migrate(TARGET)
        assert make_url(connect.call_args.args[0]).database==TARGET
        assert 'dbname' not in make_url(connect.call_args.args[0]).query
        assert all(make_url(os.environ[key]).database==TARGET for key in ['XCELSIOR_POSTGRES_DSN','DATABASE_URL','XCELSIOR_POSTGRES_DSN_API'])
        assert os.environ['PYTHON_DOTENV_DISABLED']=='1'
    upgrade.assert_not_called()


def test_real_restore_migration_uses_new_database_and_leaves_source_schema_unchanged():
    base=os.environ['XCELSIOR_POSTGRES_DSN']
    target='xcelsior_restore_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())
    with psycopg.connect(base,autocommit=True) as source:
        primary=source.execute('SELECT current_database()').fetchone()[0]
        before=source.execute('SELECT version_num FROM alembic_version').fetchall()
        source.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(target)))
        try:
            env=os.environ.copy()
            # Reproduce Compose pinning all connection variables to the source.
            env.update(XCELSIOR_POSTGRES_DSN=base,DATABASE_URL=base,XCELSIOR_POSTGRES_DSN_API=base)
            result=subprocess.run([sys.executable,str(HELPER),target],cwd=ROOT,env=env,capture_output=True,text=True,timeout=45)
            assert result.returncode==0,result.stdout+result.stderr
            assert source.execute('SELECT current_database()').fetchone()[0]==primary
            assert source.execute('SELECT version_num FROM alembic_version').fetchall()==before
            restored=make_url(base).set(database=target).render_as_string(hide_password=False)
            with psycopg.connect(restored) as check:
                assert check.execute('SELECT version_num FROM alembic_version').fetchall()==before
        finally:
            source.execute(sql.SQL('DROP DATABASE {} WITH (FORCE)').format(sql.Identifier(target)))
