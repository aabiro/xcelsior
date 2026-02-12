#!/bin/bash
# Creates the xcelsior_test database for pytest.
# Mounted into /docker-entrypoint-initdb.d/ — runs once on first container start.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE xcelsior_test OWNER $POSTGRES_USER;
EOSQL

echo "✓ xcelsior_test database created"
