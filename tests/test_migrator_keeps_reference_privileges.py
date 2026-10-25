"""The migrator must keep the privileges Alembic uses to build constraints.

Provisioning revokes every table privilege and grants back a verb list.
Without REFERENCES and TRIGGER, the table owner cannot add a foreign key or
a trigger, which is how revision 142 failed on production.
"""

from control_plane.db_roles import SERVICE_ROLES, grant_statements


def test_migrator_grants_include_references_and_trigger():
    sql = "\n".join(
        grant_statements(SERVICE_ROLES["migrator"], existing_tables=["public.hosts"])
    )
    assert (
        'GRANT SELECT, INSERT, UPDATE, DELETE, REFERENCES, TRIGGER '
        'ON public."hosts" TO "xcelsior_migrator"'
    ) in sql


def test_runtime_roles_do_not_receive_ddl_privileges():
    sql = "\n".join(
        grant_statements(SERVICE_ROLES["api"], existing_tables=["public.hosts"])
    )
    assert "REFERENCES" not in sql
    assert "TRIGGER" not in sql
