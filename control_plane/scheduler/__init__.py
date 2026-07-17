"""Transactional scheduler core (Track A, blueprint §10).

claim (Stage B) → filter (C) → score (D) → reserve/bind (E). Each stage
is a pure module; only claim and reservation touch the database, always
through short explicit transactions.
"""
