"""
database/db.py
==============
Engine, session factory, and a self-healing schema.

Two practical details:

1. WAL mode. The pipeline writes while the dashboard reads, from two
   separate processes. WAL lets that happen without "database is locked".

2. Auto-migration. SQLAlchemy's create_all() only creates missing TABLES,
   never missing COLUMNS. So when a new field is added to a model, an
   existing .db file would break. init_db() compares the models against
   the real database and adds anything missing - no manual migration,
   no deleting your data.
"""
import os
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base

from config.settings import DB_URL, DB_PATH

Base = declarative_base()

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

engine = create_engine(
    DB_URL,
    connect_args={"check_same_thread": False, "timeout": 20},
    future=True,
)


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_conn, _record):
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA busy_timeout=20000;")
    cur.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False,
                            expire_on_commit=False, future=True)


def _sqlite_type(column):
    t = str(column.type).upper()
    if "BLOB" in t or "BINARY" in t:
        return "BLOB"          # embeddings; TEXT would corrupt them
    if "INT" in t:
        return "INTEGER"
    if any(k in t for k in ("REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL")):
        return "REAL"
    if "BOOL" in t:
        return "INTEGER"
    return "TEXT"


def _auto_migrate():
    """Add any column that exists on a model but not in the database."""
    insp = inspect(engine)
    existing = set(insp.get_table_names())
    added = []
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing:
                continue
            have = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in have:
                    continue
                ddl = (f'ALTER TABLE "{table.name}" ADD COLUMN '
                       f'"{col.name}" {_sqlite_type(col)}')
                default = getattr(col.default, "arg", None)
                if default is not None and not callable(default):
                    if isinstance(default, bool):
                        ddl += f" DEFAULT {1 if default else 0}"
                    elif isinstance(default, str):
                        ddl += f" DEFAULT '{default}'"
                    else:
                        ddl += f" DEFAULT {default}"
                conn.execute(text(ddl))
                added.append(f"{table.name}.{col.name}")
    for a in added:
        print(f"[DB] migrated: added column {a}")


def init_db():
    from database import models  # noqa: F401  registers the tables
    Base.metadata.create_all(engine)
    _auto_migrate()
