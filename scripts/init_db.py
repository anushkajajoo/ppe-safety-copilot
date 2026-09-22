"""
Create the SQLite database and print the tables that exist.

Run:  python -m scripts.init_db
"""
from sqlalchemy import inspect

from server.config import get_settings
from server.database import init_db, make_engine


def main() -> None:
    settings = get_settings()
    engine = make_engine(settings.database_url)
    init_db(engine)
    tables = inspect(engine).get_table_names()
    print(f"Database: {settings.database_url}")
    print(f"Tables ({len(tables)}): {', '.join(sorted(tables))}")
    for t in sorted(tables):
        cols = [c["name"] for c in inspect(engine).get_columns(t)]
        print(f"  - {t}: {', '.join(cols)}")


if __name__ == "__main__":
    main()
