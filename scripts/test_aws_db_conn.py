import os
from sqlalchemy import create_engine, text

db_url = os.getenv("IDP_DATABASE_URL") or os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/idp")

print("Connecting to AWS RDS PostgreSQL...")
try:
    engine = create_engine(db_url, connect_args={"connect_timeout": 10})
    with engine.connect() as conn:
        res = conn.execute(text("SELECT version();")).fetchone()
        print("Connected successfully!")
        print("PostgreSQL Version:", res[0])
        tables = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")).fetchall()
        print("Existing tables in public schema:", [t[0] for t in tables])
except Exception as e:
    print("DB Connection error:", e)
