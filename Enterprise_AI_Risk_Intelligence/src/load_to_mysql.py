import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

# MySQL configuration
MYSQL_USER = "ai_project"
MYSQL_PASSWORD = "AIProject@2026"
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
DATABASE = "enterprise_ai_risk"

# Build connection safely
connection_url = URL.create(
    drivername="mysql+pymysql",
    username=MYSQL_USER,
    password=MYSQL_PASSWORD,
    host=MYSQL_HOST,
    port=MYSQL_PORT,
    database=DATABASE
)

engine = create_engine(connection_url)

# Load CSV
df = pd.read_csv("data/ai_use_cases.csv")

print(f"Loaded {len(df)} records from CSV.")

# Load into MySQL
df.to_sql(
    "ai_use_cases",
    con=engine,
    if_exists="append",
    index=False
)

print("Successfully loaded data into MySQL!")