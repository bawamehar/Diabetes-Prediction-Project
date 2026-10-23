import os

import psycopg2

# Connection settings come from environment variables - never hardcode credentials.
# DB_PASSWORD has no default on purpose.
DB_params = {
    "dbname" : os.environ.get("DB_NAME", "postgres"),
    "user" : os.environ.get("DB_USER", "postgres"),
    "password" : os.environ.get("DB_PASSWORD"),
    "host" : os.environ.get("DB_HOST", "localhost"),
    "port" : os.environ.get("DB_PORT", "5432")
}

TABLE_NAME = "diabetic_patients"

def insert_patient(stlmdata) -> bool:
    """Insert one record. Returns True on success, False if the database was unavailable."""
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(**DB_params)
        cursor = conn.cursor()

        create_query = f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                bmi FLOAT,
                age FLOAT,
                genhlth FLOAT,
                income FLOAT,
                highbp FLOAT,
                highchol FLOAT,
                education FLOAT
            );
            """
        cursor.execute(create_query)

        insert_query = f"""
        INSERT INTO {TABLE_NAME} (bmi, age, genhlth, income, highbp, highchol, education)
        VALUES (%s, %s, %s, %s, %s, %s, %s);
        """
        cursor.execute(insert_query, stlmdata)
        conn.commit()
        return True

    except Exception as e:
        # Log only the error type: messages from the driver can echo connection details.
        print("Database error:", type(e).__name__)
        return False

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
