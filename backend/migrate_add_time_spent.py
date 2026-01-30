#!/usr/bin/env python3
"""
Migration script to add time_spent column to feedback table
"""
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("URL_DOCKER", "postgresql://nuria:123456@localhost:5434/nuria_extras")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

def migrate():
    engine = create_engine(DATABASE_URL)
    
    try:
        with engine.connect() as conn:
            # Check if column already exists
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='feedback' AND column_name='time_spent'
            """))
            
            if result.fetchone():
                print("✅ Column 'time_spent' already exists!")
                return
            
            # Add the column
            print("Adding 'time_spent' column to feedback table...")
            conn.execute(text("""
                ALTER TABLE feedback 
                ADD COLUMN time_spent INTEGER NOT NULL DEFAULT 0
            """))
            conn.commit()
            
            print("✅ Migration completed successfully!")
            print("You can now restart the backend.")
            
    except Exception as e:
        print(f"❌ Error during migration: {e}")
        raise

if __name__ == "__main__":
    migrate()
