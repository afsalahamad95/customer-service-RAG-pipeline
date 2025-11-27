"""Database connection and session management."""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from ..utils import load_config, get_logger
from ..utils.exceptions import DatabaseException, ConnectionException

logger = get_logger(__name__)


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self):
        self.config = load_config()
        self.db_config = self.config.get("database", {})
        self.engine = None
        self.SessionLocal = None
        self._initialize()

    def _initialize(self):
        """Initialize database engine and session factory."""
        try:
            # Build connection URL
            db_url = (
                f"postgresql://{self.db_config['user']}:{self.db_config['password']}"
                f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            )

            # Create engine with connection pooling
            self.engine = create_engine(
                db_url,
                poolclass=QueuePool,
                pool_size=self.db_config.get("pool_size", 10),
                max_overflow=self.db_config.get("max_overflow", 20),
                pool_pre_ping=True,  # Verify connections before using
                echo=self.db_config.get("echo", False),
            )

            # Create session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

            logger.info(
                f"Database connection initialized: {self.db_config['host']}:{self.db_config['port']}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise ConnectionException(f"Database connection failed: {e}")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions.

        Usage:
            with db_manager.get_session() as session:
                # Use session
                pass
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise DatabaseException(f"Database operation failed: {e}")
        finally:
            session.close()

    def execute_raw_sql(self, sql: str, params: dict = None):
        """Execute raw SQL query."""
        with self.get_session() as session:
            result = session.execute(sql, params or {})
            return result

    def health_check(self) -> bool:
        """Check if database connection is healthy."""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Global database manager instance
db_manager = DatabaseManager()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function for FastAPI.

    Usage:
        @app.get("/endpoint")
        def endpoint(db: Session = Depends(get_db)):
            # Use db session
            pass
    """
    with db_manager.get_session() as session:
        yield session
