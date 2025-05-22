from pathlib import Path
from typing import List, Optional

from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from loguru import logger

# Get the project root directory (3 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

class AssemblyAI(BaseSettings):
    API_KEY: str
    API_KEY_2: str

    model_config = ConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="allow",
        env_prefix="ASSEMBLYAI_",
    )


ASSEMBLYAI = AssemblyAI()
logger.info(f"ASSEMBLYAI: {ASSEMBLYAI.API_KEY}...")
logger.info(f"ASSEMBLYAI: {ASSEMBLYAI.API_KEY_2}...")
logger.info(PROJECT_ROOT)

