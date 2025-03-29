from pathlib import Path
from typing import List, Optional

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class AssemblyAI(BaseSettings):
    API_KEY: str

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="allow",
        env_prefix="ASSEMBLYAI_",
    )


ASSEMBLYAI = AssemblyAI()
