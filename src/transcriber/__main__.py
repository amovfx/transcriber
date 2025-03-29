"""Transcriber Server MCP Service

This module implements an audio transcription service using AssemblyAI.
It provides functionality to transcribe an audio file and return the transcribed text.

The service is implemented as a FastMCP server that exposes a single tool:
- transcribe_file: Transcribes an audio file and returns its text content

Dependencies:
    - mcp: For FastMCP server implementation
    - transcriber API: For audio transcription functionality

Usage:
    Run this file directly to start the transcriber server:
    ```
    python transcriber_server.py
    ```
"""

import json
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP
from loguru import logger

from .config.config import DEFAULT_LANGUAGE, SUPPORTED_FORMATS, SUPPORTED_LANGUAGES
from .service.assemblyai import AssemblyAIService

# Initialize the FastMCP server and Transcriber service
mcp = FastMCP("transcriber")
transcriber_service = AssemblyAIService()


@mcp.tool(
    name="transcribe_file",
    description="Transcribe an audio file and return its text content. "
    "Supports multiple audio formats.",
)
async def transcribe_file(
    audio_path: str,
    language_code: str = DEFAULT_LANGUAGE,
    json_output_path: Optional[str] = None,
) -> str:
    """Transcribe an audio file using AssemblyAI.

    Args:
        audio_path: Path to the audio file
        language_code: Language code for transcription (default: "en")
            Supported languages: {', '.join(SUPPORTED_LANGUAGES.keys())}
        json_output_path: Optional path to save transcription result as JSON
                          If not provided, will save to same directory as audio_path
                          with filename "transcript.json"

    Returns:
        Transcribed text from the audio file
    """
    # If json_output_path is not specified, create it in the same directory as audio_path
    if json_output_path is None:
        audio_file_path = Path(audio_path)
        json_output_path = str(audio_file_path.parent / "transcript.json")
        logger.info(f"No output path specified, using: {json_output_path}")

    transcript = transcriber_service.transcribe_audio(audio_path, language_code)

    if json_output_path:
        try:
            # Ensure directory exists
            output_path = Path(json_output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save transcript to JSON file
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(
                    transcript,
                    f,
                    indent=2,
                )

            logger.info(f"Transcription saved to {json_output_path}")
        except Exception as e:
            logger.error(
                f"Failed to save transcription to {json_output_path}: {str(e)}"
            )

    return transcript


if __name__ == "__main__":  # pragma: no cover
    # Print service information
    print("Transcriber MCP Service")

    # Format supported languages
    languages = ", ".join(
        [f"{code} ({name})" for code, name in SUPPORTED_LANGUAGES.items()]
    )
    print(f"Supported languages: {languages}")

    print(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")

    # Run the MCP server
    mcp.run(transport="stdio")
