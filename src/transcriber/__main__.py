"""Transcriber Server MCP Service

This module implements a speech-to-text transcription service for audio and video files using AssemblyAI.
It automatically extracts and transcribes speech from various audio and video formats, creating
human-readable text transcripts from the content.

The service automatically saves transcript files in JSON format alongside the original media file
(or at a specified location), making it easy to access the transcribed content.

The service is implemented as a FastMCP server that exposes a single tool:
- transcribe_file: Transcribes audio/video files and saves transcripts as JSON

Supported formats include:
- Audio: MP3, WAV, M4A, FLAC, etc.
- Video: MP4, MOV, AVI, etc.

Dependencies:
    - mcp: For FastMCP server implementation
    - assemblyai: For speech-to-text transcription functionality

Usage:
    Run this file directly to start the transcriber server:
    ```
    python -m transcriber
    ```
"""

import json
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP
from loguru import logger

from .lib.transcribe import transcribe_audiofile

from .config.config import DEFAULT_LANGUAGE, SUPPORTED_FORMATS, SUPPORTED_LANGUAGES
from .service.assemblyai import AssemblyAIService

# Add import for write_json_safe
from src.utils.json_files import write_json_safe

# Initialize the FastMCP server and Transcriber service
mcp = FastMCP("transcriber")
transcriber_service = AssemblyAIService()


@mcp.tool(
    name="transcribe_file",
    description="Transcribe an audio or video file and return its text content. "
    "Supports multiple media formats.",
)
async def transcribe_file(
    audio_path: str,
    language_code: str = DEFAULT_LANGUAGE,
) -> dict:
    """Transcribe speech from an audio or video file using AssemblyAI.

    This function extracts speech from a media file and converts it to text, saving
    the transcription results as a JSON file. The transcript is automatically saved
    alongside the original file if no output path is specified.

    Args:
        audio_path: Path to the audio or video file to transcribe
        language_code: Language code for transcription (default: "en")
            Supported languages: {', '.join(SUPPORTED_LANGUAGES.keys())}
        json_output_path: Optional path to save transcription result as JSON
                          If not provided, will save to same directory as audio_path
                          with filename "transcript.json"

    Returns:
        Dict with result, json_output_path, and error if failed
    """
    try:
        transcript_obj = await transcribe_audiofile(audio_path, language_code)
        audio_file_path = Path(audio_path)
        json_output_path = str(audio_file_path.parent / "transcript.json")
        # Save using write_json_safe
        success = write_json_safe(
            json_output_path, transcript_obj.model_dump(), indent=2
        )
        if success:
            return {"result": "Success", "json_output_path": json_output_path}
        else:
            return {
                "result": "Failure",
                "error": f"Failed to write JSON to {json_output_path}",
            }
    except Exception as e:
        return {"result": "Failure", "error": str(e)}


@mcp.tool()
async def read_transcript(json_path: str) -> str:
    """Load a previously saved transcription from a JSON file and return the text.

    Args:
        json_path: Path to the JSON file containing the transcription

    Returns:
        The transcribed text from the JSON file
    """
    try:
        # Load the JSON file
        with open(json_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)

        # Extract the text field
        if "text" in transcript_data:
            logger.info(f"Successfully loaded transcription from {json_path}")
            return transcript_data["text"]
        else:
            msg = f"No 'text' field found in the JSON file at {json_path}"
            logger.error(msg)
            return {"result": "Failure", "error": msg}

    except Exception as e:
        msg = f"Failed to load transcription from {json_path}: {str(e)}"
        logger.error(msg)
        return {"result": "Failure", "error": msg}


def main():
    """Main entry point for the transcriber service when run as a script."""
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


if __name__ == "__main__":
    main()
