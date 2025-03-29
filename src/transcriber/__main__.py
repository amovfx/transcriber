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

from .config.config import DEFAULT_LANGUAGE, SUPPORTED_FORMATS, SUPPORTED_LANGUAGES
from .service.assemblyai import AssemblyAIService

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
    json_output_path: Optional[str] = None,
) -> str:
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
        Transcribed text from the audio/video file
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

            json_data = {
                "words": [word.model_dump() for word in transcript.words],
                "sentences": [
                    sentence.model_dump() for sentence in transcript.get_sentences()
                ],
                "paragraphs": [
                    paragraph.model_dump() for paragraph in transcript.get_paragraphs()
                ],
            }

            # Save transcript to JSON file
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(
                    json_data,
                    f,
                    indent=2,
                )

            logger.info(f"Transcription saved to {json_output_path}")
            return {"result": "Success", "result": transcript.text}
        except Exception as e:
            msg = f"Failed to save transcription to {json_output_path}: {str(e)}"
            logger.error(msg)

            return {"result": "Failure", "result": msg}


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
