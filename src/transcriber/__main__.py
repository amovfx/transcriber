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
from typing import Optional, List, Dict, Any
import asyncio
import concurrent.futures

from mcp.server.fastmcp import FastMCP
from loguru import logger

from .lib.transcribe import transcribe_audiofile

from .config.config import DEFAULT_LANGUAGE, SUPPORTED_FORMATS, SUPPORTED_LANGUAGES
from .service.assemblyai import AssemblyAIService

# Initialize the FastMCP server and Transcriber service
mcp = FastMCP("transcriber")
transcriber_service = AssemblyAIService()



@mcp.tool(
    name="transcribe_file",
    description="Transcribe an audio or video file or process files in a directory recursively and return their text content. "
    "Supports multiple media formats.",
)
async def transcribe_file(
    input_path: str,
    language_code: str = DEFAULT_LANGUAGE,
    recursive: bool = False,
) -> Dict[str, Any]:
    """Transcribe speech from an audio or video file or process files in a directory using AssemblyAI.

    If input_path is a directory, finds supported media files (optionally recursively)
    and transcribes them in parallel, saving results as JSON files.
    If input_path is a file, transcribes that file and saves the result as JSON.

    Args:
        input_path: Path to the audio/video file or directory to transcribe.
        language_code: Language code for transcription (default: "en")
            Supported languages: {', '.join(SUPPORTED_LANGUAGES.keys())}
        recursive: Whether to search for files recursively in subdirectories if input_path is a directory.

    Returns:
        Dict with overall status and a list of results for each file processed.
        Each item in the list will contain file path, status (success/failure), and json_path or error.
    """
    path_obj = Path(input_path)
    files_to_process: List[Path] = []
    results: List[Dict[str, Any]] = []

    if path_obj.is_dir():
        # Use rglob for mp3 files
        if recursive:
            files_to_process.extend(path_obj.rglob("*.mp3"))
        else:
            files_to_process.extend(path_obj.glob("*.mp3"))

        # For other supported formats, use the existing glob pattern approach
        other_formats = [fmt for fmt in SUPPORTED_FORMATS if fmt != "mp3"]
        patterns = [f"*.{format}" for format in other_formats]
        if recursive:
            patterns = [f"**/{pattern}" for pattern in patterns]

        for pattern in patterns:
            files_to_process.extend(path_obj.glob(pattern))

        if not files_to_process:
            return {
                "status": "failure",
                "message": f"No supported media files found in {input_path}",
                "data": [],
            }

    elif path_obj.is_file():
        # Check if the single file has a supported format
        if path_obj.suffix.lstrip(".").lower() not in SUPPORTED_FORMATS:
            return {
                "status": "failure",
                "message": f"File {input_path} has an unsupported format.",
                "data": path_obj.suffix.lstrip(".").lower(),
            }
        files_to_process.append(path_obj)
    else:
        return {
            "status": "failure",
            "message": f"Invalid input_path: {input_path} is neither a file nor a directory.",
            "data": [],
        }

    # Create tasks for parallel transcription
    tasks = [
        transcribe_audiofile(str(file_path), language_code)
        for file_path in files_to_process
    ]

    # Run tasks concurrently, allowing exceptions to be returned
    transcription_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    overall_status = "success"
    for file_path, result in zip(files_to_process, transcription_results):
        if isinstance(result, Exception):
            overall_status = (
                "partial_success" if overall_status == "success" else overall_status
            )  # If any failed, overall is not pure success
            results.append(
                {"file": str(file_path), "status": "failure", "error": str(result)}
            )
        else:
            # Assuming transcribe_audiofile returns a transcript object with a model_dump method
            json_output_path = str(file_path.parent / "transcript.json")
            try:
                with open(json_output_path, "w", encoding="utf-8") as f:
                    json.dump(result.model_dump(), f, indent=2)
                results.append(
                    {
                        "file": str(file_path),
                        "status": "success",
                        "json_path": json_output_path,
                    }
                )
            except Exception as e:
                overall_status = (
                    "partial_success" if overall_status == "success" else overall_status
                )
                results.append(
                    {
                        "file": str(file_path),
                        "status": "failure",
                        "error": f"Failed to write JSON to {json_output_path}: {e}",
                    }
                )

    message = (
        "Transcription complete."
        if overall_status == "success"
        else "Transcription complete with some failures."
    )
    if overall_status == "failure":
        message = "Transcription failed for all files."

    return {"status": overall_status, "message": message, "data": results}


@mcp.tool()
async def read_transcript(json_path: str) -> Dict[str, Any]:
    """Load a previously saved transcription from a JSON file and return the text.

    Args:
        json_path: Path to the JSON file containing the transcription

    Returns:
        Dict containing the transcribed text or an error message.
    """
    try:
        # Load the JSON file
        with open(json_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)

        # Extract the text field
        if "text" in transcript_data:
            logger.info(f"Successfully loaded transcription from {json_path}")
            return {"status": "success", "text": transcript_data["text"]}
        else:
            msg = f"No 'text' field found in the JSON file at {json_path}"
            logger.error(msg)
            return {"status": "failure", "error": msg}

    except Exception as e:
        msg = f"Failed to load transcription from {json_path}: {str(e)}"
        logger.error(msg)
        return {"status": "failure", "error": msg}


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
