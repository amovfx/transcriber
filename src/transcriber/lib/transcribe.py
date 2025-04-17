import json
from pathlib import Path
from typing import Optional, Dict, Any

from mcp.server.fastmcp import FastMCP
from loguru import logger

from ..config.config import DEFAULT_LANGUAGE, SUPPORTED_FORMATS, SUPPORTED_LANGUAGES
from ..service.assemblyai import AssemblyAIService
from ..models.transcript import Transcript

transcriber_service = AssemblyAIService()


async def transcribe_audiofile(
    audio_path: str,
    language_code: str = DEFAULT_LANGUAGE,
) -> Dict[str, Any]:
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

    audio_file_path = Path(audio_path)
    json_output_path = str(audio_file_path.parent / "transcript.json")

    transcript = transcriber_service.transcribe_audio(audio_path, language_code)

    if json_output_path:
        try:
            # Ensure directory exists
            output_path = Path(json_output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Create a Transcript object
            transcript_obj = Transcript(
                text=transcript.text,
                words=[word.model_dump() for word in transcript.words],
            )

            # Save transcript to JSON file
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(
                    transcript_obj.model_dump(),
                    f,
                    indent=2,
                )

            logger.info(f"Transcription saved to {json_output_path}")
            return {
                "result": "Success",
                "json_output_path": json_output_path,
            }
        except Exception as e:
            msg = f"Failed to save transcription to {json_output_path}: {str(e)}"
            logger.error(msg)

            return {"result": "Failure", "error": msg}
