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
) -> Transcript:
    """Transcribe speech from an audio or video file using AssemblyAI.

    This function extracts speech from a media file and converts it to text, returning
    a Transcript object. File output is handled by the caller.

    Args:
        audio_path: Path to the audio or video file to transcribe
        language_code: Language code for transcription (default: "en")
            Supported languages: {', '.join(SUPPORTED_LANGUAGES.keys())}

    Returns:
        Transcript object containing the transcribed text and word data
    """
    transcript = transcriber_service.transcribe_audio(audio_path, language_code)
    transcript_obj = Transcript(
        text=transcript.text,
        words=[word.model_dump() for word in transcript.words],
    )
    return transcript_obj
