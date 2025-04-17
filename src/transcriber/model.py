from pydantic import BaseModel, Field
from typing import List, Optional


class Word(BaseModel):
    """Representation of a single word in a transcript with timing information."""

    text: str = Field(description="The transcribed text of the word")
    start: int = Field(description="Start time of the word in milliseconds")
    end: int = Field(description="End time of the word in milliseconds")
    confidence: float = Field(
        default=1.0, description="Confidence score of the transcription (0.0-1.0)"
    )
    speaker: Optional[str] = Field(
        default=None, description="Speaker identifier if available"
    )
    channel: Optional[str] = Field(
        default=None, description="Audio channel identifier if available"
    )


class Transcript(BaseModel):
    """Complete transcript of an audio or video file with full text and timing information."""

    text: str = Field(description="The complete transcript text")
    words: List[Word] = Field(
        default_factory=list,
        description="List of individual words with timing information",
    )
