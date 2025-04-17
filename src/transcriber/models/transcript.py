from pydantic import BaseModel, Field
from typing import List, Optional
import json
from pathlib import Path


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

    @classmethod
    def from_file(cls, file_path: str) -> "Transcript":
        """Load a transcript from a JSON file.

        Args:
            file_path: Path to the transcript.json file

        Returns:
            Transcript object with the loaded data

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file cannot be parsed as valid JSON or doesn't have the expected structure
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Transcript file not found: {file_path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Validate required fields
            if "text" not in data:
                raise ValueError(
                    f"Missing required field 'text' in transcript file: {file_path}"
                )
            if "words" not in data:
                raise ValueError(
                    f"Missing required field 'words' in transcript file: {file_path}"
                )

            # Create Word objects from the raw data
            words = [Word.model_validate(word) for word in data["words"]]

            return cls(text=data["text"], words=words)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in transcript file: {file_path}")
        except Exception as e:
            raise ValueError(f"Error parsing transcript file {file_path}: {str(e)}")
