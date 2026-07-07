"""
voice.py
========
Implements Feature #13 - Voice Assistant (Text-to-Speech and
Speech-to-Text).

DESIGN DECISION - why these specific libraries:
  - Text-to-Speech: gTTS (Google Text-to-Speech). Produces natural-
    sounding audio via a simple free API, returned as MP3 bytes that
    Streamlit's `st.audio()` can play directly. Requires internet
    access at runtime (the same requirement as the LLM providers
    themselves, so it doesn't add a new constraint to the project).
  - Speech-to-Text: SpeechRecognition + Google's free Web Speech API
    backend. Takes a WAV audio clip and returns transcribed text.

HONEST LIMITATION (documented rather than hidden):
Browsers typically record audio as WebM/Opus, not WAV. Converting
between audio formats reliably needs `ffmpeg` installed on the host
machine (via `pydub`). This module assumes ffmpeg is available -
the README explains how to install it. If ffmpeg is missing,
`speech_to_text()` raises a clear, actionable error rather than
failing silently or producing garbage output.

Both functions are lazy-import (like ocr_engine.py and rag_engine.py)
so importing this module doesn't require these optional dependencies
to be installed unless voice features are actually used.
"""

import io
import logging

logger = logging.getLogger(__name__)


class VoiceError(Exception):
    """Raised when voice synthesis or recognition fails."""
    pass


# ------------------------------------------------------------
# Text-to-Speech
# ------------------------------------------------------------
_LANGUAGE_CODES = {
    "english": "en",
    "hindi": "hi",
    "urdu": "ur",
    "arabic": "ar",
}


def text_to_speech(text: str, language: str = "english") -> bytes:
    """
    Converts text into spoken audio (MP3 bytes).

    `language` should be one of MediLens's supported languages
    (Feature #14): english, hindi, urdu, arabic. Falls back to
    English if an unrecognized language name is passed.

    Returns raw MP3 bytes, ready to pass to Streamlit's
    `st.audio(audio_bytes, format="audio/mp3")`.
    """
    try:
        from gtts import gTTS
    except ImportError as exc:
        raise VoiceError(
            "The 'gtts' package is required for text-to-speech. Install it with: pip install gtts"
        ) from exc

    lang_code = _LANGUAGE_CODES.get(language.lower(), "en")

    # gTTS has a practical length limit per request; very long reports
    # are truncated with a note, rather than silently failing.
    max_chars = 3000
    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True

    try:
        tts = gTTS(text=text, lang=lang_code)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        audio_bytes = buffer.read()
    except Exception as exc:
        raise VoiceError(
            f"Text-to-speech failed (this feature requires internet access): {exc}"
        ) from exc

    if truncated:
        logger.info("Text truncated to %d characters for text-to-speech", max_chars)

    return audio_bytes


# ------------------------------------------------------------
# Speech-to-Text
# ------------------------------------------------------------
def speech_to_text(audio_bytes: bytes, source_format: str = "wav") -> str:
    """
    Transcribes spoken audio into text.

    `source_format` should match the actual format of `audio_bytes`
    (e.g. "wav", "webm", "ogg"). Non-WAV formats are converted using
    pydub, which requires ffmpeg to be installed on the host system.

    Raises VoiceError with a clear, actionable message if:
      - The required libraries aren't installed
      - ffmpeg is missing and format conversion is needed
      - The audio couldn't be understood (silence, noise, etc.)
    """
    try:
        import speech_recognition as sr
    except ImportError as exc:
        raise VoiceError(
            "The 'SpeechRecognition' package is required. Install it with: pip install SpeechRecognition"
        ) from exc

    wav_bytes = audio_bytes

    if source_format.lower() != "wav":
        try:
            from pydub import AudioSegment
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format=source_format)
            wav_buffer = io.BytesIO()
            audio_segment.export(wav_buffer, format="wav")
            wav_bytes = wav_buffer.getvalue()
        except Exception as exc:
            raise VoiceError(
                "Could not convert audio to WAV format. This usually means 'ffmpeg' is "
                "not installed on your system. Install it (e.g. 'apt install ffmpeg' on "
                f"Linux, 'brew install ffmpeg' on Mac). Details: {exc}"
            ) from exc

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError as exc:
        raise VoiceError("Could not understand the audio - please try speaking more clearly.") from exc
    except sr.RequestError as exc:
        raise VoiceError(f"Speech recognition service error (requires internet access): {exc}") from exc


def is_voice_available() -> dict:
    """
    Checks which voice dependencies are actually installed, so the
    frontend can show/hide voice UI elements gracefully instead of
    crashing when an optional dependency is missing.
    """
    result = {"text_to_speech": False, "speech_to_text": False}
    try:
        import gtts  # noqa: F401
        result["text_to_speech"] = True
    except ImportError:
        pass
    try:
        import speech_recognition  # noqa: F401
        result["speech_to_text"] = True
    except ImportError:
        pass
    return result
