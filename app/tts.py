import io

from gtts import gTTS  # type: ignore


def text_to_speech_gtts(text: str, lang: str = "ru") -> bytes:
    tts = gTTS(text=text, lang=lang, slow=False)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer.read()
