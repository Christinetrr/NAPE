import whisper
import json
from pathlib import Path
from typing import Any
from openai import OpenAI


"""
extract transcript into text
map each word to timestamp
send text + timestamps to llm to parse and extract instructions/actions 
return instruction df
"""

def extract_transcript(video_path: str):
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    model = whisper.load_model("base")
    result = model.transcribe(str(path), word_timestamps=True)
    return result


def map_words_to_timestamps(transcript_result: dict) -> list[dict]:
    """Return words with start/end times: [{word, start, end}, ...]."""
    words: list[dict] = []
    for segment in transcript_result.get("segments", []):
        for word_info in segment.get("words", []):
            words.append(
                {
                    "word": word_info.get("word", "").strip(),
                    "start": float(word_info.get("start", 0.0)),
                    "end": float(word_info.get("end", 0.0)),
                }
            )
    return words

def transcription_analysis(word_mappings: list[Any]):
    prompt = Path("prompt.txt").read_text(encoding="utf-8")
    client = OpenAI()
    response = client.responses.create(
        #NEED TO CHANGE AND LOOK INTO API CREDS
        model="gpt-4.1-mini",
        input=[
            {
               "role": "system", 
               "content": prompt
            },
            {
                "role": "user",
                "content": json.dumps(word_mappings, ensure_ascii=False),
            },
        ],
    )
    return response.output_text

if __name__ == "__main__":
    transcript = extract_transcript(video_path="resize_character.mp4")
    word_spans = map_words_to_timestamps(transcript)
    print(word_spans[:10])
