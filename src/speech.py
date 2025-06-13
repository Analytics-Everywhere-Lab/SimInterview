import os
import gradio as gr
from gtts import gTTS
from openai import OpenAI
from llm import get_llm_output
# Initialize OpenAI client for Whisper
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define a fixed system prompt for the interviewer persona
INTERVIEWER_SYSTEM_PROMPT = """
You are a professional technical interviewer. 
Your goal is to ask clear, relevant follow-up questions to evaluate the candidate's skills and experience.
Be concise, polite, and stay on topic.
"""

def transcribe_audio(audio_path):
    """speech to text."""
    with open(audio_path, "rb") as f:
        resp = openai.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    return getattr(resp, "text", "") or getattr(resp, "transcription", "")

def synthesize_speech(text, lang="en"):
    """text to speech."""
    tts = gTTS(text=text, lang=lang)
    out_path = "response.mp3"
    tts.save(out_path)
    return out_path

def voice_interaction(user_audio, history, q_idx, question_bank):
    # Nếu user chưa nói gì
    if not user_audio:
        return history, None, history, q_idx

    # STT
    user_text = transcribe_audio(user_audio)
    print("DEBUG: user_text =", user_text)

    # Next question
    next_idx = q_idx + 1
    if next_idx < len(question_bank):
        bot_text = question_bank[next_idx]
    else:
        bot_text = "🎉 Thank you! This concludes our interview."

    # TTS cho câu hỏi kế tiếp
    bot_audio = synthesize_speech(bot_text)

    # Update history: add (user_answer, bot_question)
    new_history = history + [(user_text, bot_text)]

    return new_history, bot_audio, new_history, next_idx


def interview(user_message, history):
    """
    user_message: str, câu vừa nói/vừa trả lời của ứng viên
    history: list of (str, str) tuples, danh sách các lượt (candidate, interviewer)
    
    Trả về: (new_history, new_history) để Gradio cập nhật Chatbot.
    """
    # 1. Build a single prompt that includes past conversation
    conversation = ""
    for u, b in history:
        conversation += f"Candidate: {u}\nInterviewer: {b}\n"
    conversation += f"Candidate: {user_message}\nInterviewer:"

    # 2. Call your LLM wrapper
    bot_reply = get_llm_output(
        temperature=0.7,
        max_tokens=500,
        system_role=INTERVIEWER_SYSTEM_PROMPT,
        prompt=conversation
    )

    # 3. Append to history
    new_history = history + [(user_message, bot_reply)]

    # Gradio Chatbot với Textbox thường dùng (history, history) làm outputs
    return new_history, new_history