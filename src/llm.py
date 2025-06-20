import os
import gradio as gr
from gtts import gTTS
from langchain_openai import ChatOpenAI
from openai import OpenAI
from llm_parser import get_llm_output
from config import OPENAI_API_KEY
from video import generate_interviewer_video
# Initialize OpenAI client for Whisper
openai = OpenAI(api_key=OPENAI_API_KEY)


def transcribe_audio(audio_path):
    """speech to text."""
    with open(audio_path, "rb") as f:
        resp = openai.audio.transcriptions.create(model="whisper-1", file=f)
    return getattr(resp, "text", "") or getattr(resp, "transcription", "")


def synthesize_speech(text, lang="en"):
    """text to speech."""
    tts = gTTS(text=text, lang=lang)
    out_path = "response.mp3"
    tts.save(out_path)
    return out_path


class InterviewChatbot:
    def __init__(self):
        self.PROMPT_TEMPLATE = f"""
        You are a professional interviewer in the business domain. 
        Your goal is to ask clear, relevant follow-up questions to evaluate the candidate's skills and experience.
        Be concise, polite, and stay on topic.
        When all questions in the question bank are asked, you should end the conversation and thank the candidate for their time. Tell them that you will review their answers and get back to them soon.
        Here are possible questions based on their resume and the job description:
        """
        self.client = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
            temperature=0.7,
            max_tokens=4096,
        )

    def text_interaction(self, user_text, history, q_bank):
        self.question_bank = q_bank
        print("DEBUG: user_text =", user_text)
        response = self.generate_response(
            history, {"role": "user", "content": user_text}
        )
        bot_text = response.strip()
        print("DEBUG: bot_text =", bot_text)
        # TTS for the next question
        bot_audio = synthesize_speech(bot_text)
        # Update history: add (user_answer, bot_question)
        new_history = history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": bot_text},
        ]
            
        video_path = generate_interviewer_video(bot_audio)

        return "", new_history, video_path
    
    def video_interaction(self, user_video, history, q_bank):
        self.question_bank = q_bank
        user_text = transcribe_audio(user_video)
        print("DEBUG: user_text =", user_text)
        response = self.generate_response(
            history, {"role": "user", "content": user_text}
        )
        bot_text = response.strip()
        print("DEBUG: bot_text =", bot_text)
        # TTS for the next question
        bot_audio = synthesize_speech(bot_text)
        # Update history: add (user_answer, bot_question)
        new_history = history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": bot_text},
        ]
        video_path = generate_interviewer_video(bot_audio)

        return "", new_history, video_path


    def generate_response(self, history, new_message):
        """
        Generate a response from the LLM based on the chat history and new message.
        """
        messages = (
            [{"role": "system", "content": self.PROMPT_TEMPLATE + "\n".join(self.question_bank)}]
            + history
            + [new_message]
        )
        response = self.client.invoke(messages)
        return response.content if response else ""
