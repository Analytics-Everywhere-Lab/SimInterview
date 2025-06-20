from datetime import datetime
import sys
import os
import torch

# Add the ditto-talkinghead directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ditto-talkinghead"))

from inference import StreamSDK, run

# Initialize SDK once
data_root = "ditto-talkinghead/checkpoints/ditto_trt_Ampere_Plus"
cfg_pkl = "ditto-talkinghead/checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl"
SDK = StreamSDK(cfg_pkl, data_root)


def audio_to_video(audio_path, source_path, output_path):
    """
    Convert audio and source image to video.

    :param audio_path: Path to the input audio file (.wav).
    :param source_path: Path to the source image or video file.
    :param output_path: Path to save the output video file (.mp4).
    :return: Path to the generated video file.
    """
    try:
        # Use the provided paths instead of hardcoded ones
        run(SDK, audio_path, source_path, output_path)
        return output_path
    except Exception as e:
        print(f"Error in audio_to_video: {e}")
        return None


def generate_interviewer_video(audio_file_path):
    """
    Generate interviewer video from audio response.
    """
    if audio_file_path is None:
        return None

    try:
        source_path = "ditto-talkinghead/example/interviewer.png"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "ditto-talkinghead/tmp"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"interviewer_{timestamp}.mp4")

        video_path = audio_to_video(audio_file_path, source_path, output_path)
        return video_path
    except Exception as e:
        print(f"Error generating video: {e}")
        return None
