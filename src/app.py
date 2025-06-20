import gradio as gr
from utils import handle_upload
from speech import *
from video import generate_interviewer_video
from llm import InterviewChatbot


def change_tab():
    return gr.Tabs(selected=1)


def init_chat_with_video(qbank):
    """
    Enhanced init_chat that also generates video.
    """
    if not qbank:
        return [], [], 0, None

    first_q = qbank[0]
    chat_ui = [("", first_q)]
    audio_first = synthesize_speech(first_q)
    video_first = generate_interviewer_video(audio_first)

    return chat_ui, chat_ui, 0, video_first


def voice_interaction_with_video(
    video_input, chat_history, question_idx, question_bank
):
    """
    Simplified voice_interaction that directly uses the video file.
    """
    if video_input is None:
        return chat_history, chat_history, question_idx, None

    updated_chatbot, audio_response_path, updated_chat_history, updated_question_idx = (
        voice_interaction(video_input, chat_history, question_idx, question_bank)
    )

    # Generate video response
    video_path = generate_interviewer_video(audio_response_path)

    return (
        updated_chatbot,
        updated_chat_history,
        updated_question_idx,
        video_path,
    )


def load_first_question_with_video(qbank):
    """
    Load first question and generate video.
    """
    if not qbank:
        return None

    audio_path = synthesize_speech(qbank[0])
    video_path = generate_interviewer_video(audio_path)
    return video_path


def main():
    with gr.Blocks(theme=gr.themes.Ocean()) as demo:
        with gr.Tabs() as tabs:
            with gr.TabItem("Upload CV & Job Description", id=0):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Please upload your CV and Job Description")
                        gr.Markdown("CV & Job Description must be in pdf format")
                        with gr.Group():
                            with gr.Row():
                                cv = gr.File(
                                    label="Upload CV",
                                    file_types=[".pdf"],
                                    file_count="single",
                                    interactive=True,
                                )
                                jd = gr.File(
                                    label="Upload JD",
                                    file_types=[".pdf"],
                                    file_count="single",
                                    interactive=True,
                                )
                            with gr.Row():
                                gr.Examples(
                                    examples=[["sample_resume.pdf", "sample_jd.pdf"]],
                                    inputs=[cv, jd],
                                )

                        upload_btn = gr.Button(
                            "Get Feedback",
                            variant="primary",
                            size="lg",
                        )
                        status_md = gr.Markdown("", elem_id="status_panel")
                        with gr.Accordion("📝 Feedback & Suggestions", open=True):
                            feedback_md = gr.Markdown(
                                "Upload CV & JD and press Feedback Button to get valuable feedback to improve your CV.",
                                elem_id="feedback_panel",
                            )
                        cv_store = gr.State()
                        jd_store = gr.State()
                        question_bank = gr.State([])
                        upload_btn.click(
                            fn=handle_upload,
                            inputs=[cv, jd],
                            outputs=[
                                status_md,
                                feedback_md,
                                cv_store,
                                jd_store,
                                question_bank,
                            ],
                            show_progress="full",
                        )
                btn = gr.Button(
                    "Press this button to start the interview",
                    variant="primary",
                    size="lg",
                    elem_id="start_btn",
                )
                btn.click(change_tab, None, tabs)

            with gr.TabItem("Interview Platform", id=1):
                interview_chatbot = InterviewChatbot()
                gr.Markdown("<h2 style='text-align: center;'>🎤 Voice Interview</h2>")

                with gr.Group():
                    with gr.Row():
                        # Video component with audio included
                        interviewee_video = gr.Video(
                            label="Interviewee", sources="webcam", include_audio=True
                        )
                        interviewer_video = gr.Video(label="Interviewer", autoplay=True)

                    chatbot = gr.Chatbot(
                        type="messages",
                        label="Interview Transcript",
                        show_copy_button=True,
                        show_share_button=True,
                        resizable=True,
                        render_markdown=True,
                        min_height=200,
                    )
                    msg = gr.Textbox(
                        placeholder="Type your message here...",
                        submit_btn=True,
                        show_label=False,
                    )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Record your video answer**")
                        speak_btn = gr.Button(
                            "Send Video Response", variant="primary", size="md"
                        )
                with gr.Row():
                    toggle_dark = gr.Button(value="Toggle Dark")
                toggle_dark.click(
                    None,
                    js="""
                    () => {
                        document.body.classList.toggle('dark');
                    }
                    """,
                )

                msg.submit(
                    fn=interview_chatbot.text_interaction,
                    inputs=[msg, chatbot, question_bank],
                    outputs=[msg, chatbot, interviewer_video],
                )

                speak_btn.click(
                    fn=interview_chatbot.video_interaction,
                    inputs=[interviewee_video, chatbot, question_bank],
                    outputs=[msg, chatbot, interviewer_video],
                )

        demo.launch()


if __name__ == "__main__":
    main()
