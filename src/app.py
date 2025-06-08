import gradio as gr
from utils import handle_upload  # đảm bảo utils.handle_upload có sẵn
from speech import *

def change_tab():
    return gr.Tabs(selected=1)

with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    with gr.Tabs() as tabs:
        with gr.TabItem("Upload CV & Job Description", id=0):
            page=0
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Please upload your CV and Job Description")
                    gr.Markdown("CV & Job Description must be in pdf format")

                    with gr.Row():
                        cv = gr.File(
                            label="Upload CV",
                            file_types=[".pdf"],
                            file_count="single",
                            interactive=True
                        )
                        jd = gr.File(
                            label="Upload JD",
                            file_types=[".pdf"],
                            file_count="single",
                            interactive=True
                        )

                    upload_btn = gr.Button(
                        "Get Feedback",
                        variant="primary",
                        size="lg",
                    )
                    status_md = gr.Markdown("", elem_id="status_panel")
                    with gr.Accordion("📝 Feedback & Suggestions", open=False):
                        feedback_md = gr.Markdown(
                            "Upload CV & JD and press Feedback Button to get valuable feedback to improve your CV.",
                            elem_id="feedback_panel"
                        )
                    cv_store = gr.State()
                    jd_store = gr.State()
                    upload_btn.click(
                        fn=handle_upload,
                        inputs=[cv, jd],
                        outputs=[status_md, feedback_md, cv_store, jd_store]
                    )
        with gr.TabItem("Interview Platform", id=1):
                gr.Markdown("<h2 style='text-align: center;'>🎤 Voice Interview</h2>")

                # State: current question index (starts at 0)
                question_idx = gr.State(0)
                # State: chat history seeded with the first question
                initial = [("", INITIAL_QUESTIONS[0])]
                chat_history = gr.State(initial)

                # Chatbot UI showing transcript & questions
                chatbot = gr.Chatbot(value=initial, label="Interview Transcript")

                with gr.Row():
                    
                    # Right column: voice controls
                    with gr.Column():
                        gr.Markdown("**Record your answer**")
                        audio_in = gr.Audio(sources="microphone", type="filepath", label="Record your answer")
                        speak_btn = gr.Button("Send Voice", variant="primary", size="md")
                        gr.Markdown("**Interviewer’s reply (audio)**")
                        audio_out = gr.Audio(interactive=False, label="Response Audio", autoplay=True)
                demo.load(
                    fn=lambda: synthesize_speech(INITIAL_QUESTIONS[0]),
                    inputs=None,
                    outputs=audio_out
                )
                # Hook up button: process voice, update chat, play next question
                speak_btn.click(
                    fn=voice_interaction,
                    inputs=[audio_in, chat_history, question_idx],
                    outputs=[chatbot, audio_out, chat_history, question_idx]
                )

    btn = gr.Button("Press this button to start the interview", variant="primary", size="lg", elem_id="start_btn")
    btn.click(change_tab, None, tabs)
demo.launch()

I used SQL to join orders and user tables to identify our top-spending customers. 
I struggled with cleaning up NULL values during ETL and with slow queries on millions of rows, 
so I added indexes and table partitions.