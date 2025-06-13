import gradio as gr
from utils import handle_upload, generate_question_bank
from speech import *

def change_tab():
    return gr.Tabs(selected=1)

def init_chat(qbank):
    """
    Sau khi question_bank có giá trị (Python list), khởi tạo:
     - chat_ui:    [(user,msg),(user,msg)...] cho Chatbot
     - history:    copy của chat_ui để lưu lại
     - q_idx:      0
     - audio:      file audio câu hỏi đầu
    """
    if not qbank:
        # Nếu qbank trống, trả về defaults
        return [], [], 0, None

    first_q     = qbank[0]
    chat_ui     = [("", first_q)]
    audio_first = synthesize_speech(first_q)
    return chat_ui, chat_ui, 0, audio_first

with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    with gr.Tabs() as tabs:
        
        with gr.TabItem("Upload CV & Job Description", id=0):
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
                    with gr.Accordion("📝 Feedback & Suggestions", open=True):
                        feedback_md = gr.Markdown(
                            "Upload CV & JD and press Feedback Button to get valuable feedback to improve your CV.",
                            elem_id="feedback_panel"
                        )
                    cv_store = gr.State()
                    jd_store = gr.State()
                    question_bank = gr.State([])
                    chain = (upload_btn.click(
                        fn=handle_upload,
                        inputs=[cv, jd],
                        outputs=[status_md, feedback_md, cv_store, jd_store, question_bank],
                        show_progress='full'
                    ))

        with gr.TabItem("Interview Platform", id=1):
                gr.Markdown("<h2 style='text-align: center;'>🎤 Voice Interview</h2>")                      

                # State: current question index (starts at 0)
                question_idx = gr.State(0)
                # State: chat history seeded with the first question
                chat_history = gr.State([])
                # Chatbot UI showing transcript & questions
                chatbot = gr.Chatbot(label="Interview Transcript")
                audio_out = gr.Audio(interactive=False, label="Response Audio", autoplay=True)

                with gr.Row():       
                    # Right column: voice controls
                    with gr.Column():
                        gr.Markdown("**Record your answer**")
                        audio_in = gr.Audio(sources="microphone", type="filepath", label="Record your answer")
                        speak_btn = gr.Button("Send Voice", variant="primary", size="md")
                    with gr.Column():
                        gr.Markdown("**Interviewer’s reply (audio)**")
                        #audio_out = gr.Audio(interactive=False, label="Response Audio", autoplay=True)
                chain.then(fn=init_chat,
                    inputs=[question_bank],
                    outputs=[chatbot, chat_history, question_idx, audio_out]              # audio_out
                )    
                demo.load(
                    fn=lambda qbank: synthesize_speech(qbank[0]) if qbank else None,
                    inputs=[question_bank],
                    outputs=[audio_out]
                )
                # Hook up button: process voice, update chat, play next question
                speak_btn.click(
                    fn=voice_interaction,
                    inputs=[audio_in, chat_history, question_idx, question_bank],
                    outputs=[chatbot, audio_out, chat_history, question_idx]
                )

    btn = gr.Button("Press this button to start the interview", variant="primary", size="lg", elem_id="start_btn")
    btn.click(change_tab, None, tabs)
demo.launch()