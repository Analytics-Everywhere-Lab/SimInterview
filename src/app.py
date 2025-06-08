import gradio as gr
from utils import handle_upload  # đảm bảo utils.handle_upload có sẵn



with gr.Blocks(title="Document Upload for RAG") as demo:
    with gr.Row():
        # Cột trái: nội dung chính (hiện tại chỉ để thông báo, bạn có thể bỏ hoặc điền tính năng khác)
        with gr.Column(scale=3):
            gr.Markdown("### Welcome to the Interview Prep System")
            gr.Markdown("Configure your interview settings here…")    
        # Cột phải: upload UI nhỏ gọn
        with gr.Column(scale=1):
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

demo.launch()
