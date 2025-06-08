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

            # feedback = gr.Textbox(
            #     label="Feedback and Suggestion",
            #     placeholder="Your tailored feedback will appear here after analysis…",
            #     interactive=False,
            #     lines=6
            # )
            # upload_btn.click(
            #     fn=handle_upload,
            #     inputs=[cv, jd],
            #     outputs=[feedback]
            # )
            with gr.Accordion("📝 Feedback & Suggestions", open=False):
                feedback_md = gr.Markdown(
                    "Upload CV & JD rồi nhấn **Get Feedback** để xem gợi ý cải thiện.",
                    elem_id="feedback_panel"
                )

            upload_btn.click(
                fn=handle_upload,
                inputs=[cv, jd],
                outputs=[feedback_md]
            )

demo.launch()
