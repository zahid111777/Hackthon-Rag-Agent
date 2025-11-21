import os
import re
import logging
import tempfile
from pathlib import Path
from typing import List
import numpy as np
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import gradio as gr
from gtts import gTTS

# Safe Groq import
try:
    from groq import Groq
    GROQ_OK = True
except ImportError:
    GROQ_OK = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgenticRAGAgent:
    def __init__(self):
        self.chunks = []
        self.index = None
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Perfect Groq key detection - no more "missing" error
        self.groq = None
        raw_key = os.getenv("GROQ_API_KEY")
        if raw_key and GROQ_OK:
            key = raw_key.strip()
            if key:
                try:
                    self.groq = Groq(api_key=key)
                    logger.info("Groq API key loaded and working!")
                except Exception as e:
                    logger.error(f"Groq init error: {e}")

    # Remove emojis completely from voice
    def remove_emojis(self, text: str) -> str:
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def clean_for_voice(self, text: str) -> str:
        text = self.remove_emojis(text)
        text = re.sub(r'[\*_`#\[\]]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def generate_voice(self, text: str):
        if not text or not text.strip():
            return None
        clean = self.clean_for_voice(text)
        if len(clean) < 5:
            return None
        try:
            tts = gTTS(text=clean, lang='en')
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(tmp.name)
            return tmp.name
        except Exception as e:
            logger.error(f"Voice generation failed: {e}")
            return None

    def upload_pdfs(self, files):
        if not files:
            return "No files selected."

        folder = Path("sample_data")
        folder.mkdir(exist_ok=True)
        all_chunks = []
        count = 0

        for file in files:
            if not str(file.name).lower().endswith('.pdf'):
                continue
            dest = folder / Path(file.name).name
            try:
                content = file.read() if hasattr(file, 'read') else open(file.name, 'rb').read()
                with open(dest, "wb") as f:
                    f.write(content)
            except Exception as e:
                continue

            text = ""
            try:
                with open(dest, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        t = page.extract_text()
                        if t:
                            text += t + " "
            except:
                continue

            if text.strip():
                chunks = [text[i:i+500] for i in range(0, len(text), 450)]
                all_chunks.extend([{"content": c.strip()} for c in chunks if c.strip()])
                count += 1

        if not all_chunks:
            return "No readable text found in the PDFs."

        vecs = self.embedder.encode([c["content"] for c in all_chunks], show_progress_bar=False)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vecs.astype('float32'))
        self.chunks = all_chunks

        return f"Loaded {count} PDF(s) → {len(all_chunks)} chunks ready!"

    def ask(self, question: str, history: List):
        if not question.strip():
            return history, None

        if not history:
            history = []

        if question.strip().lower() in ["hi", "hello", "hey", "hola", "howdy"]:
            reply = "Hi there! I am AI Research Agent with agentic capabilities. Upload PDF documents and ask complex questions!"
            history.append([question, reply])
            return history, self.generate_voice(reply)

        if not self.index:
            reply = "Please upload a PDF document first!"
            history.append([question, reply])
            return history, self.generate_voice(reply)

        q_vec = self.embedder.encode([question])
        q_vec = q_vec / np.linalg.norm(q_vec)
        D, I = self.index.search(q_vec.astype('float32'), k=6)
        context = "\n\n".join([self.chunks[i]["content"] for i in I[0] if i < len(self.chunks)])

        prompt = f"Context from documents:\n{context}\n\nQuestion: {question}\nAnswer clearly and accurately:"

        if not self.groq:
            reply = "GROQ_API_KEY is missing or invalid.\n\nPlease check Settings → Secrets:\nName: GROQ_API_KEY\nValue: gsk_... (no quotes, no spaces)"
        else:
            try:
                resp = self.groq.chat.completions.create(
                    model="llama-3.1-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=700
                )
                reply = resp.choices[0].message.content.strip()
            except Exception as e:
                reply = f"Groq API error: {str(e)}"

        history.append([question, reply])
        return history, self.generate_voice(reply)

# YOUR ORIGINAL UI - 100% EXACT AS YOU WANTED
def create_interface():
    agent = AgenticRAGAgent()

    with gr.Blocks(title="🤖 AI Research Agent", theme=gr.themes.Soft()) as interface:
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px;">
            <h1 style="color: white; margin: 0;">🤖 AI Research Agent - Agentic RAG</h1>
            <p style="color: white; margin: 10px 0;">Advanced Multi-Tool Research Assistant with Voice Support 🔊</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="💬 Chat", height=500)

                with gr.Row():
                    msg = gr.Textbox(label="", placeholder="Ask a complex research question...", scale=4)
                    submit_btn = gr.Button("🚀 Send", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

                audio_output = gr.Audio(label="🔊 Voice Response", autoplay=True, interactive=False)

            with gr.Column(scale=1):
                with gr.Group():
                    gr.HTML("<h3 style='text-align: center;'>📄 Upload Documents</h3>")
                    file_upload = gr.Files(label="", file_types=[".pdf"], file_count="multiple")
                upload_status = gr.Textbox(label="📊 Status", interactive=False, max_lines=10)

        def respond(message, history):
            new_hist, audio_file = agent.ask(message, history)
            return "", new_hist, audio_file

        submit_btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot, audio_output])
        msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot, audio_output])
        clear_btn.click(lambda: ([], None), outputs=[chatbot, audio_output])
        file_upload.change(agent.upload_pdfs, inputs=[file_upload], outputs=[upload_status])

    return interface

if __name__ == "__main__":
    app = create_interface()
    app.launch(server_name="0.0.0.0", server_port=7860)
