# app.py - FULLY WORKING AI RESEARCH AGENT WITH GROQ CLIENT
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
    print("❌ Groq library not installed!")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================
# 🔑 HARDCODE YOUR GROQ API KEY HERE (GLOBAL)
# ===============================
GROQ_API_KEY = "YOUR-API-KEY"
groq_client = None

if GROQ_OK:
    try:
        print("DEBUG → Initializing Groq client...")
        # Initialize with just api_key - most compatible approach
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("✅ DEBUG → Groq client initialized successfully!")
    except TypeError as te:
        # Fallback for version compatibility issues
        print(f"⚠️ TypeError during init: {te}")
        try:
            print("🔄 Attempting fallback initialization...")
            groq_client = Groq(api_key=GROQ_API_KEY)
            print("✅ Fallback initialization successful!")
        except Exception as e:
            groq_client = None
            print(f"❌ Groq initialization failed: {e}")
    except Exception as e:
        groq_client = None
        print(f"❌ Groq initialization error: {e}")
else:
    print("❌ Groq library import failed!")

class AgenticRAGAgent:
    def __init__(self):
        self.chunks = []
        self.index = None
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ AgenticRAGAgent initialized with SentenceTransformer")

    def remove_emojis(self, text: str) -> str:
        """Remove emojis from text for clean voice output"""
        emoji_pattern = re.compile("[" 
            u"\U0001F600-\U0001F64F" 
            u"\U0001F300-\U0001F5FF" 
            u"\U0001F680-\U0001F6FF" 
            u"\U0001F1E0-\U0001F1FF" 
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def clean_for_voice(self, text: str) -> str:
        """Clean text for voice synthesis"""
        text = self.remove_emojis(text)
        text = re.sub(r'[\*_`#\[\]]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def generate_voice(self, text: str):
        """Generate voice output from text"""
        if not text or not text.strip():
            return None
        clean = self.clean_for_voice(text)
        if len(clean) < 5:
            return None
        try:
            tts = gTTS(text=clean, lang='en', slow=False)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(tmp.name)
            return tmp.name
        except Exception as e:
            logger.error(f"Voice generation failed: {e}")
            return None

    def upload_pdfs(self, files):
        """Upload and process PDF files"""
        if not files:
            return "No files selected."

        folder = Path("sample_data")
        folder.mkdir(exist_ok=True)
        all_chunks = []
        count = 0

        for file in files:
            filename = str(file.name) if hasattr(file, 'name') else str(file)
            if not filename.lower().endswith('.pdf'):
                continue
            
            dest = folder / Path(filename).name
            try:
                content = file.read() if hasattr(file, 'read') else open(filename, 'rb').read()
                with open(dest, "wb") as f:
                    f.write(content)
            except Exception as e:
                logger.warning(f"Failed to save file {filename}: {e}")
                continue

            text = ""
            try:
                with open(dest, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        t = page.extract_text()
                        if t:
                            text += t + " "
            except Exception as e:
                logger.warning(f"Failed to extract text from {filename}: {e}")
                continue

            if text.strip():
                chunks = [text[i:i+500] for i in range(0, len(text), 450)]
                all_chunks.extend([{"content": c.strip()} for c in chunks if c.strip()])
                count += 1

        if not all_chunks:
            return "No readable text found in the PDFs."

        # Create embeddings and FAISS index
        print(f"Creating embeddings for {len(all_chunks)} chunks...")
        vecs = self.embedder.encode([c["content"] for c in all_chunks], show_progress_bar=True)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        dim = vecs.shape[1]
        
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vecs.astype('float32'))
        self.chunks = all_chunks

        status_msg = f"✅ Loaded {count} PDF(s) → {len(all_chunks)} chunks ready!"
        print(status_msg)
        return status_msg

    def ask(self, question: str, history: List):
        """Process user question and generate response"""
        global groq_client
        
        if not question.strip():
            return history, None

        if not history:
            history = []

        # Handle greeting
        if question.strip().lower() in ["hi", "hello", "hey", "hola", "howdy"]:
            reply = "Hi there! I am AI Research Agent with agentic capabilities. Upload PDF documents and ask complex questions!"
            history.append([question, reply])
            return history, self.generate_voice(reply)

        # Check if PDFs are loaded
        if not self.index:
            reply = "Please upload a PDF document first!"
            history.append([question, reply])
            return history, self.generate_voice(reply)

        # Retrieve relevant chunks
        q_vec = self.embedder.encode([question])
        q_vec = q_vec / np.linalg.norm(q_vec)
        D, I = self.index.search(q_vec.astype('float32'), k=6)
        context = "\n\n".join([self.chunks[i]["content"] for i in I[0] if i < len(self.chunks)])

        prompt = f"Context from documents:\n{context}\n\nQuestion: {question}\nAnswer clearly and accurately:"

        if groq_client is None:
            reply = "ERROR: Groq client is not initialized. Check your API key and connection."
            print("❌ Groq client is None - cannot process request")
        else:
            try:
                print(f"📤 Sending request to Groq API for question: {question[:50]}...")
                resp = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=700
                )
                reply = resp.choices[0].message.content.strip()
                print(f"✅ Received response from Groq API")
            except Exception as e:
                reply = f"Groq API error: {str(e)}"
                print(f"❌ Groq API error: {e}")

        history.append([question, reply])
        return history, self.generate_voice(reply)


# =========================================
# GRADIO UI
# =========================================
def create_interface():
    agent = AgenticRAGAgent()

    with gr.Blocks(title="AI Research Agent", theme=gr.themes.Soft()) as interface:
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px;">
            <h1 style="color: white; margin: 0;">AI Research Agent - Agentic RAG</h1>
            <p style="color: white; margin: 10px 0;">Advanced Multi-Tool Research Assistant with Voice Support</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
                    type="tuples"
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="",
                        placeholder="Ask a complex research question...",
                        scale=4,
                        lines=1
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", variant="secondary")

                audio_output = gr.Audio(
                    label="Voice Response",
                    autoplay=True,
                    interactive=False
                )

            with gr.Column(scale=1):
                with gr.Group():
                    gr.HTML("<h3 style='text-align: center;'>Upload Documents</h3>")
                    file_upload = gr.Files(
                        label="",
                        file_types=[".pdf"],
                        file_count="multiple"
                    )
                upload_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    max_lines=10
                )

        def respond(message, history):
            """Handle user message"""
            new_hist, audio_file = agent.ask(message, history)
            return "", new_hist, audio_file

        def clear_chat():
            """Clear chat history"""
            return [], None

        # Connect events
        submit_btn.click(
            respond,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, audio_output]
        )
        msg.submit(
            respond,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot, audio_output]
        )
        clear_btn.click(
            clear_chat,
            outputs=[chatbot, audio_output]
        )
        file_upload.change(
            agent.upload_pdfs,
            inputs=[file_upload],
            outputs=[upload_status]
        )

    return interface


if __name__ == "__main__":
    print("🚀 Starting AI Research Agent...")
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False
    )
