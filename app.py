import os
import re
import logging
import tempfile
from pathlib import Path
from typing import List, Tuple, Any
import numpy as np
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import gradio as gr
from gtts import gTTS
import requests
import math
import ast
import json

try:
    import sympy as sp
    SYMPY_OK = True
except Exception:
    SYMPY_OK = False

try:
    from groq import Groq
    GROQ_OK = True
except ImportError:
    GROQ_OK = False
    print("❌ Groq library not installed!")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "Your-api-key")
groq_client = None

if GROQ_OK:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq client initialized successfully!")
    except Exception as e:
        groq_client = None
        print(f"❌ Groq initialization error: {e}")

# GLOBAL CHAT MEMORY
chat_memory = []

# Safe evaluation for calculations
class SafeEval(ast.NodeVisitor):
    ALLOWED_NAMES = {n: getattr(math, n) for n in dir(math) if not n.startswith("__")}
    ALLOWED_NAMES.update({"abs": abs, "round": round})

    def visit(self, node):
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        if isinstance(node, ast.BinOp):
            left = self.visit(node.left)
            right = self.visit(node.right)
            return self._binop(node.op, left, right)
        if isinstance(node, ast.UnaryOp):
            operand = self.visit(node.operand)
            return self._unaryop(node.op, operand)
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in self.ALLOWED_NAMES:
                args = [self.visit(a) for a in node.args]
                return self.ALLOWED_NAMES[func.id](*args)
        if isinstance(node, ast.Name):
            if node.id in self.ALLOWED_NAMES:
                return self.ALLOWED_NAMES[node.id]
            raise ValueError(f"Use of name '{node.id}' is not allowed")
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    def _binop(self, op, a, b):
        if isinstance(op, ast.Add): return a + b
        if isinstance(op, ast.Sub): return a - b
        if isinstance(op, ast.Mult): return a * b
        if isinstance(op, ast.Div): return a / b
        if isinstance(op, ast.Mod): return a % b
        if isinstance(op, ast.Pow): return a ** b
        raise ValueError("Unsupported binary operator")

    def _unaryop(self, op, a):
        if isinstance(op, ast.UAdd): return +a
        if isinstance(op, ast.USub): return -a
        raise ValueError("Unsupported unary operator")

def safe_calc_eval(expr: str):
    expr = expr.strip()
    if SYMPY_OK:
        try:
            result = sp.sympify(expr)
            numeric = None
            try:
                numeric = float(result.evalf())
            except:
                numeric = None
            if numeric is not None:
                return True, str(numeric)
            return True, str(result)
        except:
            pass
    try:
        node = ast.parse(expr, mode='eval')
        se = SafeEval()
        val = se.visit(node)
        return True, str(val)
    except Exception as e:
        return False, f"Calc error: {e}"

# Simple web search
def web_search(query: str, max_results: int = 3) -> List[dict]:
    try:
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        resp.raise_for_status()
        text = resp.text
        results = []
        parts = text.split('result__a')
        for part in parts[1:max_results+1]:
            try:
                title = part.split('>')[1].split('<')[0]
            except:
                title = ""
            snippet = ""
            try:
                snippet = part.split('result__snippet')[1].split('>')[1].split('<')[0]
            except:
                snippet = ""
            results.append({"title": title, "snippet": snippet})
        return results
    except:
        return []

class AgenticRAGAgent:
    def __init__(self):
        self.chunks = []
        self.index = None
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.temperature = 0.3
        self.max_tokens = 500
        self.chunk_size = 512
        self.chunk_overlap = 50
        self.retrieval_k = 8
        self.enable_web_search = True
        self.enable_calculations = True
        self.enable_fact_checking = True
        self.enable_analysis = True
        print("✅ AgenticRAGAgent initialized")

    def remove_emojis(self, text: str) -> str:
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
            tts = gTTS(text=clean, lang='en', slow=False)
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
                all_chunks.extend([{"content": str(c.strip())} for c in chunks if c.strip()])
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

    def detect_math(self, text: str):
        if re.search(r'[0-9]', text) and re.search(r'[\+\-\*\/\^%=]', text):
            expr = text.strip()
            expr = re.sub(r'[a-zA-Z,?]+', '', expr)
            expr = expr.strip()
            return expr if len(expr) > 0 else None
        return None

    # Minimal fact check implementation
    def perform_fact_check(self, text: str, context: str) -> str:
        if not context or not text: return ""
        try:
            claims = [s.strip() for s in text.split('.') if s.strip() and len(s.strip()) > 10]
            verified = []
            for claim in claims[:2]:
                key_terms = [w for w in claim.split() if len(w) > 4]
                matches = sum(1 for term in key_terms if term.lower() in context.lower())
                if matches >= len(key_terms) * 0.5:
                    verified.append(f"✓ {claim[:60]}...")
            if verified:
                return "\n[✅ Fact Check]\n" + "\n".join(verified)
            return ""
        except:
            return ""

    def perform_analysis(self, text: str, context: str, question: str) -> str:
        if not text or len(text) < 20: return ""
        analysis = []
        sentence_count = len([s for s in text.split('.') if s.strip()])
        if sentence_count >= 3: analysis.append("📊 Comprehensive answer with multiple points")
        context_refs = sum(1 for word in context.split() if len(word) > 5 and word.lower() in text.lower())
        if context_refs > 0: analysis.append(f"📄 References from {context_refs} key context terms")
        word_count = len(text.split())
        if word_count > 100: analysis.append(f"📝 Detailed response ({word_count} words)")
        elif word_count > 50: analysis.append(f"📝 Moderate response ({word_count} words)")
        q_words = [w.lower() for w in question.split() if len(w) > 3]
        answer_relevance = sum(1 for w in q_words if w in text.lower())
        if answer_relevance >= len(q_words) * 0.5: analysis.append("✓ Answer directly addresses the question")
        if analysis:
            return "\n[📊 Analysis]\n" + "\n".join(analysis)
        return ""

    def ask(self, question: str, history: List) -> Tuple[List, Any]:
        global groq_client
        if not isinstance(question, str): question = str(question) if question else ""
        if not isinstance(history, list): history = []
        question = question.strip()
        if not question: return history, None

        if question.lower() in ["hi", "hello", "hey"]:
            reply = "Hi! I am your AI Research Agent. Upload PDFs and ask questions."
            history.append([question, reply])
            return history, self.generate_voice(reply)

        if not self.index:
            reply = "Please upload a PDF first!"
            history.append([question, reply])
            return history, self.generate_voice(reply)

        # Retrieve chunks
        try:
            q_vec = self.embedder.encode([question])
            q_vec = q_vec / np.linalg.norm(q_vec)
            D, I = self.index.search(q_vec.astype('float32'), k=self.retrieval_k)
            context_list = [self.chunks[i]["content"] for i in I[0] if i < len(self.chunks)]
            context = "\n\n".join(context_list).strip()
        except:
            context = ""

        prompt = f"Context from documents:\n{context}\n\nQuestion: {question}\nAnswer clearly and accurately:" if context else f"Question: {question}\nAnswer clearly and accurately:"

        calc_note = ""
        web_note = ""
        fact_note = ""
        analysis_note = ""

        if self.enable_calculations:
            expr = self.detect_math(question)
            if expr:
                ok, res = safe_calc_eval(expr)
                if ok:
                    calc_note = f"\n[🧮 Calculator] {expr} = {res}"

        if self.enable_web_search:
            keywords = ["latest", "today", "current", "recent", "news"]
            if any(k in question.lower() for k in keywords):
                results = web_search(question)
                if results:
                    web_note = "\n[🌐 Web Sources]:\n" + "\n".join([f"- {r.get('title','')}" for r in results[:2]])

        reply = "Error processing request."

        # Groq API call
        if groq_client:
            try:
                messages = [{"role": "user", "content": prompt}]
                resp = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=float(self.temperature),
                    max_tokens=int(self.max_tokens)
                )
                if resp and resp.choices and len(resp.choices) > 0:
                    reply = str(resp.choices[0].message.content).strip()
                else:
                    reply = "No response from API"
            except Exception as e:
                reply = f"Error: {e}"

        # Add tools notes
        if calc_note: reply += calc_note
        if web_note: reply += web_note
        if self.enable_fact_checking: fact_note = self.perform_fact_check(reply, context)
        if fact_note: reply += fact_note
        if self.enable_analysis: analysis_note = self.perform_analysis(reply, context, question)
        if analysis_note: reply += analysis_note

        history.append([question, reply])
        return history, self.generate_voice(reply)

    def update_settings(self, temp, tokens, chunk_size, overlap, k, web, calc, fact, analysis):
        self.temperature = float(temp)
        self.max_tokens = int(tokens)
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(overlap)
        self.retrieval_k = int(k)
        self.enable_web_search = bool(web)
        self.enable_calculations = bool(calc)
        self.enable_fact_checking = bool(fact)
        self.enable_analysis = bool(analysis)
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

# ===== Gradio Interface =====
def create_interface():
    global chat_memory
    chat_memory = []
    agent = AgenticRAGAgent()

    with gr.Blocks(title="AI Research Agent") as interface:
        gr.HTML("""
        <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);border-radius:15px;">
        <h1 style="color:white;margin:0;">🤖 AI Research Agent - Agentic RAG</h1>
        <p style="color:white;margin:10px 0;">Advanced Multi-Tool Research Assistant with Voice Support 🎤🔊</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="💬 Chat", height=500)
                with gr.Row():
                    msg = gr.Textbox(placeholder="Ask a complex research question...", scale=4, lines=1)
                    submit_btn = gr.Button("🚀 Send", variant="primary", scale=1)
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                audio_output = gr.Audio(label="🔊 Voice Response", autoplay=True, interactive=False)

            with gr.Column(scale=1):
                gr.HTML("<h3 style='text-align:center;'>📄 Upload Documents</h3>")
                pdf_upload = gr.Files(file_types=[".pdf"], label="Upload PDFs")
                upload_status = gr.Textbox(label="📊 Status", interactive=False, max_lines=10)

                with gr.Accordion("⚙️ AI Parameters", open=False):
                    temperature_slider = gr.Slider(0.0, 1.0, value=0.3, step=0.1, label="🌡️ Temperature")
                    max_tokens_slider = gr.Slider(100, 2000, value=500, step=50, label="📝 Max Tokens")

                with gr.Accordion("📄 Document Processing", open=False):
                    chunk_size_slider = gr.Slider(256, 1024, value=512, step=64, label="📄 Chunk Size")
                    chunk_overlap_slider = gr.Slider(0, 200, value=50, step=10, label="🔗 Chunk Overlap")
                    retrieval_k_slider = gr.Slider(3, 15, value=8, step=1, label="🔍 Retrieved Chunks")

                with gr.Accordion("🛠️ Agentic Tools", open=False):
                    enable_web = gr.Checkbox(value=True, label="🌐 Web Search")
                    enable_calc = gr.Checkbox(value=True, label="🧮 Calculator")
                    enable_fact = gr.Checkbox(value=True, label="✅ Fact Check")
                    enable_analysis = gr.Checkbox(value=True, label="📊 Analysis")

                apply_btn = gr.Button("⚡ Apply Settings", variant="primary", size="lg")
                settings_status = gr.Textbox(label="⚙️ Settings Status", interactive=False, max_lines=10, value="Settings ready.")

        def respond(message):
            global chat_memory
            updated_history, audio_file = agent.ask(message, chat_memory)
            chat_memory = updated_history if isinstance(updated_history, list) else []
            display_history = []
            for item in chat_memory:
                if isinstance(item, list) and len(item) == 2:
                    display_history.append({"role": "user", "content": str(item[0])})
                    display_history.append({"role": "assistant", "content": str(item[1])})
            return "", display_history, audio_file

        def clear_chat():
            global chat_memory
            chat_memory = []
            return []

        submit_btn.click(respond, inputs=[msg], outputs=[msg, chatbot, audio_output])
        msg.submit(respond, inputs=[msg], outputs=[msg, chatbot, audio_output])
        clear_btn.click(clear_chat, outputs=[chatbot])
        pdf_upload.change(agent.upload_pdfs, inputs=[pdf_upload], outputs=[upload_status])
        apply_btn.click(agent.update_settings, inputs=[
            temperature_slider, max_tokens_slider, chunk_size_slider,
            chunk_overlap_slider, retrieval_k_slider, enable_web,
            enable_calc, enable_fact, enable_analysis
        ], outputs=[settings_status])

    return interface

if __name__ == "__main__":
    print("🚀 Starting AI Research Agent with Full UI...")
    app = create_interface()
    app.launch(server_name="0.0.0.0", server_port=7860, show_error=True)

