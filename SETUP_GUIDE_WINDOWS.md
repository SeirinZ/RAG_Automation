# RAG RCA Chatbot - Windows Setup Guide

Panduan lengkap untuk setup RAG RCA Chatbot di Windows PC (server).

---

## CHECKLIST SEBELUM MULAI

Di Windows PC yang akan jadi server, pastikan sudah ada:
- [x] VSCode (sudah)
- [x] GitHub account connected (sudah)
- [x] Ollama installed (sudah)
- [x] LLM model downloaded (sudah)
- [ ] Python 3.12+
- [ ] Git

---

## STEP 1: Install Prerequisites di Windows

### 1.1 Install Python 3.12+
1. Download dari https://www.python.org/downloads/
2. **PENTING**: Saat install, centang **"Add Python to PATH"**
3. Verify:
   ```powershell
   python --version
   # Harus 3.12 atau lebih tinggi
   ```

### 1.2 Install Git
1. Download dari https://git-scm.com/download/win
2. Install dengan default settings
3. Verify:
   ```powershell
   git --version
   ```

### 1.3 Verify Ollama
```powershell
ollama list
# Harus ada llama3.1:8b
```

Jika belum ada model:
```powershell
ollama pull llama3.1:8b
```

---

## STEP 2: Transfer Code ke Windows PC

### Option A: Via GitHub (Recommended)

**Di Mac (saat ini):**
```bash
cd /Users/satriabintangadyatmaputra/Desktop/satria/RAG_Automation

# Initialize git repo (jika belum)
git init

# Create .gitignore
cat > .gitignore << 'EOF'
.venv/
__pycache__/
*.pyc
chroma_db/
chroma_db_old/
pdf_uploads/
n8n_data/
.DS_Store
*.log
.env
EOF

# Add and commit
git add .
git commit -m "Initial commit - RAG RCA Chatbot"

# Create repo di GitHub, lalu:
git remote add origin https://github.com/YOUR_USERNAME/RAG_Automation.git
git push -u origin main
```

**Di Windows PC:**
```powershell
# Clone repo
cd C:\Users\YourName\Desktop
git clone https://github.com/YOUR_USERNAME/RAG_Automation.git
cd RAG_Automation
```

### Option B: Via USB / File Transfer

1. Copy seluruh folder **KECUALI**:
   - `.venv/` (jangan copy, buat baru)
   - `chroma_db/` (bisa copy atau regenerate)
   - `__pycache__/`

2. Yang **HARUS** dicopy:
   - `app/` (semua file)
   - `scripts/`
   - `processed_pdfs/` (4 file PDF RCA)
   - `main.py`
   - `requirements.txt`

---

## STEP 3: Setup Python Environment di Windows

```powershell
# Masuk ke folder project
cd C:\Users\YourName\Desktop\RAG_Automation

# Buat virtual environment
python -m venv .venv

# Activate virtual environment
# PENTING: Di PowerShell
.\.venv\Scripts\Activate.ps1

# Jika ada error "execution policy", jalankan dulu:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Kemudian activate lagi
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Catatan**: Install akan download sekitar 2-3 GB (PyTorch, transformers, dll)

---

## STEP 4: Buat Folders yang Diperlukan

```powershell
# Di dalam folder RAG_Automation
mkdir pdf_uploads -ErrorAction SilentlyContinue
mkdir chroma_db -ErrorAction SilentlyContinue
```

---

## STEP 5: Ingest PDF RCA Documents

Jalankan sekali untuk memasukkan dokumen ke database:

```powershell
# Pastikan virtual environment aktif
# Lihat (.venv) di awal command prompt

python -c "
from app.vector_store import vector_store
from app.models import model_manager

# Initialize
model_manager.initialize()
vector_store.initialize()

# Ingest semua PDF di processed_pdfs
import os
pdf_dir = './processed_pdfs'
for filename in os.listdir(pdf_dir):
    if filename.endswith('.pdf'):
        filepath = os.path.join(pdf_dir, filename)
        print(f'Ingesting: {filename}')
        vector_store.ingest_document(filepath)

print('Done! All documents ingested.')
"
```

---

## STEP 6: Jalankan Server

```powershell
# Pastikan virtual environment aktif
python main.py
```

Output yang diharapkan:
```
============================================================
STARTING RAG SYSTEM
============================================================
>>> Step 1: Models
>>> Step 2: Vector Store
>>> Step 3: Retrieval
============================================================
SYSTEM READY
============================================================
API: http://0.0.0.0:8000
Docs: http://0.0.0.0:8000/docs
============================================================
```

---

## STEP 7: Test API

Buka browser: `http://localhost:8000/docs`

Test query di Swagger UI atau via PowerShell:
```powershell
curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d "{\"query\": \"Apa root cause dari AR 117422?\"}"
```

---

## STEP 8: Akses dari Komputer Lain (Network)

### Cari IP Address Windows PC:
```powershell
ipconfig
# Cari IPv4 Address, misal: 192.168.1.100
```

### Dari komputer lain di network yang sama:
```
http://192.168.1.100:8000/docs
```

### Buka Firewall (jika perlu):
```powershell
# Run as Administrator
netsh advfirewall firewall add rule name="RAG API" dir=in action=allow protocol=TCP localport=8000
```

---

## TROUBLESHOOTING

### Error: "torch" install gagal
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Error: Ollama tidak connect
```powershell
# Pastikan Ollama service running
ollama serve
```

### Error: chromadb permission
```powershell
# Hapus dan buat ulang
Remove-Item -Recurse -Force chroma_db
mkdir chroma_db
# Kemudian ingest ulang
```

### Error: Model not found
```powershell
ollama pull llama3.1:8b
```

---

## MENAMBAH RCA PDF BARU

### Via API (Recommended):
```powershell
curl -X POST "http://localhost:8000/upload" -F "file=@C:\path\to\new_rca.pdf"
```

### Manual:
1. Copy PDF ke folder `processed_pdfs/`
2. Edit `app/vector_store.py`, tambahkan ke `FILENAME_TO_AR`:
   ```python
   FILENAME_TO_AR = {
       # ... existing ...
       'new_file.pdf': ('AR_NUMBER', 'Title of RCA'),
   }
   ```
3. Ingest ulang atau restart server

---

## QUICK REFERENCE

| Task | Command |
|------|---------|
| Activate venv | `.\.venv\Scripts\Activate.ps1` |
| Start server | `python main.py` |
| Check Ollama | `ollama list` |
| API Docs | `http://localhost:8000/docs` |
| Health check | `http://localhost:8000/health` |

---

## FILE STRUCTURE

```
RAG_Automation/
├── app/
│   ├── __init__.py
│   ├── api.py           # FastAPI endpoints
│   ├── config.py        # Settings
│   ├── models.py        # LLM & embeddings
│   ├── retrieval.py     # Hybrid search + reranking
│   ├── vector_store.py  # ChromaDB + ingestion
│   └── openai_compat.py # OpenAI-compatible endpoint
├── scripts/
│   └── preprocess_pdf.py
├── processed_pdfs/      # RCA documents (4 files)
├── pdf_uploads/         # For new uploads
├── chroma_db/           # Vector database
├── main.py              # Entry point
├── requirements.txt     # Dependencies
└── SETUP_GUIDE_WINDOWS.md
```
