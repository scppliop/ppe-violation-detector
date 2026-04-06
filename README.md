# 🦺 Beyond Detection: AI-Powered Safety Violation Reporting for Construction Sites

> A three-stage AI pipeline that transforms raw construction site images into fully cited, formal WorkSafeBC incident reports — automatically.

---

## 📌 Overview

Conventional computer-vision systems detect a safety violation and output a bare label like `"no hard hat"` — and stop there. A supervisor still has to manually identify which regulation was violated, assess the risk level, and write a report.

This project goes beyond detection. Given a single construction site image, the system:

1. **Detects** PPE violations and work zones using YOLOv8
2. **Describes** the violation in context using Gemini 2.5 Flash (VLM)
3. **Retrieves** the relevant WorkSafeBC OHS regulations using a FAISS RAG pipeline
4. **Generates** a formal supervisor incident report using LLaMA 3.3

### Key Results
| Metric | Result |
|---|---|
| Citation overlap | **100%** |
| Hallucination rate | **0%** |
| WorkSafeBC sections indexed | **21** |
| Min similarity threshold | **0.50** |

---

## 🏗️ Pipeline Architecture

```
Image Input
    │
    ▼
┌─────────────────────────────────────────┐
│  Stage 1 — Detection                    │
│  YOLOv8 (PPE) + Zone model + Harness    │
│  → worker crops + violation JSON        │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Stage 2 — Description (VLM)            │
│  Gemini 2.5 Flash                       │
│  → violation_types, severity, zone,     │
│     description (JSON)                  │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Stage 3 — Regulation Retrieval (RAG)   │
│  sentence-transformers + FAISS          │
│  → priority_sections[] + WorkSafeBC     │
│     chunks (score ≥ 0.50)               │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Report Generation                      │
│  LLaMA 3.3-70b (Groq API)               │
│  → formal supervisor incident report    │
└─────────────────────────────────────────┘
```

### RAG Design

The RAG pipeline is built **from scratch** — no LangChain or LlamaIndex. It combines two strategies:

- **Priority-section selection**: rule-based logic that selects regulations based on violation type, work zone, and nearby equipment before any semantic search
- **Semantic fallback**: FAISS `IndexFlatIP` cosine similarity search over 21 WorkSafeBC chunks using `all-MiniLM-L6-v2` embeddings

This combination achieves 100% citation overlap with 0% hallucination rate.

---

## 🚨 Violations Detected

| Violation | WorkSafeBC Sections |
|---|---|
| Missing hard hat | §8.11(1), §8.11(2), §8.11(4) |
| Missing high-visibility vest | §8.24, §8.25 |
| Missing harness (elevated zones) | §11.2, §11.4, §11.9 |
| Restricted-zone entry | §20.9 |

### Risk Score Calculation

```
Risk Score = Base Score × Zone Weight × Severity Weight  (capped at 100)
```

| Violation | Base Score | Zone Multiplier | Severity × | Max Score |
|---|---|---|---|---|
| Missing harness | 25 | ×2.0 (elevated) | ×1.5 (high) | 75 |
| Missing hard hat | 15 | ×1.5 (restricted) | ×1.5 (high) | 34 |
| Missing high-vis vest | 10 | ×1.5 (vehicle) | ×1.5 (high) | 23 |

---

## 📋 Example Output

**Input:** Construction site image with worker in restricted trench zone

**Detected:**
- Worker: `worker_001`
- Violations: Missing high-vis vest · Missing harness
- Zone: `restricted-zone` / `elevated-work-zone`
- Severity: `high` (active excavator overhead)
- Risk Score: `56 / 100`
- Cited regulations: §8.24, §8.25, §11.4, §11.2

**Generated Supervisor Report:**
> "On 2026-04-05 17:06, worker_001 was observed in a restricted trench zone without high-visibility apparel while an excavator operated directly overhead, in clear violation of WorkSafeBC §8.24 and §11.4. Immediate corrective action is required to prevent serious injury or fatality."

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| PPE Detection | YOLOv8 (`best.pt`) |
| Zone Detection | YOLOv8 (`zone_best.pt`) |
| Harness Detection | YOLOv8 (`harness_best.pt`) |
| VLM | Gemini 2.5 Flash (Google GenAI SDK) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector Store | FAISS `IndexFlatIP` (`faiss-cpu`) |
| LLM | LLaMA 3.3-70b-versatile (Groq API) |
| UI | Streamlit |
| Regulations | WorkSafeBC OHS Regulation 2026 |

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/ppe-violation-detector.git
cd ppe-violation-detector
pip install -r requirements.txt
```

### Requirements

```
ultralytics
google-genai
sentence-transformers
faiss-cpu
streamlit
groq
opencv-python
Pillow
numpy
```

### API Keys

In `app.py`, set your API keys:

```python
GEMINI_API_KEY = "your-gemini-api-key"
GROQ_API_KEY   = "your-groq-api-key"
```

### Model Files

Place the following model files in the project root:

```
best.pt          # PPE detection model
zone_best.pt     # Zone classification model
harness_best.pt  # Harness detection model
```

---

## 🚀 Usage

```bash
streamlit run app.py
```

1. Open the browser at `http://localhost:8501`
2. Upload a construction site image (JPG / PNG)
3. The system detects violations, retrieves WorkSafeBC regulations, and generates a supervisor report automatically

---

## 📁 Project Structure

```
ppe-violation-detector/
│
├── app.py                  # Main Streamlit application
├── prompt_v4.txt           # Gemini VLM prompt
├── best.pt                 # PPE YOLO model
├── zone_best.pt            # Zone YOLO model
├── harness_best.pt         # Harness YOLO model
├── crops/                  # Temporary worker crop images
└── README.md
```

---

## 📊 WorkSafeBC Regulations Indexed

**Part 8 — PPE:** §8.8, §8.9, §8.11(1), §8.11(2), §8.11(4), §8.11(6), §8.22, §8.24, §8.25

**Part 11 — Fall Protection:** §11.2, §11.3, §11.4, §11.9

**Part 20 — Construction:** §20.4, §20.9

Total: **21 sections**

---

## 🔭 Future Work

- Expand to mask, gloves, and safety-shoe detection using the [Keremberke construction dataset](https://huggingface.co/datasets/keremberke/construction-safety-object-detection)
- Heavy-equipment proximity detection (excavator / dump truck IoU logic) for Part 20 violations
- Real-time video stream processing with ByteTrack worker tracking
- Multi-language report generation for diverse construction sites
- Automated alert dispatch to supervisors via email or site management system

---

## 👥 Team

| Role | Responsibility |
|---|---|
| Stage 1 | PPE & zone detection (YOLOv8) |
| Stage 2 | VLM description pipeline (Gemini) |
| Stage 3 | RAG pipeline & system integration |

Course: **Generative AI — Northeastern University, 2026**

---

## 📄 References

1. WorkSafeBC, "OHS Regulation," Jan. 2026. [worksafebc.com](https://www.worksafebc.com)
2. Keremberke, "Construction Safety Object Detection Dataset," Hugging Face, 2023.
3. Jocher et al., YOLOv8, Ultralytics, 2023.
4. Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," EMNLP, 2019.
5. Johnson et al., "Billion-scale similarity search with GPUs," IEEE Transactions, 2021.
6. Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box," ECCV, 2022.
