import os
import cv2
import json
import numpy as np
import faiss
import streamlit as st
import google.genai as genai
from PIL import Image
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from groq import Groq
from collections import defaultdict



# ============================================================
# Configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH    = os.path.join(BASE_DIR, "prompt_v4.txt")
CROPS_DIR      = os.path.join(BASE_DIR, "crops")
GEMINI_API_KEY = "your-gemini-api-key"  # Insert Gemini API key
GROQ_API_KEY   = "your-gemini-api-key"  # Insert Groq API key

os.makedirs(CROPS_DIR, exist_ok=True)

# ============================================================
# Stage 1 — YOLO Constants
# ============================================================
VIOLATION_MAP = {
    'NO-Hardhat':     'missing_hard_hat',
    'NO-Safety Vest': 'missing_high_vis',
    'no-harness':     'missing_harness'
}
PPE_VIOLATION_CLASSES = ['NO-Hardhat', 'NO-Safety Vest']
FRAME_CONTEXT_CLASSES = ['Hardhat', 'Safety Vest', 'machinery', 'vehicle', 'Safety Cone']

# ============================================================
# Stage 3 — RAG Constants
# ============================================================
VIOLATION_BASE_SCORES = {
    "missing_harness":  25,
    "missing_hard_hat": 15,
    "missing_high_vis": 10,
}
ZONE_WEIGHTS   = {"elevated-work-zone": 2.0, "restricted-zone": 1.5, "vehicle-zone": 1.5, "safe": 1.0}
SEVERITY_WEIGHTS = {"high": 1.5, "medium": 1.0, "low": 0.5}
MIN_SCORE      = 0.5

VIOLATION_QUERIES_WITH_EQUIPMENT = {
    "missing_hard_hat": [
        "hard hat safety headgear head protection falling flying thrown objects risk worker",
        "hard hat safety headgear CSA ANSI standard requirement industrial protective headwear",
    ],
    "missing_high_vis": [
        "high visibility apparel vest vehicle mobile equipment struck speed Class 2 Class 3 CSA Z96",
        "distinguishing apparel fluorescent retroreflective colour contrast daytime nighttime worker",
    ],
    "missing_harness": [
        "full body harness safety belt fall arrest restraint personal fall protection system",
        "fall protection obligation height 3m guardrail fall arrest restraint worker instruction",
    ],
}

VIOLATION_QUERIES_NO_EQUIPMENT = {
    "missing_hard_hat": [
        "hard hat safety headgear head protection falling flying thrown objects risk worker",
        "hard hat safety headgear CSA ANSI standard requirement industrial protective headwear",
    ],
    "missing_high_vis": [
        "distinguishing apparel fluorescent retroreflective colour contrast daytime nighttime worker",
    ],
    "missing_harness": [
        "full body harness safety belt fall arrest restraint personal fall protection system",
        "fall protection obligation height 3m guardrail fall arrest restraint worker instruction",
    ],
}

ALLOWED_VIOLATION_TYPES = {"missing_hard_hat", "missing_high_vis", "missing_harness", "unknown"}
ALLOWED_SEVERITY        = {"low", "medium", "high"}
ALLOWED_ZONES           = {"safe", "elevated-work-zone", "restricted-zone", "vehicle-zone"}
VIOLATION_LABEL = {
    "missing_hard_hat": "Missing Hard Hat",
    "missing_high_vis": "Missing High-Vis Vest",
    "missing_harness":  "Missing Harness",
}

worksafebc_chunks = [
    {"section": "8.11(1)", "topic": "hard hat safety headgear head protection falling flying thrown objects risk eliminate", "text": "Before a worker starts a work assignment in a work area where there is a risk of head injury to the worker from falling, flying or thrown objects, or other harmful contacts, the employer must take measures to eliminate the risk, or if it is not practicable to eliminate the risk, minimize the risk to the lowest level practicable by applying engineering controls, administrative controls, or requiring the worker to wear safety headgear."},
    {"section": "8.11(2)", "topic": "hard hat safety headgear CSA ANSI standard requirement", "text": "Safety headgear must meet the requirements of one of the following standards: CSA Standard CAN/CSA-Z94.1-05 or CAN/CSA-Z94.1-15, Industrial protective headwear — Performance, selection, care, and use; or ANSI Standard ANSI/ISEA Z89.1-2009 or ANSI/ISEA Z89.1-2014, American National Standard for Industrial Head Protection."},
    {"section": "8.11(4)", "topic": "hard hat chin strap retention height elevation 3m climbing high winds", "text": "Chin straps or other effective means of retention must be used on safety headgear when workers are climbing or working from a height exceeding 3 m (10 ft), or are exposed to high winds or other conditions that may cause loss of the headgear."},
    {"section": "8.11(6)", "topic": "hard hat damaged missing modified components removed from service", "text": "Damaged headgear or headgear with missing, mismatched, or modified components must be removed from service."},
    {"section": "8.22",    "topic": "safety footwear foot protection slipping tripping crushing puncture electrical", "text": "A worker's footwear must be of a design, construction, and material appropriate to the protection required and that allows the worker to safely perform their work."},
    {"section": "8.8",     "topic": "PPE supervisor responsibility available worn cleaned inspected stored", "text": "The supervisor must ensure that appropriate personal protective equipment is available to workers, properly worn when required, and properly cleaned, inspected, maintained and stored."},
    {"section": "8.9",     "topic": "PPE worker responsibility inspect report malfunction training use", "text": "A worker who is required to use personal protective equipment must use the equipment in accordance with training and instruction, inspect the equipment before use, and report any equipment malfunction to the supervisor or employer."},
    {"section": "8.24",    "topic": "high visibility apparel vest vehicle mobile equipment struck speed Class 2 Class 3 CSA Z96", "text": "A worker who is exposed to vehicles or mobile equipment travelling at speeds in excess of 30 km/h must wear high visibility apparel that meets the requirements for Class 2 or Class 3 apparel in CSA Standard Z96-15, High-Visibility Safety Apparel."},
    {"section": "8.25",    "topic": "distinguishing apparel fluorescent retroreflective colour contrast daytime nighttime", "text": "If distinguishing apparel is required for the purpose of identifying a worker's location or well-being, the apparel must be of a colour which contrasts with the environment and must have at least 775 sq cm (120 sq in) of fluorescent trim for daytime use and retroreflective trim for nighttime use, on both the front and back."},
    {"section": "11.2",    "topic": "fall protection obligation height 3m guardrail fall arrest restraint worker instruction", "text": "An employer must ensure that a fall protection system is used when work is being done at a place from which a fall of 3 m (10 ft) or more may occur. The employer must ensure guardrails or other fall restraint are used when practicable. A worker must use the fall protection system provided by the employer."},
    {"section": "11.3",    "topic": "fall protection plan written workplace guardrail 7.5m", "text": "The employer must have a written fall protection plan for a workplace if work is being done at a location where workers are not protected by permanent guardrails and from which a fall of 7.5 m (25 ft) or more may occur."},
    {"section": "11.4",    "topic": "full body harness safety belt fall arrest restraint personal fall protection system", "text": "A worker must wear a full body harness or other harness acceptable to the Board when using a personal fall protection system for fall arrest or fall restraint."},
    {"section": "11.9",    "topic": "fall protection equipment inspection maintenance qualified person workshift", "text": "Equipment used in a fall protection system must be inspected by a qualified person before use on each workshift, kept free from substances that could contribute to its deterioration, and maintained in good working order."},
    {"section": "20.4",    "topic": "safe access ladder scaffold platform elevation floor grade delivery materials", "text": "Where practicable, suitable ladders, work platforms and scaffolds meeting the requirements of Part 13 must be provided for and used by a worker for activities requiring positioning at elevations above a floor or grade."},
    {"section": "20.9",    "topic": "falling material danger barricade warning signs canopy restricted zone workers entry", "text": "If falling material could endanger workers, the danger area must be barricaded or effectively guarded to prevent entry by workers, and conspicuous warning signs must be displayed on all sides and approaches."},
]

# ============================================================
# Load Models (cached)
# ============================================================
@st.cache_resource
def load_models():
    ppe     = YOLO(os.path.join(BASE_DIR, "best.pt"))
    zone    = YOLO(os.path.join(BASE_DIR, "zone_best.pt"))
    harness = YOLO(os.path.join(BASE_DIR, "harness_best.pt"))

    gemini = genai.Client(api_key=GEMINI_API_KEY)

    embedder   = SentenceTransformer("all-MiniLM-L6-v2")
    texts      = [f"{c['topic']} {c['text']}" for c in worksafebc_chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    idx = faiss.IndexFlatIP(embeddings.shape[1])
    idx.add(embeddings)

    groq = Groq(api_key=GROQ_API_KEY)

    return ppe, zone, harness, gemini, embedder, idx, groq

# ============================================================
# Helper Functions
# ============================================================
def iou(b1, b2):
    x1,y1,x2,y2 = max(b1[0],b2[0]),max(b1[1],b2[1]),min(b1[2],b2[2]),min(b1[3],b2[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1]); a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if a1+a2-inter>0 else 0

def is_inside(inner, outer, threshold=0.5):
    x1,y1,x2,y2 = max(inner[0],outer[0]),max(inner[1],outer[1]),min(inner[2],outer[2]),min(inner[3],outer[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    ia = (inner[2]-inner[0])*(inner[3]-inner[1])
    return (inter/ia)>threshold if ia>0 else False

def expand_crop(box, shape, padding=80):
    h,w = shape[:2]
    return max(0,int(box[0])-padding),max(0,int(box[1])-padding),min(w,int(box[2])+padding),min(h,int(box[3])+padding)

def retrieve_regulations(query, embedder, idx):
    q = embedder.encode([query], convert_to_numpy=True)
    q = q/np.linalg.norm(q,axis=1,keepdims=True)
    scores, indices = idx.search(q, len(worksafebc_chunks))
    results = [{"section":worksafebc_chunks[i]["section"],"text":worksafebc_chunks[i]["text"],"score":float(s)} for s,i in zip(scores[0],indices[0]) if float(s)>=MIN_SCORE]
    if not results:
        i = int(indices[0][0])
        results = [{"section":worksafebc_chunks[i]["section"],"text":worksafebc_chunks[i]["text"],"score":float(scores[0][0]),"fallback":True}]
    return results

def get_priority_regulations(vtypes, zone, nearby_equipment):
    priority = []

    for vtype in vtypes:
        if vtype == "missing_hard_hat":
            if zone == "elevated-work-zone":
                priority += ["8.11(1)", "8.11(4)"]
            else:
                priority += ["8.11(1)"]
        elif vtype == "missing_harness":
            if zone == "elevated-work-zone":
                priority += ["11.4", "11.9"]
        elif vtype == "missing_high_vis":
            if nearby_equipment:
                priority += ["8.24", "8.25"]
            else:
                priority += ["8.25"]

    seen = []
    for s in priority:
        if s not in seen:
            seen.append(s)
    priority = seen

    for fallback in ["8.8", "8.9"]:
        if len(priority) < 3 and fallback not in priority:
            priority.append(fallback)

    return priority


def retrieve_per_violation(vtypes, embedder, idx, zone="safe", nearby_equipment=None):
    priority_sections = get_priority_regulations(vtypes, zone, nearby_equipment)
    queries = VIOLATION_QUERIES_WITH_EQUIPMENT if nearby_equipment else VIOLATION_QUERIES_NO_EQUIPMENT

    # Build a lookup of all chunks by section
    chunk_lookup = {c["section"]: c for c in worksafebc_chunks}

    all_regs, seen = [], set()

    # First: add priority sections in order
    for section in priority_sections:
        if section in chunk_lookup and section not in seen:
            q = embedder.encode([f"{chunk_lookup[section]['topic']} {chunk_lookup[section]['text']}"], convert_to_numpy=True)
            q = q / np.linalg.norm(q, axis=1, keepdims=True)
            scores, indices = idx.search(q, len(worksafebc_chunks))
            score = next((float(s) for s, i in zip(scores[0], indices[0]) if worksafebc_chunks[i]["section"] == section), 0.5)
            all_regs.append({"section": section, "text": chunk_lookup[section]["text"], "score": score})
            seen.add(section)

    return all_regs

def calculate_risk_score(vtypes, zone, severity):
    known = [v for v in vtypes if v in VIOLATION_BASE_SCORES]
    if not known: return 0
    return min(100, int(sum(VIOLATION_BASE_SCORES[v] for v in known)*ZONE_WEIGHTS.get(zone,1.0)*SEVERITY_WEIGHTS.get(severity,1.0)))

def risk_color(score):
    return "🔴" if score>=70 else "🟡" if score>=40 else "🟢"

# ============================================================
# Main Pipeline
# ============================================================
def run_pipeline(uploaded_image, model_ppe, model_zone, model_harness, gemini, embedder, faiss_idx, groq_client):
    img_cv = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
    tmp_path = os.path.join(BASE_DIR, "tmp_input.jpg")
    cv2.imwrite(tmp_path, img_cv)

    ppe_results     = model_ppe(tmp_path,     conf=0.3, augment=True, verbose=False)[0]
    zone_results    = model_zone(tmp_path,    conf=0.4, augment=True, verbose=False)[0]
    harness_results = model_harness(tmp_path, conf=0.3, augment=True, verbose=False)[0]

    zone_boxes = [{"bbox":b.xyxy[0].tolist(),"zone_type":model_zone.names[int(b.cls)],"conf":float(b.conf)} for b in zone_results.boxes]
    persons, ppe_violations, frame_ctx_items = [], [], []

    for b in ppe_results.boxes:
        cls = model_ppe.names[int(b.cls)]; bbox = b.xyxy[0].tolist(); conf = float(b.conf)
        if cls=='Person': persons.append({"bbox":bbox,"conf":conf})
        elif cls in PPE_VIOLATION_CLASSES: ppe_violations.append({"type":cls,"bbox":bbox})
        elif cls in FRAME_CONTEXT_CLASSES: frame_ctx_items.append({"type":cls})
        # NMS: remove duplicate person bboxes
        filtered_persons = []
        for p in persons:
            duplicate = False
            for fp in filtered_persons:
                if iou(p["bbox"], fp["bbox"]) > 0.5:
                    duplicate = True
                    break
            if not duplicate:
                filtered_persons.append(p)
        persons = filtered_persons

    harness_dets = [{"type":model_harness.names[int(b.cls)],"bbox":b.xyxy[0].tolist()} for b in harness_results.boxes]

    frame_context    = list(set(i["type"] for i in frame_ctx_items))
    nearby_equipment = list(set(i["type"] for i in frame_ctx_items if i["type"] in ["machinery","vehicle"]))
    total_persons    = len(persons)
    total_violations = len(ppe_violations)+len([h for h in harness_dets if h["type"]=="no-harness"])
    hardhat_wearers  = sum(1 for b in ppe_results.boxes if model_ppe.names[int(b.cls)]=="Hardhat")
    vest_wearers     = sum(1 for b in ppe_results.boxes if model_ppe.names[int(b.cls)]=="Safety Vest")
    harness_wearers  = sum(1 for b in harness_results.boxes if model_harness.names[int(b.cls)]=="harness")
    best_zone        = max(zone_boxes, key=lambda z:z["conf"])["zone_type"] if zone_boxes else "unknown"
    norm_zone        = "safe" if best_zone=="unknown" else best_zone

    worker_records = []
    violation_persons = 0
    for i, person in enumerate(persons):
        pb = person["bbox"]; violations = []
        for v in ppe_violations:
            if is_inside(v["bbox"], pb):
                mapped = VIOLATION_MAP.get(v["type"])
                if mapped and mapped not in violations: violations.append(mapped)
        for h in harness_dets:
            if h["type"]=="no-harness" and is_inside(h["bbox"], pb):
                if "missing_harness" not in violations: violations.append("missing_harness")
        # Filter out missing_harness in non-elevated zones
        if norm_zone != "elevated-work-zone":
            violations = [v for v in violations if v != "missing_harness"]
        if not violations: continue
        violation_persons += 1
        x1,y1,x2,y2 = expand_crop(pb, img_cv.shape)
        crop = img_cv[y1:y2,x1:x2]
        crop_path = os.path.join(CROPS_DIR, f"live_worker_{i+1:03d}.jpg")
        cv2.imwrite(crop_path, crop)
        worker_records.append({"worker_id":f"worker_{i+1:03d}","bbox":pb,"crop_path":crop_path,"violations":violations,"conf":round(person["conf"],3),"zone":norm_zone})


    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt = f.read()

    results = []
    for w in worker_records:
        user_text = (
            f"Metadata:\n- worker_id: {w['worker_id']}\n- trigger_candidates: {w['violations']}\n"
            f"- zone: {w['zone']}\n- confidence: {w['conf']}\n"
            f"- total_persons_in_frame: {total_persons}\n- violation_persons_in_frame: {violation_persons}\n"
            f"- total_violations_in_frame: {total_violations}\n- hardhat_wearers_in_frame: {hardhat_wearers}\n"
            f"- vest_wearers_in_frame: {vest_wearers}\n- harness_wearers_in_frame: {harness_wearers}\n"
            f"- nearby_equipment: {nearby_equipment if nearby_equipment else 'none'}\n"
            f"- frame_context: {frame_context if frame_context else 'none'}\n"
        )
        crop_img  = Image.open(w["crop_path"])
        response  = gemini.models.generate_content(model="gemini-2.5-flash", contents=[prompt+"\n\n"+user_text, crop_img])
        raw       = response.text.strip().replace("```json","").replace("```","").strip()

        try:
            res = json.loads(raw)
        except:
            res = {"violation_types":w["violations"],"severity":"medium","description":raw,"zone":w["zone"]}

        vtypes = res.get("violation_types", w["violations"])
        if isinstance(vtypes, str): vtypes = [vtypes]
        vtypes = [t for t in vtypes if t in ALLOWED_VIOLATION_TYPES] or w["violations"]
        # Filter missing_harness in non-elevated zones
        if w["zone"] != "elevated-work-zone":
            vtypes = [v for v in vtypes if v != "missing_harness"]
        if not vtypes:
            vtypes = [v for v in w["violations"] if v != "missing_harness"]
        seen = []
        for t in vtypes:
            if t not in seen: seen.append(t)
        if len(seen)>1 and "unknown" in seen: seen.remove("unknown")

        severity   = res.get("severity","medium") if res.get("severity") in ALLOWED_SEVERITY else "medium"
        zone       = res.get("zone", w["zone"]) if res.get("zone") in ALLOWED_ZONES else w["zone"]
        risk_score = calculate_risk_score(seen, zone, severity)
        regs       = retrieve_per_violation(seen, embedder, faiss_idx, zone, nearby_equipment)
        citations  = list({r["section"] for r in regs})
        top_regs   = list({r["section"]:r for r in regs}.values())[:3]

        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        groq_prompt = f"""You are a construction site safety supervisor writing a formal incident report.
Date and Time: {now}
Worker ID: {w['worker_id']}
Detected Violations: {', '.join(v.replace('_',' ') for v in seen)}
Severity: {severity}
Work Zone: {zone}
Risk Score: {risk_score}/100
Description: {res.get('description','')}
Applicable WorkSafeBC Regulations: {', '.join(citations)}
Write a 2-3 sentence formal supervisor report in English. Use the exact date and time provided above instead of [current date]."""

        groq_res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":groq_prompt}],
            max_tokens=200
        )
        results.append({
            "worker_id":         w["worker_id"],
            "crop_path":         w["crop_path"],
            "violation_types":   seen,
            "severity":          severity,
            "risk_score":        risk_score,
            "zone":              zone,
            "description":       res.get("description",""),
            "top_regs":          top_regs,
            "citations":         citations,
            "supervisor_report": groq_res.choices[0].message.content.strip(),
        })

    return results, worker_records

# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="Construction Safety Monitor", page_icon="🦺", layout="wide")
st.title("🦺 Construction Site Safety Monitor")
st.markdown("Upload a construction site image to detect PPE violations and generate a safety report.")

model_ppe, model_zone, model_harness, gemini, embedder, faiss_idx, groq_client = load_models()

uploaded = st.file_uploader("📸 Upload Construction Site Image", type=["jpg","jpeg","png"])

if uploaded:
    image = Image.open(uploaded)

    st.subheader("📸 Uploaded Image")
    st.image(image, use_column_width=True)

    with st.spinner("🔍 Analyzing image... This may take a moment."):
        results, worker_records = run_pipeline(image, model_ppe, model_zone, model_harness, gemini, embedder, faiss_idx, groq_client)

    if not results:
        st.success("✅ No violations detected in this image.")
    else:
        st.markdown("---")
        st.subheader(f"⚠️ {len(results)} Violation(s) Detected")

        col1, col2, col3 = st.columns(3)
        col1.metric("Workers with Violations", len(results))
        col2.metric("Avg Risk Score", f"{sum(r['risk_score'] for r in results)//len(results)}/100")
        col3.metric("Zone", results[0]["zone"])

        st.markdown("---")

        for r in results:
            sev_icon  = {"high":"🔴","medium":"🟡","low":"🟢"}.get(r["severity"],"⚪")
            with st.expander(f"👷 {r['worker_id']} — {', '.join(VIOLATION_LABEL.get(v,v) for v in r['violation_types'])} {sev_icon}", expanded=True):
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.image(Image.open(r["crop_path"]), caption=f"Crop: {r['worker_id']}", use_column_width=True)

                with col2:
                    
                    st.markdown(f"**Risk Score:** {risk_color(r['risk_score'])} {r['risk_score']}/100")
                    st.markdown(f"**Zone:** {r['zone']}")
                    st.markdown(f"**Violations:** {', '.join(VIOLATION_LABEL.get(v,v) for v in r['violation_types'])}")
                    st.markdown("**Description:**")
                    st.info(r["description"])

                st.markdown("**📚 Applicable Regulations:**")
                for reg in r["top_regs"]:
                    st.markdown(f"- **Section {reg['section']}** (relevance: {reg['score']:.2f}): {reg['text']}")

                st.markdown("**📝 Supervisor Report:**")
                st.success(r["supervisor_report"])
