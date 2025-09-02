# backend/app.py
import os, re
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Document parsing
import PyPDF2, pdfplumber, docx

# NLP + similarity
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional
try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except Exception:
    TEXTSTAT_AVAILABLE = False

# Init NLTK (quiet)
for r in ["punkt"]:
    try:
        nltk.download(r, quiet=True)
    except Exception:
        pass

ALLOWED = {"pdf","doc","docx"}
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

SKILL_CATEGORIES = {
    "programming_languages":["python","java","javascript","typescript","c","c++","c#","ruby","php","swift","kotlin","go","rust","scala","r","matlab","dart"],
    "web_technologies":["html","css","react","angular","vue","svelte","node","node.js","django","flask","spring","express","bootstrap","jquery","graphql","rest","soap","xml","json","next.js","tailwind"],
    "databases":["sql","mysql","postgresql","mongodb","oracle","redis","elasticsearch","sqlite","mariadb","dynamodb","neo4j","bigquery","snowflake","redshift"],
    "cloud_platforms":["aws","azure","gcp","docker","kubernetes","terraform","jenkins","github actions","gitlab ci","heroku","ansible","prometheus","grafana"],
    "ai_ml":["machine learning","deep learning","tensorflow","pytorch","keras","scikit-learn","opencv","nlp","computer vision","pandas","numpy","matplotlib","xgboost"],
    "devops_tools":["git","github","gitlab","bitbucket","ci/cd","helm","istio","monitoring","logging"],
    "soft_skills":["leadership","communication","teamwork","problem solving","project management","agile","scrum","critical thinking","time management","adaptability","creativity"]
}

def allowed_file(name): 
    return "." in name and name.rsplit(".",1)[1].lower() in ALLOWED

def clean_text(t): 
    return re.sub(r"\s+"," ", t or "").strip()

def text_from_pdf(path):
    txt = ""
    try:
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                txt += (p.extract_text() or "") + "\n"
    except Exception:
        pass
    if len(txt.strip()) < 50:
        try:
            r = PyPDF2.PdfReader(path)
            for pg in r.pages:
                txt += (pg.extract_text() or "") + "\n"
        except Exception:
            pass
    return clean_text(txt)

def text_from_docx(path):
    try:
        d = docx.Document(path)
        parts = [para.text for para in d.paragraphs]
        for table in d.tables:
            for row in table.rows:
                parts.append(" ".join(c.text for c in row.cells))
        return clean_text("\n".join(parts))
    except Exception:
        return ""

def extract_contact(text):
    email = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', text)
    phone = re.findall(r'(\+?\d[\d\s().-]{8,}\d)', text)
    linkedin = re.findall(r'(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+', text, flags=re.I)
    return {"email": email[0] if email else None, "phone": phone[0] if phone else None, "linkedin": linkedin[0] if linkedin else None}

def extract_education(text):
    degrees = ["bachelor","master","phd","mba","btech","mtech","b.e","bsc","msc","mca","bca"]
    year_pat = r'\b(19|20)\d{2}\b'
    out, seen = [], set()
    for s in sent_tokenize(text):
        ls = s.lower()
        if any(d in ls for d in degrees):
            yr = re.findall(year_pat, s)
            key = s.strip()[:120]
            if key not in seen:
                out.append({"degree": key, "year": yr[0] if yr else None})
                seen.add(key)
    return out[:5]

def extract_skills(text):
    found = {k: [] for k in SKILL_CATEGORIES}
    low = text.lower()
    for cat, skills in SKILL_CATEGORIES.items():
        for s in skills:
            if s in low and s not in found[cat]:
                found[cat].append(s)
    return found

def extract_experience(text):
    yrs = 0
    for pat in [r'(\d+)\+?\s*(?:years|yrs)\s*experience', r'experience[:\s]+(\d+)\+?\s*(?:years|yrs)']:
        m = re.findall(pat, text, flags=re.I)
        if m:
            try: yrs = max(yrs, max(int(x) for x in m))
            except: pass
    return {"positions": [], "total_years": yrs}

def extract_projects(text):
    keys = ["project","developed","designed","built","implemented","created"]
    out = []
    for s in sent_tokenize(text):
        if any(k in s.lower() for k in keys):
            out.append({"description": s.strip()[:200], "technologies": []})
    return out[:6]

def extract_certs(text):
    out = []
    for s in sent_tokenize(text):
        if any(w in s.lower() for w in ["certified","certification","certificate"]):
            out.append({"name": s.strip()[:120], "year": None, "active": True, "category": "general"})
    return out[:6]

def tfidf_sim(a,b):
    if not a or not b: return 0.0
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    X = vec.fit_transform([a,b])
    return float(cosine_similarity(X[0:1], X[1:2])[0][0])

def flatten(skdict):
    return set(x for arr in skdict.values() for x in arr)

def score_resume(text, skills, jd_text):
    sim = tfidf_sim(text, jd_text) if jd_text else 0.0
    jd_skills = extract_skills(jd_text) if jd_text else {k:[] for k in SKILL_CATEGORIES}
    flat_r = flatten(skills); flat_j = flatten(jd_skills)
    overlap = (len(flat_r & flat_j) / max(1, len(flat_j))) if flat_j else 0.0
    score = (sim*0.6 + overlap*0.4) * 100.0
    return max(0, min(100, round(score)))

def level(score):
    return "Exceptional" if score>=90 else "Excellent" if score>=80 else "Strong" if score>=70 else "Good" if score>=60 else "Qualified"

def analyze_one(fname, text, jd_text):
    contact = extract_contact(text)
    edu = extract_education(text)
    skills = extract_skills(text)
    exp = extract_experience(text)
    projs = extract_projects(text)
    certs = extract_certs(text)
    sc = score_resume(text, skills, jd_text)
    highlights = []
    if TEXTSTAT_AVAILABLE:
        try:
            highlights.append("Readability: " + (textstat.text_standard(text) or "n/a"))
        except Exception: pass
    if exp.get("total_years"): highlights.append(f"Experience mentioned: {exp['total_years']} years")
    if any(skills.values()): highlights.append(f"Detected {len(flatten(skills))} skills")
    return {
        "filename": fname,
        "matchScore": sc,
        "strength_level": level(sc),
        "comparison": "",
        "contact_info": contact,
        "education": edu,
        "skills": skills,
        "experience": exp,
        "projects": projs,
        "certifications": certs,
        "keyHighlights": highlights[:6],
        "recommendations": ["Tailor summary to JD keywords", "Ensure LinkedIn and contact are visible"]
    }

def load_text(path):
    ext = path.rsplit(".",1)[-1].lower()
    return text_from_pdf(path) if ext=="pdf" else text_from_docx(path)

@app.get("/health")
def health(): return jsonify({"ok": True})

@app.post("/upload")
def upload():
    jd_text = request.form.get("jd_text","").strip()
    if "jd_file" in request.files and request.files["jd_file"]:
        jd_file = request.files["jd_file"]
        if allowed_file(jd_file.filename):
            jn = secure_filename(jd_file.filename)
            jp = os.path.join(app.config["UPLOAD_FOLDER"], jn)
            jd_file.save(jp)
            jd_text2 = load_text(jp)
            if jd_text2: jd_text = (jd_text + "\n" + jd_text2).strip()

    files = request.files.getlist("files[]") or ([request.files["file"]] if "file" in request.files else [])
    if not files: return jsonify({"error":"No files uploaded"}), 400

    results = []
    for f in files:
        if not f or not allowed_file(f.filename): continue
        name = secure_filename(f.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], name)
        f.save(path)
        txt = load_text(path)
        results.append(analyze_one(name, txt, jd_text))

    results.sort(key=lambda x: x["matchScore"], reverse=True)
    scores = [r["matchScore"] for r in results]
    median = sorted(scores)[len(scores)//2] if scores else 0
    for i, r in enumerate(results, start=1):
        r["rank"] = i
        diff = r["matchScore"] - median
        r["comparison"] = (f"{abs(diff)} pts above median" if diff>=0 else f"{abs(diff)} pts below median")
    return jsonify({"total_resumes": len(results), "results": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
