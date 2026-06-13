from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# --- 1. INITIALIZE API ---
app = FastAPI(title="SIM Anomaly Detection API", version="1.0")

# CRITICAL: Allow React (port 5173) to communicate with this Python API (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. LOAD DATABASE & AI MODELS ---
print("Loading National Database & AI Vectorizers...")
try:
    df = pd.read_csv('synthetic_sim_dataset_1000.csv')
    df['NIN'] = df['NIN'].astype(str)
    df['Combined_Text'] = df['First_Name'].str.lower() + " " + df['Last_Name'].str.lower() + " " + df['NIN']
    
    # We fit a fresh vectorizer for similarity checking
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
    db_matrix = vectorizer.fit_transform(df['Combined_Text'])
    print("✅ System Ready.")
except Exception as e:
    print(f"❌ Error loading database: {e}")

# --- 3. DEFINE INCOMING DATA STRUCTURE ---
class SIMRegistration(BaseModel):
    firstName: str
    lastName: str
    nin: str

# --- 4. CREATE THE VERIFICATION ENDPOINT ---
@app.post("/api/verify")
async def verify_sim(request: SIMRegistration):
    # Simulate a slight processing delay so the frontend spinner looks cool
    time.sleep(1.0)
    
    incoming_nin = str(request.nin).strip()
    incoming_fname = request.firstName.strip().lower()
    incoming_lname = request.lastName.strip().lower()

    # RULE 1: STRICT DETERMINISTIC CHECK (Exact NIN Match)
    existing_nin = df[df['NIN'] == incoming_nin]
    if not existing_nin.empty:
        return {
            "status": "error_nin",
            "message": "Identity Fraud Detected",
            "conflictingRecord": {
                "name": f"{existing_nin.iloc[0]['First_Name'].title()} {existing_nin.iloc[0]['Last_Name'].title()}",
                "nin": existing_nin.iloc[0]['NIN']
            }
        }

    # RULE 2: AI FUZZY DETECTION (Typo/Manipulation)
    new_record = f"{incoming_fname} {incoming_lname} {incoming_nin}"
    new_vector = vectorizer.transform([new_record])
    
    similarities = cosine_similarity(new_vector, db_matrix).flatten()
    max_sim_score = float(np.max(similarities))
    best_match_idx = int(np.argmax(similarities))
    
    # If mathematically over 75% similar, flag it!
    if max_sim_score > 0.75:
        return {
            "status": "error_fuzzy",
            "message": "Fuzzy Duplicate Detected",
            "matchScore": round(max_sim_score * 100, 1),
            "conflictingRecord": {
                "name": f"{df.iloc[best_match_idx]['First_Name'].title()} {df.iloc[best_match_idx]['Last_Name'].title()}",
                "nin": str(df.iloc[best_match_idx]['NIN'])
            }
        }

    # RULE 3: APPROVED
    return {
        "status": "success",
        "message": "Unique Identity Verified"
    }