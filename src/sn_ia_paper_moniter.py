import arxiv
import csv
import os
from google import genai
from notion_client import Client
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# --- CONFIGURATION ---
GEMINI_KEY = "AIzaSyCXVqmJtO9GyM1IZ3en9uqjGz72-17M7sY"
NOTION_TOKEN = "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"
NOTION_DATABASE_ID = "3308c1c13618805ea3b3f0010184b76c"
# 'cat' specifies the category. 
# We search for SNe Ia specifically within HE, CO, and GA.
# Grouping categories and terms correctly with parentheses
SEARCH_QUERY = (
    '(cat:astro-ph.CO OR cat:astro-ph.HE OR cat:astro-ph.GA) AND '
    '("Type Ia" OR "SN Ia" OR "SNIa") AND '
    '("host galaxy" OR "standardization" OR "standardisation" OR "mass step" OR "environmental correlation")'
)
# Initialize Clients
ai_client = genai.Client(api_key=GEMINI_KEY)
notion = Client(auth=NOTION_TOKEN)

def save_to_csv(data):
    file_exists = os.path.isfile("sn_ia_research_log.csv")
    with open("sn_ia_research_log.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "title", "link", "score", "summary"])
        if not file_exists: writer.writeheader()
        writer.writerow(data)

def push_to_notion(data):
    try:
        notion.pages.create(
            parent={"database_id": NOTION_DATABASE_ID},
            properties={
                "Name": {"title": [{"text": {"content": data["title"]}}]},
                "Link": {"url": data["link"]},
                "Score": {"number": int(data["score"])},
                "AI Summary": {"rich_text": [{"text": {"content": data["summary"]}}]},
                "Date Accessed": {"date": {"start":data["date"]} }
            }
        )
    except Exception as e:
        print(f"Notion Error: {e}")

def summarize(paper):
    # This structured prompt uses "Chain of Thought" for better scoring
    prompt = f"""
    You are an expert Senior Researcher in Observational Cosmology. 
    Evaluate this abstract for an Honours student researching the 'Standardization of Type Ia Supernovae and their Correlations with Host Galaxy Properties'.

    CRITERIA FOR SCORING:
    - 10: Direct analysis of SN Ia luminosity correlations with host mass, metallicity, or star-formation rate (e.g., the 'mass step').
    - 8: New light-curve fitting methods (SALT3/Sugar) or large SN Ia samples (DES, LSST, Pantheon+).
    - 5: General SN Ia physics or progenitor studies without host galaxy context.
    - 1: Other transients (TDEs, SN II) or general cosmology without SNe Ia.

    PAPER TO EVALUATE:
    Title: {paper.title}
    Abstract: {paper.summary}

    OUTPUT FORMAT (STRICT):
    Score: [0-10] | Summary: [1 sentence max focusing on the host-galaxy or standardization takeaway]
    """
    
    try:
        res = ai_client.models.generate_content(
            model="gemini-2.5-flash-lite", 
            contents=prompt
        )
        # Use a more robust split in case the AI adds extra text
        text = res.text.strip()
        if "|" in text:
            parts = text.split("|")
            score = parts[0].replace("Score:", "").strip()
            summary = parts[1].replace("Summary:", "").strip()
        else:
            score, summary = "5", text # Fallback
        return score, summary
    except Exception as e:
        print(f"Summarize Error: {e}")
        return "0", "AI failed to process."

def main():
    arxiv_client = arxiv.Client()
    search = arxiv.Search(query=SEARCH_QUERY, max_results=100, sort_by=arxiv.SortCriterion.SubmittedDate)
    print("Starting search...")
    
    count = 0
    for paper in arxiv_client.results(search):
        count += 1
        print(f"Found paper: {paper.title}")
        print(f"Published: {paper.published}")
        # Only process papers from the last 24 hours
        if paper.updated > datetime.now(tz=ZoneInfo("Australia/Sydney")) - timedelta(days=1):
            print(f"Processing: {paper.title}")
            score, summary = summarize(paper)
            
            payload = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "title": paper.title,
                "link": paper.entry_id,
                "score": score,
                "summary": summary
            }
            
            save_to_csv(payload)
            push_to_notion(payload)
            print(f"Done! Score: {score}")
        else:
            print(f"Skipped (too old): {paper.title}")
    
    print(f"Search complete. Found {count} papers.")

if __name__ == "__main__":
    main()