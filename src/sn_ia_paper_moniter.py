import arxiv
import csv
import os
from google import genai
from notion_client import Client
from datetime import datetime, timedelta, timezone

# --- CONFIGURATION ---
GEMINI_KEY = "AIzaSyCXVqmJtO9GyM1IZ3en9uqjGz72-17M7sY"
NOTION_TOKEN = "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"
NOTION_DATABASE_ID = "3308c1c13618805ea3b3f0010184b76c"
# 'cat' specifies the category. 
# We search for SNe Ia specifically within HE, CO, and GA.
SEARCH_QUERY = (
    '(cat:astro-ph.HE OR cat:astro-ph.CO OR cat:astro-ph.GA) AND '
    '(abs:"Type Ia" OR abs:"SN Ia" OR abs:"SNIa") AND '
    '(abs:progenitor OR abs:standardization OR abs:"host galaxy")'
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
                "AI Summary": {"rich_text": [{"text": {"content": data["summary"]}}]}
            }
        )
    except Exception as e:
        print(f"Notion Error: {e}")

def summarize(paper):
    prompt = f"Evaluate this SN Ia abstract. Return ONLY: 'Score: [0-10] | Summary: [1 sentence]'. \nAbstract: {paper.summary}"
    try:
        res = ai_client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
        # Simple parser for the "Score: X | Summary: Y" format
        parts = res.text.split("|")
        score = parts[0].replace("Score:", "").strip()
        summary = parts[1].replace("Summary:", "").strip()
        return score, summary
    except:
        return "0", "AI failed to process."

def main():
    arxiv_client = arxiv.Client()
    search = arxiv.Search(query=SEARCH_QUERY, max_results=5, sort_by=arxiv.SortCriterion.SubmittedDate)
    
    for paper in arxiv_client.results(search):
        # Only process papers from the last 24 hours
        if paper.published > datetime.now(timezone.utc) - timedelta(days=1):
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

if __name__ == "__main__":
    main()