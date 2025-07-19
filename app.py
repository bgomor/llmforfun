import os
import re
import streamlit as st
import openai
from dotenv import load_dotenv
from typing import Optional
from langchain_community.tools.tavily_search import TavilySearchResults
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header
import docx
import pdfplumber
from PyPDF2 import PdfReader
import requests
from streamlit_extras.let_it_rain import rain
import pandas as pd

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def job_search_page():
    st.set_page_config(
        page_title="BeAsst - UK Visa Sponsorship Jobs",
        page_icon=":uk:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    colored_header(
        label=":uk: UK Visa Sponsorship Job Search",
        description="Click the button below to retrieve all jobs in the UK that offer visa sponsorship.",
        color_name="violet-70"
    )
    add_vertical_space(1)

    if st.button("Search All UK Visa Sponsorship Jobs"):
        with st.spinner("Searching for UK visa sponsorship jobs..."):
            jobs = search_uk_visa_sponsorship_jobs("")
        if jobs:
            st.success(f"Found {len(jobs)} jobs with visa sponsorship in the UK.")
            for job in jobs:
                st.markdown(f"**[{job['title']}]({job['url']})**  \n"
                            f"**Company:** {job['company']}  \n"
                            f"**Location:** {job['location']}  \n"
                            f"**Summary:** {job['summary']}")
                st.markdown("---")
            # Download to Excel
            import io
            df = pd.DataFrame(jobs)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            output.seek(0)
            st.download_button(
                label="Download Jobs as Excel",
                data=output,
                file_name="uk_visa_sponsorship_jobs.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.info("No jobs found matching your criteria.")

def search_uk_visa_sponsorship_jobs(query):
    """
    Searches for UK jobs with visa sponsorship using the Adzuna API (free tier).
    Returns a list of job dicts.

    For production, you may consider partnerships or scraping (with permission), but for open-source/public tools, Adzuna is the most accessible and compliant option.
    """
    # You need to set your Adzuna app_id and app_key as environment variables
    ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
    ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        return []
    url = (
        f"https://api.adzuna.com/v1/api/jobs/gb/search/1"
        f"?app_id={ADZUNA_APP_ID}&app_key={ADZUNA_APP_KEY}"
        f"&results_per_page=100"
        f"&what={requests.utils.quote(query + ' visa sponsorship')}"
        f"&where=UK"
        f"&content-type=application/json"
    )
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()
        jobs = []
        for job in data.get("results", []):
            # Filter for explicit mention of visa sponsorship in description/title
            text = (job.get("description", "") + " " + job.get("title", "")).lower()
            if "visa sponsorship" in text or "sponsorship available" in text:
                jobs.append({
                    "title": job.get("title", "N/A"),
                    "company": job.get("company", {}).get("display_name", "N/A"),
                    "location": job.get("location", {}).get("display_name", "N/A"),
                    "summary": job.get("description", "")[:500] + "...",
                    "url": job.get("redirect_url", "#")
                })
        return jobs
    except Exception:
        return []

# Ensure only one Streamlit entry point is used for navigation

def run_app():
    st.set_page_config(
        page_title="BeAsst",
        page_icon=":sparkles:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to:",
        ("AI Code Reviewer", "CV & Cover Letter Reviewer", "UK Visa Sponsorship Job Search"),
        key="main_nav_radio"
    )
    if page == "AI Code Reviewer":
        main_with_selection()
    elif page == "CV & Cover Letter Reviewer":
        cv_review_page()
    elif page == "UK Visa Sponsorship Job Search":
        job_search_page()


# (Removed duplicate entry point block)
def extract_text_from_pdf(file):
    """
    Extracts text from a PDF file using pdfplumber, with a fallback to PyPDF2.
    Accepts a file-like object (e.g., from Streamlit uploader).
    Returns the extracted text as a string.
    """
    try:
        file.seek(0)
        with pdfplumber.open(file) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return text
    except Exception:
        try:
            file.seek(0)
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception as e:
            st.error(f"Failed to extract text from PDF: {e}")
            return ""

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_file(uploaded_file):
    if uploaded_file is None:
        return ""
    file_type = uploaded_file.type
    if file_type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    elif file_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
        return extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload a PDF or Word document.")
        return ""

def review_and_update_cv(cv_text, job_desc):
    prompt = (
        "You are an expert career coach and resume writer. "
        "Given the following CV and job description, review the CV and update it to better match the job description. "
        f"\n\nCV:\n{cv_text}\n\nJob Description:\n{job_desc}"
    )
    try:
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a professional resume writer and career coach."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2048,
            temperature=0.3
        )
        return response.choices[0].message.content.strip() if response.choices else None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def generate_cover_letter(improved_cv, job_desc):
    prompt = (
        "You are an expert cover letter writer. "
        "Given the following improved CV and job description, write a tailored cover letter for the job. "
        f"\n\nImproved CV:\n{improved_cv}\n\nJob Description:\n{job_desc}"
    )
    try:
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a professional cover letter writer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.3
        )
        return response.choices[0].message.content.strip() if response.choices else None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def cv_review_page():
    st.set_page_config(
        page_title="BeAsst - CV Reviewer",
        page_icon=":briefcase:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    colored_header(
        label=":briefcase: CV & Cover Letter Reviewer",
        description="Upload your CV and a job description to get an improved CV and a tailored cover letter.",
        color_name="violet-70"
    )
    add_vertical_space(1)

    st.markdown("### 1. Upload your CV (PDF or Word)")
    uploaded_cv = st.file_uploader("Upload CV", type=["pdf", "docx", "doc"], key="cv_upload")
    st.markdown("### 2. Paste the Job Description")
    job_desc = st.text_area("Job Description", height=200, placeholder="Paste the job description here...", key="job_desc_text_area")

    if st.button("Review & Update CV"):
        if not uploaded_cv or not job_desc.strip():
            st.warning("Please upload your CV and paste the job description.")
            return
        with st.spinner("Extracting CV text..."):
            cv_text = extract_text_from_file(uploaded_cv)
        if not cv_text.strip():
            st.error("Could not extract text from the uploaded CV.")
            return
        with st.spinner("Reviewing and updating your CV..."):
            improved_cv = review_and_update_cv(cv_text, job_desc)
        if improved_cv:
            st.subheader("Improved CV")
            st.text_area("Improved CV", improved_cv, height=400)
            st.download_button(
                label="Download Improved CV (txt)",
                data=improved_cv,
                file_name="improved_cv.txt",
                mime="text/plain"
            )
            with st.spinner("Generating cover letter..."):
                cover_letter = generate_cover_letter(improved_cv, job_desc)
            if cover_letter:
                st.subheader("Tailored Cover Letter")
                st.text_area("Cover Letter", cover_letter, height=300)
                st.download_button(
                    label="Download Cover Letter (txt)",
                    data=cover_letter,
                    file_name="cover_letter.txt",
                    mime="text/plain"
                )
        else:
            st.error("Failed to generate improved CV.")

# Add multi-page navigation
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to:",
        ("AI Code Reviewer", "CV & Cover Letter Reviewer", "UK Visa Sponsorship Job Search"),
        key="main_nav_radio_sidebar"
    )
    if page == "AI Code Reviewer":
        main_with_selection()
    elif page == "CV & Cover Letter Reviewer":
        cv_review_page()
    elif page == "UK Visa Sponsorship Job Search":
        job_search_page()


FALLBACK_BEST_PRACTICES = (
    "1. Use built-in data structures (list, dict, set, tuple) efficiently.\n"
    "2. Prefer comprehensions over loops for creating lists/dicts/sets.\n"
    "3. Use collections.defaultdict and Counter for counting tasks.\n"
    "4. Consider namedtuple or dataclasses for lightweight data containers.\n"
    "5. Use generators for large data to save memory.\n"
    "6. Avoid using mutable default arguments in functions.\n"
    "7. Leverage standard library modules like heapq, bisect, and itertools."
)


def fetch_latest_best_practices_langchain(query: str = "python data structures best practices", tavily_api_key: Optional[str] = TAVILY_API_KEY) -> str:
    """
    Fetches latest best practices from the web using LangChain's TavilySearchResults tool.
    Returns a summarized string of best practices.
    """
    if not tavily_api_key:
        return FALLBACK_BEST_PRACTICES

    try:
        search_tool = TavilySearchResults(api_key=tavily_api_key)
        results = search_tool.invoke({"query": query, "max_results": 5})
        best_practices = [result.get("content") for result in results if result.get("content")]
        if best_practices:
            return "\n".join(f"{i+1}. {bp}" for i, bp in enumerate(best_practices))
    except Exception as e:
        print(f"An error occurred while fetching best practices: {e}")

    return FALLBACK_BEST_PRACTICES

def get_user_code():
    return st.text_area(
        "Paste your Python code here:",
        height=300,
        placeholder="Enter your code...",
        key="user_code_input_area"
    )

def ask_about_suggestions_ui():
    st.markdown("**Have questions about the suggestions? Ask below!**")
    user_question = st.text_input(
        "Ask the AI about the suggestions or request clarification/changes:",
        placeholder="E.g., Why should I use a defaultdict here? Can you explain suggestion 2?",
        key="suggestion_question_input"
    )
    if user_question and st.button("Ask AI About Suggestions"):
        with st.spinner("Getting answer from AI..."):
            suggestions = st.session_state.get("suggestions", "")
            question_prompt = (
                "You are an expert Python code reviewer. "
                "Given the following code review suggestions and the user's question, "
                "answer the question or clarify the suggestions. "
                "If the user requests changes to the suggestions, update them accordingly and return the new suggestions as a numbered list. "
                "Otherwise, just answer the question concisely.\n\n"
                f"Suggestions:\n{suggestions}\n\nUser Question:\n{user_question}"
            )
            try:
                response = openai.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a senior Python developer and code reviewer."},
                        {"role": "user", "content": question_prompt}
                    ],
                    max_tokens=512,
                    temperature=0.2
                )
                answer = response.choices[0].message.content.strip() if response.choices else None
                if answer:
                    # If the answer looks like a new suggestion list, update session state
                    if re.match(r"^\d+\.\s", answer):
                        st.session_state["suggestions"] = answer
                        st.success("Suggestions updated based on your input!")
                        st.markdown("**Updated Suggestions:**")
                        st.markdown(answer)
                    else:
                        st.markdown("**AI Answer:**")
                        st.markdown(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")

def review_code(user_code):
    best_practices = fetch_latest_best_practices_langchain()
    review_prompt = (
        "You are an expert Python code reviewer. "
        "Review the following code and provide actionable suggestions for improvement. "
        "Here are some current best practices for reference:\n"
        f"{best_practices}\n"
        "Only comment on real improvements, not style preferences. "
        "Return the suggestions as a numbered list.\n\n"
        f"Code:\n{user_code}"
    )
    try:
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a senior Python developer and code reviewer."},
                {"role": "user", "content": review_prompt}
            ],
            max_tokens=512,
            temperature=0.2
        )
        return response.choices[0].message.content.strip() if response.choices else None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def display_suggestions(suggestions):
    st.subheader("AI Suggestions")
    st.markdown(suggestions)
    try:
        st.session_state["suggestions"] = suggestions
    except AttributeError:
        pass

def apply_suggestions(user_code, suggestions):
    with st.spinner("Applying suggestions..."):
        # Prompt for code rewriting
        rewrite_prompt = (
            "Given the following Python code and the suggestions below, "
            "rewrite the code to implement the suggestions. "
            "Return only the improved code, nothing else.\n\n"
            f"Original Code:\n{user_code}\n\n"
            f"Suggestions:\n{suggestions}"
        )
        try:
            response = openai.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a senior Python developer and code refactorer."},
                    {"role": "user", "content": rewrite_prompt}
                ],
                max_tokens=2048,
                temperature=0.2
            )
            improved_code = response.choices[0].message.content.strip() if response.choices else None
            if improved_code:
                # Remove markdown code fences if present
                improved_code = re.sub(r"^```(?:python)?\n?", "", improved_code)
                improved_code = re.sub(r"\n?```$", "", improved_code)
                st.subheader("Improved Code")
                st.code(improved_code, language="python")
                st.download_button(
                    label="Download Improved Code",
                    data=improved_code,
                    file_name="improved_code.py",
                    mime="text/x-python"
                )
        except Exception as e:
            st.error(f"An error occurred: {e}")

def select_suggestions_ui(suggestions_text):
    # Parse suggestions into a list
    suggestions_list = re.findall(r"\d+\.\s+(.*?)(?=\n\d+\.|\Z)", suggestions_text, re.DOTALL)
    if not suggestions_list:
        st.warning("No suggestions found to select.")
        return []

    st.markdown("**Select which suggestions to apply:**")
    selected = []
    for i, suggestion in enumerate(suggestions_list):
        if st.checkbox(f"{i+1}. {suggestion.strip()}", key=f"suggestion_{i}"):
            selected.append(suggestion.strip())
    return selected

def main_with_selection():
    st.set_page_config(
        page_title="BeAsst - AI Code Reviewer",
        page_icon=":sparkles:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    colored_header(
        label=":sparkles: AI Code Reviewer",
        description="Paste your code below and get AI-powered suggestions to improve it!",
        color_name="violet-70"
    )

    add_vertical_space(1)

    user_code = get_user_code()

    if "reviewed_code" not in st.session_state:
        st.session_state["reviewed_code"] = user_code

    if st.button("Review Code"):
        suggestions = review_code(user_code)
        if suggestions:
            display_suggestions(suggestions)
            st.session_state["suggestions"] = suggestions
            st.session_state["reviewed_code"] = user_code

    if "suggestions" in st.session_state and st.session_state["suggestions"]:
        add_vertical_space(2)
        st.markdown("**Would you like the AI to apply some suggestions and rewrite your code?**")
        selected_suggestions = select_suggestions_ui(st.session_state["suggestions"])
        if selected_suggestions:
            if st.button("Apply Selected Suggestions"):
                apply_suggestions(st.session_state["reviewed_code"], "\n".join(selected_suggestions))
        if st.button("Re-review my code"):
            # Allow user to re-review, possibly after editing code or suggestions
            st.session_state.pop("suggestions", None)
            st.rerun()
        # Allow the user to ask questions about the suggestions and update them
        ask_about_suggestions_ui()
        st.rerun()

if __name__ == "__main__":
    run_app()
    # Add custom CSS for improved UI
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #7c3aed;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1.5em;
            font-weight: 600;
            font-size: 1.1em;
            border: none;
            transition: background 0.2s;
            margin: 0.25em 0.5em 0.25em 0;
        }
        .stButton>button:hover {
            background-color: #a78bfa;
            color: #22223b;
        }
        .stDownloadButton>button {
            background-color: #10b981;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1.5em;
            font-weight: 600;
            font-size: 1.1em;
            border: none;
            transition: background 0.2s;
            margin: 0.25em 0.5em 0.25em 0;
        }
        .stDownloadButton>button:hover {
            background-color: #34d399;
            color: #22223b;
        }
        .stTextArea textarea {
            background-color: #f3f0ff;
            border-radius: 8px;
            font-size: 1.05em;
            font-family: 'Fira Mono', 'Menlo', 'Monaco', 'Consolas', monospace;
            padding: 1em;
            border: 1px solid #a78bfa;
        }
        .stCheckbox>label {
            font-size: 1.05em;
            padding-left: 0.5em;
        }
        .stSubheader, .stMarkdown h2 {
            color: #7c3aed;
        }
        </style>
    """, unsafe_allow_html=True)

