import os
import re
import pandas as pd
import streamlit as st
import openai
from dotenv import load_dotenv
from typing import Optional
from langchain_community.tools.tavily_search import TavilySearchResults
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header

# Load environment variables
load_dotenv()

# Constants
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
FALLBACK_BEST_PRACTICES = (
    "1. Use built-in data structures (list, dict, set, tuple) efficiently.\n"
    "2. Prefer comprehensions over loops for creating lists/dicts/sets.\n"
    "3. Use collections.defaultdict and Counter for counting tasks.\n"
    "4. Consider namedtuple or dataclasses for lightweight data containers.\n"
    "5. Use generators for large data to save memory.\n"
    "6. Avoid using mutable default arguments in functions.\n"
    "7. Leverage standard library modules like heapq, bisect, and itertools."
)

openai.api_key = OPENAI_API_KEY

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
        best_practices = [result.get("content") for result in results.get("results", []) if result.get("content")]
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
        key="user_code_text_area"
    )

def review_code(user_code):
    if not user_code.strip():
        st.toast("Please enter some code to review.", icon="⚠️")
        return None

    with st.spinner("Analyzing your code with AI..."):
        # Fetch latest best practices using LangChain
        best_practices = fetch_latest_best_practices_langchain()
        # Prompt for code review
        review_prompt = (
            "You are an expert Python code reviewer. "
            "Given the following code, provide a concise list of suggestions to make it meet Python best practices. "
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

if __name__ == "__main__":
    main_with_selection()
    # Add custom CSS for improved UI
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

    # Fixed sidebar with app details and instructions
    with st.sidebar:
        # Centered LLM icon at the top
        st.markdown(
            """
            <div style="display: flex; justify-content: left; align-items: center; margin-bottom: 1em;">
                <img src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/72x72/1f916.png" alt="LLM Icon" width="64" height="64">
            </div>
            """,
            unsafe_allow_html=True
        )
        st.title("About BeAsst")
        st.markdown(
            """
            **BeAsst - AI Code Reviewer**  
            This app uses OpenAI and LangChain to review your Python code and suggest improvements based on the latest best practices.

            **How it works:**
            1. **Paste your Python code** in the main area.
            2. Click **Review Code** to get AI-powered suggestions.
            3. **Select** which suggestions you want to apply.
            4. Click **Apply Selected Suggestions** to get improved code.
            5. Download the improved code or re-review as needed.

            _Your code is processed securely and only for the purpose of providing suggestions and improvements._
            """
        )
        st.info("Tip: Only real improvements are suggested, not style preferences.")