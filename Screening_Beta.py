# Import necessary libraries
import streamlit as st
import pandas as pd
import aiohttp
import asyncio
import logging
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from bs4 import BeautifulSoup
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import rispy  # For parsing RIS and NBIB files
import bibtexparser  # For parsing BibTeX files
import re

# Configure logging
logging.basicConfig(
    filename="sciassist_screening.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Set Streamlit page configuration
st.set_page_config(
    page_title="Comprehensive SR & MA Screening Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600&family=Open+Sans:wght@300;400;600&display=swap');

    /* Base Styles */
    body {
        font-family: 'Open Sans', sans-serif;
        background-color: #f0f2f5;
        color: #333;
    }

    .stApp {
        background-color: #ffffff;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Montserrat', sans-serif;
        color: #333;
        margin-top: 1em;
        margin-bottom: 0.5em;
    }

    h1 {
        font-size: 2.5em;
        font-weight: 600;
        border-bottom: 3px solid #4CAF50;
        padding-bottom: 10px;
    }

    h2 {
        font-size: 2em;
        font-weight: 500;
        color: #4CAF50;
    }

    /* Inputs and Textareas */
    textarea, input[type="text"], select, input[type="password"] {
        background-color: #fff;
        border: 1px solid #ccc;
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
        width: 100%;
        box-sizing: border-box;
        transition: border-color 0.3s ease;
        margin-bottom: 1em;
    }

    textarea:focus, input[type="text"]:focus, select:focus, input[type="password"]:focus {
        border-color: #4CAF50;
        outline: none;
    }

    /* Buttons */
    .stButton > button {
        background-color: #4CAF50;
        color: #fff;
        border: none;
        border-radius: 8px;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: 500;
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.1s ease;
        margin-top: 1em;
    }

    .stButton > button:hover {
        background-color: #45A049;
        transform: translateY(-2px);
    }

    /* Info Box */
    .info-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4CAF50;
        border-radius: 8px;
        padding: 20px;
        margin: 20px 0;
        color: #2e7d32;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .info-box h4 {
        margin-top: 0;
        font-weight: 600;
    }

    /* Results Container */
    .results-container {
        background-color: #fafafa;
        border-radius: 8px;
        padding: 20px;
        margin: 20px 0;
        color: #333;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* Data Visualization */
    .js-plotly-plot {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* Dataframe Styling */
    .dataframe {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
    }

    .dataframe thead {
        background-color: #4CAF50;
        color: #fff;
    }

    .dataframe th, .dataframe td {
        padding: 12px;
        border: 1px solid #ddd;
        text-align: left;
    }

    .dataframe tr:nth-child(even) {
        background-color: #f9f9f9;
    }

    /* Word Cloud */
    .wordcloud {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }

    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 6px;
    }

    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 6px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    .stApp > div {
        animation: fadeIn 0.5s ease-out;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        h1 {
            font-size: 2em;
        }

        h2 {
            font-size: 1.75em;
        }

        .stButton > button {
            padding: 10px 20px;
            font-size: 14px;
        }

        textarea, input[type="text"], select, input[type="password"] {
            font-size: 14px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize global variables
embedding_model = None
classifier = None

@st.cache_resource(show_spinner=True)
def load_models():
    """
    Load the embedding model and the zero-shot classification pipeline.
    """
    global embedding_model, classifier
    logger.info("Loading models...")
    try:
        # Load embedding model
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        
        # Load zero-shot classifier
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )
        
        logger.info("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        st.error("An error occurred while loading the models. Please check the logs.")
    return embedding_model, classifier

def get_exclusion_regex(exclusion_terms):
    """
    Create a compiled regex pattern to match exclusion terms, handling plural and hyphenated variations.
    """
    regex_patterns = []
    for term in exclusion_terms:
        escaped_term = re.escape(term)
        # Match exact word, plural (s?), or hyphenated variations
        pattern = rf"\b{escaped_term}s?\b|\b{escaped_term}-analysis\b|\b{escaped_term}-based\b"
        regex_patterns.append(pattern)
    combined_pattern = "|".join(regex_patterns)
    return re.compile(combined_pattern, re.IGNORECASE)

def is_valid_study_design(article, exclusion_regex, classifier, nlp_threshold=0.5):
    """
    Determines if an article is valid based on exclusion terms and NLP classification.

    Args:
        article (dict): Article data containing title and abstract.
        exclusion_regex (re.Pattern): Compiled regex pattern for exclusion terms.
        classifier (pipeline): Hugging Face zero-shot classifier.
        nlp_threshold (float): Confidence threshold for NLP classification.

    Returns:
        bool: True if the study is valid, False otherwise.
        list: List of reasons for exclusion.
    """
    reasons = []
    pub_types = article.get("pub_type", "")
    title = article.get("title", "")
    abstract = article.get("abstract", "")
    
    combined_text = f"{pub_types} {title} {abstract}"
    if exclusion_regex.search(combined_text):
        reasons.append("Exclusion terms found in publication types, title, or abstract.")
        return False, reasons
    
    # Use NLP classifier to detect study type
    sequence = f"{title}. {abstract}"
    try:
        classification = classifier(sequence, candidate_labels=["Yes", "No"])
        if classification['labels'][0] == "Yes" and classification['scores'][0] >= nlp_threshold:
            reasons.append("Identified as a meta-analysis or systematic review by NLP classifier.")
            return False, reasons
    except Exception as e:
        logger.error(f"NLP classification error: {e}")
        # Do not exclude based on NLP if an error occurs
        pass
    
    return True, reasons

def evaluate_pico_relevance(article, user_pico):
    """
    Evaluates the relevance of an article based on user-defined PICO elements.

    Args:
        article (dict): Article data containing title and abstract.
        user_pico (dict): User-defined PICO elements.

    Returns:
        float: Overall relevance score.
        dict: Individual relevance scores for each PICO element.
    """
    text = f"{article.get('title', '')} {article.get('abstract', '')}"
    relevance_scores = {}
    for element, user_input in user_pico.items():
        if user_input.strip():
            user_embedding = embedding_model.encode(user_input, convert_to_tensor=True)
            article_embedding = embedding_model.encode(text, convert_to_tensor=True)
            similarity = util.cos_sim(user_embedding, article_embedding).item()
            relevance_scores[element] = similarity
        else:
            relevance_scores[element] = None  # Ignore this element in averaging
    non_null_scores = [v for v in relevance_scores.values() if v is not None]
    if non_null_scores:
        overall_relevance = sum(non_null_scores) / len(non_null_scores)
    else:
        overall_relevance = 0.0
    return overall_relevance, relevance_scores

def evaluate_topic_relevance(article, study_topic):
    """
    Evaluates the relevance of an article based on the overall study topic.

    Args:
        article (dict): Article data containing title and abstract.
        study_topic (str): User-defined study topic.

    Returns:
        float: Similarity score between the article and study topic.
    """
    text = f"{article.get('title', '')} {article.get('abstract', '')}"
    article_embedding = embedding_model.encode(text, convert_to_tensor=True)
    topic_embedding = embedding_model.encode(study_topic, convert_to_tensor=True)
    similarity = util.cos_sim(article_embedding, topic_embedding).item()
    return similarity

def screen_articles(articles, user_pico, study_topic, exclusion_regex, classifier, relevance_threshold=0.5, pico_weight=0.5):
    """
    Screens articles based on exclusion criteria and relevance scores.

    Args:
        articles (list): List of articles to screen.
        user_pico (dict): User-defined PICO elements.
        study_topic (str): User-defined study topic.
        exclusion_regex (re.Pattern): Compiled regex pattern for exclusion terms.
        classifier (pipeline): Hugging Face zero-shot classifier.
        relevance_threshold (float): Threshold to determine article inclusion.
        pico_weight (float): Weighting factor for PICO relevance.

    Returns:
        list: Screened articles.
        list: Excluded articles.
        dict: Exclusion reasons and their counts.
    """
    screened_articles = []
    excluded_articles = []
    exclusion_reasons = {}
    for article in articles:
        is_valid, reasons = is_valid_study_design(article, exclusion_regex, classifier)
        if is_valid:
            pico_relevance, pico_scores = evaluate_pico_relevance(article, user_pico)
            topic_relevance = evaluate_topic_relevance(article, study_topic)
            overall_relevance = pico_weight * pico_relevance + (1 - pico_weight) * topic_relevance
            if overall_relevance >= relevance_threshold:
                article["relevance_score"] = overall_relevance
                article["pico_scores"] = pico_scores
                article["topic_relevance"] = topic_relevance
                article["study_id"] = f"{article.get('first_author_last_name', '')}{article.get('year', '')}"
                screened_articles.append(article)
        else:
            excluded_articles.append(article)
            for reason in reasons:
                exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1
    return sorted(screened_articles, key=lambda x: x["relevance_score"], reverse=True), excluded_articles, exclusion_reasons

async def fetch_pubmed_ids(query, email, api_key=None):
    """
    Fetches PubMed IDs based on the search query.

    Args:
        query (str): PubMed search query.
        email (str): User's email.
        api_key (str, optional): NCBI API key.

    Returns:
        list: List of PubMed IDs.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": 10000,  # Consider reducing if performance issues arise
        "usehistory": "y",
        "retmode": "json",
        "sort": "relevance",
        "email": email,
        "tool": "ComprehensiveSRScreeningTool",
    }
    if api_key:
        params["api_key"] = api_key

    async with aiohttp.ClientSession() as session:
        async with session.get(base_url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data["esearchresult"]["idlist"]
            else:
                logger.error(f"Failed to fetch PubMed IDs: {response.status}")
                return []

async def fetch_pubmed_details(ids, email, api_key=None):
    """
    Fetches detailed PubMed data for given IDs.

    Args:
        ids (list): List of PubMed IDs.
        email (str): User's email.
        api_key (str, optional): NCBI API key.

    Returns:
        str: XML data as a string.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "xml",
        "email": email,
        "tool": "ComprehensiveSRScreeningTool",
    }
    if api_key:
        params["api_key"] = api_key

    async with aiohttp.ClientSession() as session:
        async with session.get(base_url, params=params) as response:
            if response.status == 200:
                return await response.text()
            else:
                logger.error(f"Failed to fetch PubMed details: {response.status}")
                return None

def parse_pubmed_xml(xml_string):
    """
    Parses PubMed XML data into a list of article dictionaries.

    Args:
        xml_string (str): XML data as a string.

    Returns:
        list: List of articles with relevant details.
    """
    soup = BeautifulSoup(xml_string, "xml")
    articles = []
    for article in soup.find_all("PubmedArticle"):
        pmid = article.find("PMID").text if article.find("PMID") else ""
        title = article.find("ArticleTitle").text if article.find("ArticleTitle") else ""
        abstract = (
            " ".join([abstract_text.text for abstract_text in article.find_all("AbstractText")])
            if article.find("AbstractText")
            else ""
        )
        pub_type = [pt.text for pt in article.find_all("PublicationType")]
        authors = []
        for author in article.find_all("Author"):
            last_name = author.find("LastName")
            initials = author.find("Initials")
            if last_name and initials:
                authors.append(f"{last_name.text} {initials.text}")
            elif last_name:
                authors.append(last_name.text)
        year = ""
        if article.find("PubDate"):
            pub_date = article.find("PubDate")
            if pub_date.find("Year"):
                year = pub_date.find("Year").text
            elif pub_date.find("MedlineDate"):
                year = pub_date.find("MedlineDate").text[:4]

        articles.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "pub_type": ", ".join(pub_type),
                "authors": authors,
                "first_author_last_name": authors[0].split()[0] if authors else "",
                "year": year,
            }
        )
    return articles

def parse_pubmed_format(file_content):
    """
    Parses PubMed .txt (MEDLINE) format into a list of article dictionaries.

    Args:
        file_content (str): File content as a string.

    Returns:
        list: List of articles with relevant details.
    """
    articles = []
    entries = file_content.strip().split('\nPMID- ')
    for entry in entries:
        if not entry.strip():
            continue
        article = {}
        authors = []
        lines = ('PMID- ' + entry).strip().split('\n')
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            if line.startswith('PMID- '):
                article['pmid'] = line.replace('PMID- ', '').strip()
            elif line.startswith('TI  - '):
                title_lines = [line.replace('TI  - ', '').strip()]
                idx += 1
                while idx < len(lines) and lines[idx].startswith('      '):
                    title_lines.append(lines[idx].strip())
                    idx += 1
                article['title'] = ' '.join(title_lines)
                continue
            elif line.startswith('AB  - '):
                abstract_lines = [line.replace('AB  - ', '').strip()]
                idx += 1
                while idx < len(lines) and lines[idx].startswith('      '):
                    abstract_lines.append(lines[idx].strip())
                    idx += 1
                article['abstract'] = ' '.join(abstract_lines)
                continue
            elif line.startswith('AU  - '):
                authors.append(line.replace('AU  - ', '').strip())
            elif line.startswith('DP  - '):
                article['publication_date'] = line.replace('DP  - ', '').strip()
            elif line.startswith('PT  - '):
                pub_type_lines = [line.replace('PT  - ', '').strip()]
                idx += 1
                while idx < len(lines) and lines[idx].startswith('      '):
                    pub_type_lines.append(lines[idx].strip())
                    idx += 1
                article['pub_type'] = ', '.join(pub_type_lines)
                continue
            idx += 1
        if authors:
            article['authors'] = authors
            article['first_author_last_name'] = (
                authors[0].split(',')[0] if ',' in authors[0] else authors[0].split()[-1]
            )
            article['year'] = article.get('publication_date', '').split()[0]
        if article:
            articles.append(article)
    return articles

def parse_ris_format(file_content):
    """
    Parses RIS format into a list of article dictionaries.

    Args:
        file_content (str): RIS file content as a string.

    Returns:
        list: List of articles with relevant details.
    """
    try:
        records = rispy.loads(file_content)
    except Exception as e:
        logger.error(f"RIS parsing error: {e}")
        return []
    articles = []
    for record in records:
        title = record.get('title', '')
        abstract = record.get('abstract', '')
        authors = record.get('authors', [])
        pub_type = record.get('type_of_reference', '')
        year = record.get('year', '')
        first_author_last_name = authors[0].split()[-1] if authors else ''
        articles.append({
            'title': title,
            'abstract': abstract,
            'authors': authors,
            'pub_type': pub_type,
            'year': year,
            'first_author_last_name': first_author_last_name
        })
    return articles

def parse_bibtex_format(file_content):
    """
    Parses BibTeX format into a list of article dictionaries.

    Args:
        file_content (str): BibTeX file content as a string.

    Returns:
        list: List of articles with relevant details.
    """
    try:
        parser = bibtexparser.loads(file_content)
    except Exception as e:
        logger.error(f"BibTeX parsing error: {e}")
        return []
    articles = []
    for entry in parser.entries:
        title = entry.get('title', '')
        abstract = entry.get('abstract', '')
        authors = [author.strip() for author in entry.get('author', '').split(' and ')]
        pub_type = entry.get('ENTRYTYPE', '')
        year = entry.get('year', '')
        first_author_last_name = authors[0].split()[-1] if authors else ''
        articles.append({
            'title': title,
            'abstract': abstract,
            'authors': authors,
            'pub_type': pub_type,
            'year': year,
            'first_author_last_name': first_author_last_name
        })
    return articles

def visualize_relevance_distribution(screened_articles):
    """
    Visualizes the distribution of relevance scores among screened articles.

    Args:
        screened_articles (list): List of screened articles.
    """
    if not screened_articles:
        st.warning("No articles passed the screening to visualize relevance distribution.")
        return
    relevance_scores = [article["relevance_score"] for article in screened_articles]
    fig = px.histogram(
        x=relevance_scores,
        nbins=20,
        title="Distribution of Relevance Scores",
        labels={'x': 'Relevance Score', 'y': 'Number of Articles'},
        template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)

def visualize_pico_radar(user_pico):
    """
    Visualizes the length of PICO elements using a radar chart.

    Args:
        user_pico (dict): User-defined PICO elements.
    """
    categories = [key for key in user_pico.keys() if user_pico[key].strip()]
    values = [len(user_pico[category].split()) for category in categories]
    if not categories:
        st.warning("No PICO elements to visualize.")
        return

    fig = go.Figure(
        data=go.Scatterpolar(r=values, theta=categories, fill="toself", name="PICO Elements")
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(values) + 1])),
        showlegend=False,
        title="PICO Elements Length",
        template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)

def visualize_exclusion_reasons(exclusion_reasons):
    """
    Visualizes the reasons for excluding articles.

    Args:
        exclusion_reasons (dict): Dictionary with exclusion reasons and their counts.
    """
    if not exclusion_reasons:
        st.info("No articles were excluded.")
        return
    reasons = list(exclusion_reasons.keys())
    counts = list(exclusion_reasons.values())
    fig = px.bar(
        x=reasons,
        y=counts,
        title="Exclusion Reasons",
        labels={'x': 'Reason', 'y': 'Number of Articles'},
        template='plotly_dark'
    )
    st.plotly_chart(fig, use_container_width=True)

def main():
    global embedding_model, classifier

    st.title("Comprehensive Systematic Review & Meta-Analysis Screening Tool")

    # Load models
    embedding_model, classifier = load_models()

    if embedding_model is None or classifier is None:
        st.error("Failed to load necessary models. Please try reloading the page.")
        return

    st.header("Step 1: Define Research Topic and PICO")

    study_topic = st.text_area("Enter your research topic:", height=100)

    st.subheader("PICO Elements")
    col1, col2 = st.columns(2)
    with col1:
        population = st.text_input("Population:")
        intervention = st.text_input("Intervention:")
    with col2:
        comparison = st.text_input("Comparison:")
        outcome = st.text_input("Outcome:")

    user_pico = {
        "Population": population,
        "Intervention": intervention,
        "Comparison": comparison,
        "Outcome": outcome,
    }

    if st.button("Generate PICO Visualization"):
        if any(user_pico.values()):
            visualize_pico_radar(user_pico)
        else:
            st.warning("Please fill in at least one PICO element.")

    st.header("Step 2: Article Input")

    input_method = st.radio(
        "Choose input method:", ("PubMed Search", "Upload Search Results"), index=0
    )

    if input_method == "PubMed Search":
        query = st.text_area("Enter your PubMed search query:", height=100)
        email = st.text_input("Enter your email (required for PubMed API):")
        api_key = st.text_input(
            "Enter your NCBI API key (optional):",
            type="password",
            help="Providing an API key increases the request rate limit.",
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload your search results",
            type=["txt", "nbib", "ris", "bib"],
            help="Accepted formats: .txt, .nbib, .ris, .bib"
        )

    st.header("Step 3: Customize Exclusion Terms")

    default_exclusion_terms = [
        "review",
        "meta-analysis",
        "systematic review",
        "comment",
        "editorial",
        "letter",
        "abstract",
        "protocol",
        "case report",
        "case series",
        "guideline",
        "retracted"
    ]

    exclusion_terms = st.text_area(
        "Enter exclusion terms separated by commas (e.g., review, meta-analysis, systematic review):",
        value=", ".join(default_exclusion_terms),
        height=100,
        help="These terms will be used to exclude non-eligible studies."
    )

    # Process exclusion terms
    exclusion_terms_list = [term.strip() for term in exclusion_terms.split(",") if term.strip()]
    exclusion_regex = get_exclusion_regex(exclusion_terms_list)

    relevance_threshold = st.slider(
        "Set relevance threshold for screening (lower to include more articles):", 0.0, 1.0, 0.5, 0.01
    )

    if st.button("Run Comprehensive Screening"):
        if not study_topic.strip() or not any([v.strip() for v in user_pico.values()]):
            st.error("Please enter the study topic and at least one PICO element.")
            return

        if input_method == "PubMed Search":
            if not query.strip() or not email.strip():
                st.error("Please enter a PubMed search query and your email.")
                return
        else:
            if not uploaded_file:
                st.error("Please upload a file with search results.")
                return

        with st.spinner("Retrieving and screening articles..."):
            try:
                if input_method == "PubMed Search":
                    pubmed_ids = asyncio.run(fetch_pubmed_ids(query, email, api_key))
                    if not pubmed_ids:
                        st.warning("No articles found for the given query.")
                        return
                    xml_data = asyncio.run(fetch_pubmed_details(pubmed_ids, email, api_key))
                    if not xml_data:
                        st.error("Failed to retrieve article details.")
                        return
                    articles = parse_pubmed_xml(xml_data)
                else:
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    file_content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
                    try:
                        if file_extension == 'txt':
                            articles = parse_pubmed_format(file_content)
                        elif file_extension in ['nbib', 'ris']:
                            articles = parse_ris_format(file_content)
                        elif file_extension == 'bib':
                            articles = parse_bibtex_format(file_content)
                        else:
                            st.error("Unsupported file format.")
                            return
                    except Exception as e:
                        st.error(f"An error occurred while parsing the file: {str(e)}")
                        return

                if not articles:
                    st.warning("No articles found or parsed from the input.")
                    return

                st.subheader("All Retrieved Articles")
                st.dataframe(pd.DataFrame(articles))

                screened_articles, excluded_articles, exclusion_reasons = screen_articles(
                    articles, user_pico, study_topic, exclusion_regex, classifier, relevance_threshold, pico_weight=0.5
                )

                st.subheader("Screening Results")
                st.write(f"Total articles retrieved: **{len(articles)}**")
                st.write(f"Articles passing screening: **{len(screened_articles)}**")
                st.write(f"Articles excluded: **{len(excluded_articles)}**")

                if excluded_articles:
                    st.subheader("Excluded Articles")
                    excluded_df = pd.DataFrame(excluded_articles)
                    st.dataframe(excluded_df)
                    visualize_exclusion_reasons(exclusion_reasons)

                if screened_articles:
                    st.subheader("Screened Articles")
                    screened_df = pd.DataFrame(screened_articles)
                    st.dataframe(screened_df)

                    visualize_relevance_distribution(screened_articles)

                    # Save results
                    csv = screened_df.to_csv(index=False)
                    st.download_button(
                        label="Download Screened Articles (CSV)",
                        data=csv,
                        file_name="screened_articles.csv",
                        mime="text/csv",
                    )

                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                        screened_df.to_excel(
                            writer, index=False, sheet_name="Screened Articles"
                        )
                    excel_buffer.seek(0)
                    st.download_button(
                        label="Download Screened Articles (Excel)",
                        data=excel_buffer,
                        file_name="screened_articles.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                    # Visualize most common keywords
                    st.subheader("Most Common Keywords in Screened Articles")
                    all_text = " ".join(
                        [f"{article.get('title', '')} {article.get('abstract', '')}" for article in screened_articles]
                    )
                    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
                        all_text
                    )
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation="bilinear")
                    plt.axis("off")
                    st.pyplot(plt)

                    # Detailed review of screened articles
                    st.subheader("Detailed Review of Screened Articles")
                    article_titles = [
                        f"{article.get('study_id', '')}: {article.get('title', '')}" for article in screened_articles
                    ]
                    selected_article_title = st.selectbox(
                        "Select an article to review:", article_titles
                    )

                    if selected_article_title:
                        article = next(
                            article
                            for article in screened_articles
                            if f"{article.get('study_id', '')}: {article.get('title', '')}"
                            == selected_article_title
                        )
                        st.write(f"**Title:** {article.get('title', '')}")
                        st.write(f"**Authors:** {', '.join(article.get('authors', []))}")
                        st.write(f"**Publication Year:** {article.get('year', '')}")
                        st.write(f"**Abstract:** {article.get('abstract', '')}")
                        st.write(f"**Relevance Score:** {article.get('relevance_score', 0.0):.2f}")
                        st.write(f"**Publication Type:** {article.get('pub_type', '')}")

                        # PICO relevance breakdown
                        st.write("**PICO Relevance Breakdown:**")
                        for element, score in article.get("pico_scores", {}).items():
                            if score is not None:
                                st.write(f"- {element}: {score:.2f}")
                            else:
                                st.write(f"- {element}: Not provided")

                        # Topic relevance
                        st.write(f"**Topic Relevance:** {article.get('topic_relevance', 0.0):.2f}")
                else:
                    st.warning("No articles passed the screening. Consider adjusting the relevance threshold or refining your search query and PICO elements.")

            except Exception as e:
                logger.error(f"Error during screening process: {e}")
                st.error(
                    f"An error occurred during the screening process: {str(e)}. Please check the logs for more details."
                )

if __name__ == "__main__":
    main()
