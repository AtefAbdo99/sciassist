import streamlit as st
import numpy as np
import pandas as pd
import aiohttp
import asyncio
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer, util
import logging
from io import BytesIO, StringIO
import json
from bs4 import BeautifulSoup
import re
from rispy import load as load_ris
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import plotly.express as px
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(
    filename='sciassist_screening.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)

# Set Streamlit page configuration
st.set_page_config(
    page_title="Advanced SR & MA Screening Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    body {
        background-color: #f0f4f8;
        color: #1e1e1e;
        font-family: 'Roboto', sans-serif;
    }
    .stButton>button {
        background-color: #4a90e2;
        color: white;
        border: none;
        padding: 0.6em 1.2em;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #357abd;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .stTextInput>div>div>input {
        border: 1px solid #bdc3c7;
        padding: 0.6em;
        border-radius: 8px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stTextInput>div>div>input:focus {
        border-color: #4a90e2;
        box-shadow: 0 0 0 2px rgba(74,144,226,0.2);
    }
    .stSelectbox>div>div>div {
        background-color: #ffffff;
    }
    .stHeader {
        background-color: #4a90e2;
        padding: 1em;
        border-radius: 8px;
        color: white;
        font-weight: 700;
        margin-bottom: 1em;
    }
    .stSubheader {
        color: #4a90e2;
        font-weight: 700;
        border-bottom: 2px solid #4a90e2;
        padding-bottom: 0.3em;
        margin-top: 1em;
    }
    .info-box {
        background-color: #e7f3fe;
        border-left: 6px solid #2196F3;
        margin-bottom: 15px;
        padding: 1em;
        border-radius: 0 8px 8px 0;
    }
    .results-container {
        background-color: white;
        padding: 1em;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-top: 1em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model = None
bert_tokenizer = None
embedding_model = None

@st.cache_resource
def load_models():
    global bert_model, bert_tokenizer, embedding_model
    logger.info("Loading BERT model and embedding model...")
    try:
        bert_model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        bert_model = BertModel.from_pretrained(bert_model_name).to(device)
        logger.info("BERT model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading BERT model: {e}")
        st.error("An error occurred while loading the BERT model. Please try again later.")
        return None, None, None
    
    try:
        embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        embedding_model = SentenceTransformer(embedding_model_name)
        logger.info("Embedding model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        st.error("An error occurred while loading the embedding model. Please try again later.")
        return None, None, None
    
    return bert_model, bert_tokenizer, embedding_model

def extract_pico_elements(text):
    global bert_model, bert_tokenizer
    
    if bert_model is None or bert_tokenizer is None:
        st.error("BERT model or tokenizer not initialized. Please reload the page.")
        return {}

    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    
    pico_elements = ['Population', 'Intervention', 'Comparison', 'Outcome']
    pico_dict = {}
    
    for element in pico_elements:
        element_inputs = bert_tokenizer(element, return_tensors="pt").to(device)
        with torch.no_grad():
            element_outputs = bert_model(**element_inputs)
        element_embedding = element_outputs.last_hidden_state.mean(dim=1)
        similarity = torch.nn.functional.cosine_similarity(embeddings, element_embedding)
        pico_dict[element] = similarity.item()
    
    return pico_dict

def calculate_relevance_score(study_text, user_pico):
    study_pico = extract_pico_elements(study_text)
    return sum(study_pico[element] * user_pico[element] for element in user_pico)

def is_valid_study_design(pub_types):
    excluded_types = {'review', 'meta-analysis', 'systematic review', 'comment', 'editorial', 'letter', 'abstract'}
    return not any(pt.lower() in excluded_types for pt in pub_types.split(', '))

async def fetch(session, url, params):
    try:
        async with session.get(url, params=params, timeout=30) as response:
            return await response.text()
    except asyncio.TimeoutError:
        logger.error(f"Timeout error while fetching data from {url}")
        return None
    except Exception as e:
        logger.error(f"Error fetching data from {url}: {e}")
        return None

async def fetch_pubmed_ids(query, email, api_key=None):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": 10000,
        "usehistory": "y",
        "retmode": "json",
        "sort": "relevance",
        "email": email,
        "tool": "AdvancedSRScreeningTool"
    }
    if api_key:
        params["api_key"] = api_key

    async with aiohttp.ClientSession() as session:
        response_text = await fetch(session, base_url, params)
        if response_text:
            try:
                response = json.loads(response_text)
                return response['esearchresult']['idlist']
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON response from PubMed")
                return []
        return []

async def fetch_pubmed_details(ids, email, api_key=None):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "xml",
        "email": email,
        "tool": "AdvancedSRScreeningTool"
    }
    if api_key:
        params["api_key"] = api_key

    async with aiohttp.ClientSession() as session:
        response_text = await fetch(session, base_url, params)
        return response_text

def parse_pubmed_xml(xml_string):
    soup = BeautifulSoup(xml_string, 'xml')
    articles = []
    for article in soup.find_all('PubmedArticle'):
        pmid = article.find('PMID').text if article.find('PMID') else ''
        title = article.find('ArticleTitle').text if article.find('ArticleTitle') else ''
        abstract = ' '.join([abstract_text.text for abstract_text in article.find_all('AbstractText')]) if article.find('AbstractText') else ''
        pub_type = [pt.text for pt in article.find_all('PublicationType')]
        articles.append({
            'pmid': pmid,
            'title': title,
            'abstract': abstract,
            'pub_type': ', '.join(pub_type)
        })
    return articles

def parse_ris_file(file_content):
    entries = load_ris(StringIO(file_content))
    articles = []
    for entry in entries:
        articles.append({
            'pmid': entry.get('accession_number', ''),
            'title': entry.get('title', ''),
            'abstract': entry.get('abstract', ''),
            'pub_type': entry.get('type_of_reference', '')
        })
    return articles

def parse_pubmed_format(file_content):
    entries = file_content.split('\n\n')
    articles = []
    for entry in entries:
        article = {}
        for line in entry.split('\n'):
            if line.startswith('PMID-'):
                article['pmid'] = line.split('-')[1].strip()
            elif line.startswith('TI  -'):
                article['title'] = line[6:].strip()
            elif line.startswith('AB  -'):
                article['abstract'] = line[6:].strip()
            elif line.startswith('PT  -'):
                article['pub_type'] = line[6:].strip()
        if article:
            articles.append(article)
    return articles

def screen_articles(articles, user_pico, relevance_threshold=0.5):
    screened_articles = []
    for article in articles:
        article = {k: str(v) if v is not None else '' for k, v in article.items()}
        study_text = f"{article.get('title', '')} {article.get('abstract', '')}"
        
        if is_valid_study_design(article.get('pub_type', '')):
            relevance_score = calculate_relevance_score(study_text, user_pico)
            if relevance_score >= relevance_threshold:
                article['relevance_score'] = relevance_score
                screened_articles.append(article)
    
    return sorted(screened_articles, key=lambda x: x['relevance_score'], reverse=True)

def cluster_similar_articles(screened_articles):
    texts = [f"{article['title']} {article['abstract']}" for article in screened_articles]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    clusters = {}
    for i, article in enumerate(screened_articles):
        similar_indices = cosine_similarities[i].argsort()[:-5:-1]  # Get top 4 similar articles
        clusters[article['pmid']] = [screened_articles[j]['pmid'] for j in similar_indices if i != j]
    
    return clusters

def visualize_relevance_distribution(screened_articles):
    relevance_scores = [article['relevance_score'] for article in screened_articles]
    fig = px.histogram(x=relevance_scores, nbins=20, title="Distribution of Relevance Scores")
    fig.update_layout(
        xaxis_title="Relevance Score",
        yaxis_title="Number of Articles",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#1e1e1e")
    )
    fig.update_traces(marker_color='#4a90e2', marker_line_color='#357abd', marker_line_width=1.5, opacity=0.8)
    st.plotly_chart(fig, use_container_width=True)

def visualize_pico_radar(user_pico):
    categories = list(user_pico.keys())
    values = list(user_pico.values())
    
    fig = go.Figure(data=go.Scatterpolar(
      r=values,
      theta=categories,
      fill='toself',
      line_color='#4a90e2'
    ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 1]
        )),
      showlegend=False,
      title="PICO Elements Relevance",
      paper_bgcolor='rgba(0,0,0,0)',
      plot_bgcolor='rgba(0,0,0,0)',
      font=dict(color="#1e1e1e")
    )
    st.plotly_chart(fig, use_container_width=True)

def main():
    global bert_model, bert_tokenizer, embedding_model
    
    st.markdown('<h1 class="stHeader">Advanced SR & MA Screening Tool</h1>', unsafe_allow_html=True)

    # Load models
    bert_model, bert_tokenizer, embedding_model = load_models()
    
    if bert_model is None or bert_tokenizer is None or embedding_model is None:
        st.error("Failed to load necessary models. Please try reloading the page.")
        return

    st.markdown('<h2 class="stSubheader">Step 1: Define Research Topic or PICO</h2>', unsafe_allow_html=True)
    
    input_type = st.radio("Choose input type:", ("Research Topic", "PICO Elements"))
    
    user_pico = {}
    pico_generated = False

    if input_type == "Research Topic":
        research_topic = st.text_area("Enter your research topic:", help="Type in your research question or topic of interest.")
        if st.button("Generate PICO"):
            if research_topic:
                with st.spinner("Generating PICO..."):
                    user_pico = extract_pico_elements(research_topic)
                    st.success("PICO generated successfully!")
                    visualize_pico_radar(user_pico)
                    st.write(user_pico)
                    pico_generated = True
            else:
                st.warning("Please enter a research topic before generating PICO.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            user_pico['Population'] = st.text_input("Population:", help="Target population for the study")
            user_pico['Intervention'] = st.text_input("Intervention:", help="Treatment or intervention being studied")
        with col2:
            user_pico['Comparison'] = st.text_input("Comparison:", help="Alternative to the intervention (if applicable)")
            user_pico['Outcome'] = st.text_input("Outcome:", help="Outcomes of interest")
        
        if all(user_pico.values()):
            user_pico = {k: float(v) if v.replace('.', '').isdigit() else 0.5 for k, v in user_pico.items()}
            visualize_pico_radar(user_pico)
            pico_generated = True
        elif any(user_pico.values()):
            st.warning("Please fill in all PICO fields or leave them all empty to skip this step.")

    if not pico_generated:
        st.warning("PICO elements have not been generated or filled in. You can proceed, but screening may be less accurate.")

    st.markdown('<h2 class="stSubheader">Step 2: Article Input</h2>', unsafe_allow_html=True)
    
    input_method = st.radio("Choose input method:", 
                            ("PubMed Search", "Upload Search Results"),
                            help="Select 'PubMed Search' to fetch articles directly from PubMed. Select 'Upload Search Results' to upload your own list of articles.")

    if input_method == "PubMed Search":
        query = st.text_area("Enter your PubMed search query:", help="Type in your PubMed search query to retrieve articles.")
        email = st.text_input("Enter your email (required for PubMed API):", help="Your email is required by the PubMed API for identification purposes.")
        api_key = st.text_input("Enter your NCBI API key (optional):", type="password", help="Optional NCBI API key for increased request limits.")
    else:
        uploaded_file = st.file_uploader("Upload your search results (CSV, Excel, RIS, or PubMed format)", 
                                         type=["csv", "xlsx", "ris", "txt"],
                                         help="Upload a file containing your search results in one of the supported formats.")

    relevance_threshold = st.slider("Set relevance threshold for screening:", 0.0, 1.0, 0.5, 0.01)

    if st.button("Run Intelligent Automated Screening", key="run_screening"):
        if not pico_generated:
            proceed = st.warning("PICO elements are not set. Do you want to proceed with default values?")
            if not proceed:
                st.stop()
            user_pico = {k: 0.5 for k in ['Population', 'Intervention', 'Comparison', 'Outcome']}

        if input_method == "PubMed Search":
            if not query:
                st.error("Please enter a PubMed search query.")
                return
            if not email:
                st.error("Please enter your email address for the PubMed API.")
                return
        else:  # Upload Search Results
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
                    file_content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
                    
                    if file_extension == 'csv':
                        articles = pd.read_csv(StringIO(file_content)).to_dict('records')
                    elif file_extension == 'xlsx':
                        articles = pd.read_excel(BytesIO(uploaded_file.getvalue())).to_dict('records')
                    elif file_extension == 'ris':
                        articles = parse_ris_file(file_content)
                    elif file_extension == 'txt':
                        articles = parse_pubmed_format(file_content)
                    else:
                        st.error("Unsupported file format.")
                        return

                if not articles:
                    st.warning("No articles found or parsed from the input.")
                    return

                # Proceed with screening and results display
                st.markdown('<div class="results-container">', unsafe_allow_html=True)
                st.subheader("All Retrieved Articles")
                st.write(pd.DataFrame(articles))
                st.markdown('</div>', unsafe_allow_html=True)

                screened_articles = screen_articles(articles, user_pico, relevance_threshold)
                
                st.markdown('<div class="results-container">', unsafe_allow_html=True)
                st.subheader("Screening Results")
                st.write(f"Total articles retrieved: {len(articles)}")
                st.write(f"Articles passing screening: {len(screened_articles)}")
                
                st.subheader("Screened Articles")
                screened_df = pd.DataFrame(screened_articles)
                st.dataframe(screened_df.style.background_gradient(subset=['relevance_score'], cmap='Blues'))
                st.markdown('</div>', unsafe_allow_html=True)

                visualize_relevance_distribution(screened_articles)

                st.markdown('<div class="results-container">', unsafe_allow_html=True)
                st.subheader("Similar Article Clusters")
                clusters = cluster_similar_articles(screened_articles)
                for pmid, similar_pmids in clusters.items():
                    st.write(f"Articles similar to PMID {pmid}: {', '.join(similar_pmids)}")
                st.markdown('</div>', unsafe_allow_html=True)

                # Save results
                csv = screened_df.to_csv(index=False)
                st.download_button(
                    label="Download Screened Articles (CSV)",
                    data=csv,
                    file_name="screened_articles.csv",
                    mime="text/csv",
                )
                
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    screened_df.to_excel(writer, index=False, sheet_name='Screened Articles')
                excel_buffer.seek(0)
                st.download_button(
                    label="Download Screened Articles (Excel)",
                    data=excel_buffer,
                    file_name="screened_articles.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            except Exception as e:
                logger.error(f"Error during screening process: {e}")
                st.error(f"An error occurred during the screening process: {str(e)}. Please try again.")

    st.markdown("""
    <div class="info-box">
    <h4>How to use this Advanced SR & MA Screening Tool:</h4>
    <ol>
        <li><strong>Define Research Topic or PICO:</strong> Enter either your research topic for automatic PICO generation or input PICO elements manually.</li>
        <li><strong>Article Input:</strong> Choose to search PubMed directly or upload your own search results in various formats.</li>
        <li><strong>Set Relevance Threshold:</strong> Adjust the threshold for article inclusion based on relevance scores.</li>
        <li><strong>Run Screening:</strong> The tool will automatically retrieve articles, show all results, perform intelligent screening, and present the screened articles.</li>
        <li><strong>Review Results:</strong> Examine the screened articles, relevance score distribution, and similar article clusters.</li>
        <li><strong>Download Results:</strong> Save the screened articles in CSV or Excel format for further analysis.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()