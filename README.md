# Comprehensive Systematic Review & Meta-Analysis Screening Tool

## Overview

The **Comprehensive Systematic Review & Meta-Analysis Screening Tool** is a Streamlit-based web application designed to assist researchers in screening and analyzing articles for systematic reviews and meta-analyses. The tool leverages Natural Language Processing (NLP) to evaluate the relevance of articles based on user-defined criteria and PICO elements (Population, Intervention, Comparison, Outcome).

## Features

- **Define Research Topic and PICO Elements:** Specify the research topic and detail the Population, Intervention, Comparison, and Outcome (PICO) elements to guide the screening process.

- **Article Input Methods:**
  - **PubMed Search:** Perform a direct search on PubMed using a search query.
  - **Upload Search Results:** Upload search results in various formats (`.txt`, `.nbib`, `.ris`, `.bib`).

- **Customize Exclusion Terms:** Modify exclusion terms to filter out non-eligible studies effectively.

- **Relevance Scoring:** Evaluate the relevance of articles based on PICO elements and overall study topic.

- **Comprehensive Screening Results:** View and download screened and excluded articles, with detailed feedback and visualizations.

- **Visualizations:**
  - Relevance score distributions.
  - Exclusion reasons breakdown.
  - Word clouds of common keywords in screened articles.
  - Radar charts of PICO elements' lengths.

- **Detailed Article Review:** Select individual articles to review detailed information, including relevance scores and PICO relevance breakdowns.

## Installation

### Prerequisites

- **Python 3.7 or higher**: Ensure Python is installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

- **pip**: Python's package installer should be up-to-date. Upgrade pip using:
  ```bash
  pip install --upgrade pip
