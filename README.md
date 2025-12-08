# CareerConnect  
### CareerConnect: A RAG Assistant for Job and Course Recommendations  

## Overview  
CareerConnect is a RAG system that analyzes resumes, identifies strengths, retrieves relevant jobs, and recommends targeted upskilling courses.

Users simply upload a resume, and the system automatically:

- Extracts skills  
- Matches relevant jobs  
- Recommends personalized Udemy courses  
- Generates an LLM-powered career summary  

This project demonstrates a complete **RAG (Retrieval-Augmented Generation)** pipeline running on HPC GPUs.

## Project Objectives  
- Build an automated resume analysis system  
- Perform semantic job matching using embeddings + FAISS  
- Recommend targeted courses for missing skills  
- Deploy an interactive Streamlit UI  
- Run LLM inference (Mistral-7B) on GMU Hopper cluster  
- Provide student-friendly documentation  

## System Architecture  

- **Backend API** → FastAPI  
- **Embedding Model** → BGE-small  
- **Vector Store** → FAISS  
- **LLM Layer** → Mistral-7B via HuggingFace transformers  
- **UI** → Streamlit  
- **Data Sources** → LinkedIn Jobs, Monster Jobs, Udemy Courses  

## Features  

### Resume Skill Extraction  
NLP and taxonomy-based extraction (ESCO/O*NET).

### Semantic Job Retrieval  
Top-k job matches using FAISS index + cosine similarity.

### Personalized Upskilling  
Recommends Udemy courses linked to missing technical skills.

### LLM Summary Generation  
Mistral-7B generates a strengths-focused multi-section profile summary.

## Methodology  

1. Parse resume → Extract text  
2. Identify skills from curated skill list  
3. Embed skills + job descriptions  
4. Semantic retrieval using FAISS  
5. Identify missing skills  
6. Recommend best matching Udemy courses  
7. Generate detailed career summary using an LLM  

This forms the complete **RAG system** for job & skill recommendations.

## How to Run the Project  

### **1. Start Backend API**
```bash
uvicorn src.api_server_http:app --host 0.0.0.0 --port 8010
```
### **2. Launch Streamlit UI**
```bash
streamlit run src/ui/ui_app.py --server.port 8501
```
