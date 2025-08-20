import os
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

import streamlit as st
from openai import OpenAI
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, ClusterTimeoutOptions
from couchbase.auth import PasswordAuthenticator
from couchbase.vector_search import VectorQuery, VectorSearch
from couchbase.search import SearchRequest, MatchNoneQuery
from datetime import timedelta

from cb_connection import CouchbaseConnection


load_dotenv()


def extract_pdf_text_from_bytes(data: bytes) -> str:
    with open("/tmp/_uploaded_jd.pdf", "wb") as f:
        f.write(data)
    reader = PdfReader("/tmp/_uploaded_jd.pdf")
    texts: List[str] = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(texts)


def main() -> None:
    st.set_page_config(page_title="Agentic HR Sourcing", layout="wide")
    st.title("Agentic HR Sourcing - Candidate Match")
    st.caption("Upload a Job Description PDF. We will find the best matching candidates using enhanced hybrid search.")

    # Connect once and cache
    if "cb_conn" not in st.session_state:
        try:
            st.session_state.cb_conn = CouchbaseConnection()
        except Exception as e:
            st.error(f"Failed to connect to Couchbase: {e}")
            return
    cb = st.session_state.cb_conn

    uploaded = st.file_uploader("Upload Job Description (PDF)", type=["pdf"], accept_multiple_files=False)
    top_k = st.number_input("Top K candidates", min_value=1, max_value=25, value=5)
    
    # Add search method selection
    search_method = st.selectbox(
        "Search Method",
        ["Hybrid Search (Recommended)", "Vector Search Only", "Skills-Based Search"],
        help="Hybrid search combines vector similarity with skills matching and experience scoring for better accuracy."
    )

    if st.button("Find Candidates", type="primary"):
        if not uploaded:
            st.warning("Please upload a JD PDF first")
            return

        with st.spinner("Extracting text and analyzing job requirements..."):
            jd_text = extract_pdf_text_from_bytes(uploaded.read())
            if not jd_text.strip():
                st.error("Could not extract text from the uploaded PDF.")
                return
            
            # Extract key information from job description
            required_skills = cb._extract_skills_from_text(jd_text)
            required_years = cb._extract_years_experience(jd_text)
            
            st.info(f"📋 **Job Requirements Detected:**\n"
                   f"**Skills:** {', '.join(required_skills) if required_skills else 'Not detected'}\n"
                   f"**Experience:** {required_years} years" if required_years else "**Experience:** Not specified")

        with st.spinner("Searching for matching candidates..."):
            if search_method == "Hybrid Search (Recommended)":
                results: List[Dict] = cb.get_candidates_by_hybrid_search(jd_text, num_results=int(top_k))
            elif search_method == "Vector Search Only":
                query_embedding = cb.generate_enhanced_embedding(jd_text, is_job_description=True)
                results: List[Dict] = cb.get_candidates_by_vector(query_embedding, num_results=int(top_k))
            else:  # Skills-Based Search
                if required_skills:
                    results: List[Dict] = cb.search_candidates_by_skills(required_skills, num_results=int(top_k))
                else:
                    st.error("No skills detected in job description for skills-based search.")
                    return

        if not results:
            st.info("No candidates found. Make sure the `candidates` documents with embeddings exist.")
            return

        st.subheader("Top Matches")
        
        # Display search method info
        if search_method == "Hybrid Search (Recommended)":
            st.success("🎯 **Using Hybrid Search** - Results ranked by combined score (Vector + Skills + Experience)")
        elif search_method == "Vector Search Only":
            st.info("🔍 **Using Vector Search** - Results ranked by semantic similarity only")
        else:
            st.info("🎯 **Using Skills-Based Search** - Results ranked by skills match score")
        
        for i, cand in enumerate(results, 1):
            # Create expandable section for each candidate
            with st.expander(f"#{i} {cand.get('name', 'Unknown')} - Score: {cand.get('_combined_score', cand.get('_score', 0)):.3f}", expanded=i <= 3):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Name:** {cand.get('name', 'Unknown')}")
                    st.markdown(f"**Email:** {cand.get('email', 'N/A')}")
                    st.markdown(f"**Phone:** {cand.get('phone', 'N/A')}")
                    st.markdown(f"**Location:** {cand.get('location', 'N/A')}")
                    
                    # Display skills with highlighting
                    skills = cand.get("skills") or []
                    if isinstance(skills, list):
                        st.markdown("**Skills:**")
                        # Highlight matching skills
                        if required_skills and search_method == "Hybrid Search (Recommended)":
                            for skill in skills:
                                if skill.lower() in [req.lower() for req in required_skills]:
                                    st.markdown(f"✅ **{skill}**")
                                else:
                                    st.markdown(f"• {skill}")
                        else:
                            st.markdown(", ".join(skills))
                    
                    st.markdown(f"**Summary:** {cand.get('summary', 'N/A')}")
                
                with col2:
                    # Display scoring breakdown
                    if search_method == "Hybrid Search (Recommended)" and "_combined_score" in cand:
                        st.markdown("**Scoring Breakdown:**")
                        st.metric("Combined Score", f"{cand.get('_combined_score', 0):.3f}")
                        st.metric("Vector Score", f"{cand.get('_score', 0):.3f}")
                        st.metric("Skills Match", f"{cand.get('_skills_score', 0):.3f}")
                        st.metric("Experience Match", f"{cand.get('_experience_score', 0):.3f}")
                    elif search_method == "Skills-Based Search" and "_skills_score" in cand:
                        st.markdown("**Skills Match Score:**")
                        st.metric("Skills Score", f"{cand.get('_skills_score', 0):.3f}")
                    else:
                        st.markdown("**Vector Similarity Score:**")
                        st.metric("Score", f"{cand.get('_score', 0):.3f}")
                
                # Experience and Education in expandable section
                with st.expander("📚 Experience & Education Details"):
                    st.markdown("**Experience:**")
                    st.write(cand.get("experience") or "Not specified")
                    st.markdown("**Education:**")
                    st.write(cand.get("education") or "Not specified")
                
                st.divider()


if __name__ == "__main__":
    main()

