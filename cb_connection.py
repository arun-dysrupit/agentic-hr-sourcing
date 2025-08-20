import os
import re
from datetime import timedelta
from typing import Any, Dict, List, Optional
import json

from dotenv import load_dotenv
from openai import OpenAI
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, ClusterTimeoutOptions
from couchbase.auth import PasswordAuthenticator
from couchbase.vector_search import VectorQuery, VectorSearch
from couchbase.search import SearchRequest, MatchNoneQuery, MatchQuery, ConjunctionQuery, DisjunctionQuery, MatchAllQuery


load_dotenv()


class CouchbaseConnection:
    """Enhanced Couchbase connection with LLM-based analysis for better search accuracy."""

    def __init__(self) -> None:
        connection_string = os.getenv("CB_CONNECTION_STRING")
        username = os.getenv("CB_USERNAME")
        password = os.getenv("CB_PASSWORD")
        bucket_name = os.getenv("CB_BUCKET")
        collection_name = os.getenv("CB_COLLECTION")

        if not all([connection_string, username, password, bucket_name, collection_name]):
            raise ValueError("Missing required Couchbase environment variables")

        auth = PasswordAuthenticator(username, password)
        timeout_options = ClusterTimeoutOptions(
            kv_timeout=timedelta(seconds=10),
            query_timeout=timedelta(seconds=20),
            search_timeout=timedelta(seconds=20),
        )
        options = ClusterOptions(auth, timeout_options=timeout_options)

        self.cluster = Cluster(connection_string, options)
        self.cluster.ping()

        self.bucket = self.cluster.bucket(bucket_name)
        self.scope = self.bucket.scope("_default")
        # Use _default collection where candidates are stored
        self.collection = self.bucket.collection("_default")
        # Use the correct search index name based on the debug output
        self.search_index_name = os.getenv("CB_SEARCH_INDEX", "vector_search_recruiter")

        # LLM client for analysis
        self._llm_client = OpenAI(
            base_url=os.getenv("NEBIUS_API_BASE"),
            api_key=os.getenv("NEBIUS_API_KEY"),
        )

        # Embeddings client
        self._embedding_client = OpenAI(
            base_url=os.getenv("NEBIUS_API_BASE"),
            api_key=os.getenv("NEBIUS_API_KEY"),
        )

    def analyze_job_description(self, job_description: str) -> Dict[str, Any]:
        """Use LLM to analyze job description and extract structured requirements."""
        prompt = f"""
        Analyze this job description and extract structured information. Return ONLY valid JSON with the following structure:

        {{
            "required_skills": ["skill1", "skill2", "skill3"],
            "preferred_skills": ["skill1", "skill2"],
            "years_experience": 3,
            "experience_level": "entry|mid|senior|lead",
            "job_title": "string",
            "key_requirements": ["requirement1", "requirement2"],
            "technical_requirements": ["tech1", "tech2"],
            "soft_skills": ["skill1", "skill2"],
            "industry": "string",
            "location_requirements": "string"
        }}

        Job Description:
        {job_description}

        Extract the most important and specific requirements. Be precise about years of experience and required skills.
        """

        try:
            response = self._llm_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean the response to extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            # Find JSON content
            start_brace = content.find("{")
            end_brace = content.rfind("}")
            if start_brace != -1 and end_brace != -1:
                content = content[start_brace:end_brace + 1]
            
            analysis = json.loads(content)
            
            # Ensure required fields exist
            analysis.setdefault("required_skills", [])
            analysis.setdefault("preferred_skills", [])
            analysis.setdefault("years_experience", None)
            analysis.setdefault("experience_level", "mid")
            analysis.setdefault("job_title", "")
            analysis.setdefault("key_requirements", [])
            analysis.setdefault("technical_requirements", [])
            analysis.setdefault("soft_skills", [])
            analysis.setdefault("industry", "")
            analysis.setdefault("location_requirements", "")
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing job description: {e}")
            # Fallback to basic extraction
            return self._fallback_jd_analysis(job_description)

    def _fallback_jd_analysis(self, job_description: str) -> Dict[str, Any]:
        """Fallback analysis when LLM fails."""
        # Basic pattern matching as fallback
        skills_pattern = r'\b(?:React|Angular|Vue|Node\.js|Python|Java|JavaScript|TypeScript|AWS|Azure|Docker|Kubernetes|MongoDB|PostgreSQL|MySQL|Redis|GraphQL|REST|API|Git|CI/CD|DevOps|Agile|Scrum|HTML|CSS|SASS|LESS|Bootstrap|Tailwind|jQuery|Express\.js|Django|Flask|Spring|Hibernate|JPA|Maven|Gradle|npm|yarn|Webpack|Babel|ESLint|Prettier)\b'
        experience_pattern = r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)'
        
        skills = re.findall(skills_pattern, job_description, re.IGNORECASE)
        experience_match = re.search(experience_pattern, job_description, re.IGNORECASE)
        years_experience = int(experience_match.group(1)) if experience_match else None
        
        return {
            "required_skills": list(set(skills)),
            "preferred_skills": [],
            "years_experience": years_experience,
            "experience_level": "mid" if years_experience and years_experience >= 3 else "entry",
            "job_title": "",
            "key_requirements": [],
            "technical_requirements": list(set(skills)),
            "soft_skills": [],
            "industry": "",
            "location_requirements": ""
        }

    def analyze_resume(self, resume_text: str) -> Dict[str, Any]:
        """Use LLM to analyze resume and extract structured candidate information."""
        prompt = f"""
        Analyze this resume and extract structured information. Return ONLY valid JSON with the following structure:

        {{
            "name": "string",
            "email": "string",
            "phone": "string",
            "location": "string",
            "years_experience": 5,
            "total_experience_years": 5,
            "skills": ["skill1", "skill2", "skill3"],
            "technical_skills": ["tech1", "tech2"],
            "soft_skills": ["soft1", "soft2"],
            "experience": "string",
            "education": "string",
            "summary": "string",
            "work_history": [
                {{
                    "company": "string",
                    "title": "string",
                    "duration": "string",
                    "years": 2,
                    "description": "string",
                    "technologies": ["tech1", "tech2"]
                }}
            ],
            "certifications": ["cert1", "cert2"],
            "languages": ["lang1", "lang2"],
            "github": "string",
            "linkedin": "string"
        }}

        Resume Text:
        {resume_text}

        Extract accurate information. Be precise about years of experience and skills.
        """

        try:
            response = self._llm_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean the response to extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            # Find JSON content
            start_brace = content.find("{")
            end_brace = content.rfind("}")
            if start_brace != -1 and end_brace != -1:
                content = content[start_brace:end_brace + 1]
            
            analysis = json.loads(content)
            
            # Ensure required fields exist
            analysis.setdefault("name", "Unknown")
            analysis.setdefault("email", "")
            analysis.setdefault("phone", "")
            analysis.setdefault("location", "")
            analysis.setdefault("years_experience", 0)
            analysis.setdefault("total_experience_years", 0)
            analysis.setdefault("skills", [])
            analysis.setdefault("technical_skills", [])
            analysis.setdefault("soft_skills", [])
            analysis.setdefault("experience", "")
            analysis.setdefault("education", "")
            analysis.setdefault("summary", "")
            analysis.setdefault("work_history", [])
            analysis.setdefault("certifications", [])
            analysis.setdefault("languages", [])
            analysis.setdefault("github", "")
            analysis.setdefault("linkedin", "")
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing resume: {e}")
            # Fallback to basic extraction
            return self._fallback_resume_analysis(resume_text)

    def _fallback_resume_analysis(self, resume_text: str) -> Dict[str, Any]:
        """Fallback analysis when LLM fails."""
        # Basic extraction as fallback
        lines = resume_text.split('\n')
        name = lines[0].strip() if lines else "Unknown"
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, resume_text)
        email = email_match.group(0) if email_match else ""
        
        # Extract phone
        phone_pattern = r'[\+]?[1-9][\d]{0,15}'
        phone_match = re.search(phone_pattern, resume_text)
        phone = phone_match.group(0) if phone_match else ""
        
        return {
            "name": name,
            "email": email,
            "phone": phone,
            "location": "",
            "years_experience": 0,
            "total_experience_years": 0,
            "skills": [],
            "technical_skills": [],
            "soft_skills": [],
            "experience": "",
            "education": "",
            "summary": "",
            "work_history": [],
            "certifications": [],
            "languages": [],
            "github": "",
            "linkedin": ""
        }

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings for text."""
        try:
            resp = self._embedding_client.embeddings.create(
                model="intfloat/e5-mistral-7b-instruct",
                input=text,
                timeout=30,
            )
            return resp.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return [0.0] * 1024  # Return zero vector as fallback

    def upsert_candidate(self, doc_id: str, candidate_doc: Dict[str, Any]) -> None:
        """Store candidate document with enhanced data model."""
        self.collection.upsert(doc_id, candidate_doc)

    def get_candidates_by_llm_analysis(self, job_description: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Get candidates using LLM-based job description analysis for maximum accuracy."""
        print("🔍 Analyzing job description with LLM...")
        
        # Analyze job description
        jd_analysis = self.analyze_job_description(job_description)
        print(f"✅ Job analysis complete: {jd_analysis['required_skills']} skills, {jd_analysis['years_experience']} years exp")
        
        # Generate embedding for semantic search
        query_embedding = self.generate_embedding(job_description)
        
        # Get candidates via vector search first
        search_req = SearchRequest.create(MatchNoneQuery()).with_vector_search(
            VectorSearch.from_vector_query(
                VectorQuery("embedding", query_embedding, num_candidates=num_results * 3)
            )
        )
        
        try:
            result = self.scope.search(self.search_index_name, search_req, timeout=timedelta(seconds=20))
            rows = list(result.rows())
        except Exception as e:
            print(f"Vector search failed: {e}")
            return []

        # Process and score candidates
        candidates = []
        for row in rows:
            try:
                doc = self.collection.get(row.id, timeout=timedelta(seconds=5))
                if doc and doc.value:
                    data = doc.value
                    data["_id"] = row.id
                    data["_vector_score"] = row.score
                    
                    # Calculate comprehensive match score
                    match_score = self._calculate_comprehensive_match_score(data, jd_analysis)
                    data["_match_score"] = match_score
                    
                    candidates.append(data)
            except Exception as e:
                print(f"Error processing candidate {row.id}: {e}")
                continue

        # Sort by match score and return top results
        candidates.sort(key=lambda x: x["_match_score"], reverse=True)
        return candidates[:num_results]

    def _calculate_comprehensive_match_score(self, candidate: Dict[str, Any], jd_analysis: Dict[str, Any]) -> float:
        """Calculate comprehensive match score using multiple factors."""
        score = 0.0
        max_score = 0.0
        
        # 1. Skills matching (40% weight)
        required_skills = set(skill.lower() for skill in jd_analysis.get("required_skills", []))
        preferred_skills = set(skill.lower() for skill in jd_analysis.get("preferred_skills", []))
        
        candidate_skills = set()
        if candidate.get("skills"):
            candidate_skills.update(skill.lower() for skill in candidate["skills"])
        if candidate.get("technical_skills"):
            candidate_skills.update(skill.lower() for skill in candidate["technical_skills"])
        
        # Required skills match
        required_matches = len(required_skills & candidate_skills)
        required_score = (required_matches / len(required_skills)) if required_skills else 0.0
        
        # Preferred skills bonus
        preferred_matches = len(preferred_skills & candidate_skills)
        preferred_score = (preferred_matches / len(preferred_skills)) * 0.5 if preferred_skills else 0.0
        
        skills_score = min(1.0, required_score + preferred_score)
        score += skills_score * 0.4
        max_score += 0.4
        
        # 2. Experience matching (30% weight)
        required_years = jd_analysis.get("years_experience")
        candidate_years = candidate.get("years_experience", 0)
        
        if required_years and candidate_years:
            if candidate_years >= required_years:
                exp_score = 1.0  # Perfect or overqualified
            elif candidate_years >= required_years * 0.7:
                exp_score = 0.8  # Close match
            elif candidate_years >= required_years * 0.5:
                exp_score = 0.6  # Moderate match
            else:
                exp_score = 0.2  # Underqualified
        else:
            exp_score = 0.5  # Neutral if unknown
        
        score += exp_score * 0.3
        max_score += 0.3
        
        # 3. Vector similarity (20% weight)
        vector_score = candidate.get("_vector_score", 0.0)
        score += vector_score * 0.2
        max_score += 0.2
        
        # 4. Work history relevance (10% weight)
        work_history_score = 0.0
        if candidate.get("work_history"):
            relevant_jobs = 0
            for job in candidate["work_history"]:
                job_desc = job.get("description", "").lower()
                job_tech = job.get("technologies", [])
                
                # Check if job description mentions required skills
                for skill in required_skills:
                    if skill in job_desc or any(skill in tech.lower() for tech in job_tech):
                        relevant_jobs += 1
                        break
            
            work_history_score = relevant_jobs / len(candidate["work_history"])
        
        score += work_history_score * 0.1
        max_score += 0.1
        
        # Normalize score
        final_score = score / max_score if max_score > 0 else 0.0
        
        # Store individual scores for transparency
        candidate["_skills_score"] = skills_score
        candidate["_experience_score"] = exp_score
        candidate["_vector_score"] = vector_score
        candidate["_work_history_score"] = work_history_score
        
        return final_score

    def search_candidates_by_skills(self, required_skills: List[str], num_results: int = 5) -> List[Dict[str, Any]]:
        """Search candidates by specific skills using proper Couchbase search with prefilters."""
        if not required_skills:
            return []
        
        print(f"🔍 Searching candidates by skills: {required_skills}")
        
        # Create a skills filter using OR logic
        skill_queries = []
        for skill in required_skills:
            skill_queries.append(MatchQuery("skills", skill))
            skill_queries.append(MatchQuery("technical_skills", skill))
        
        # Use OR logic for skills matching
        skills_filter = DisjunctionQuery(*skill_queries)
        
        # Create search request with skills prefilter
        search_req = SearchRequest.create(
            VectorSearch.from_vector_query(
                VectorQuery(
                    field_name="embedding",
                    vector=[0.0] * 1024,  # Neutral vector for skills-only search
                    prefilter=skills_filter,
                    num_candidates=num_results * 2
                )
            )
        )
        
        try:
            result = self.scope.search(
                self.search_index_name, 
                search_req, 
                timeout=timedelta(seconds=20)
            )
            rows = list(result.rows())
        except Exception as e:
            print(f"Skills search failed: {e}")
            return []

        candidates = []
        for row in rows:
            try:
                doc = self.collection.get(row.id, timeout=timedelta(seconds=5))
                if doc and doc.value:
                    data = doc.value
                    data["_id"] = row.id
                    data["_score"] = row.score
                    
                    # Calculate skills match score
                    skills_score = self._calculate_skills_match_score(data, required_skills)
                    data["_skills_score"] = skills_score
                    
                    candidates.append(data)
            except Exception as e:
                print(f"Error processing candidate {row.id}: {e}")
                continue
        
        # Sort by skills score
        candidates.sort(key=lambda x: x.get("_skills_score", 0), reverse=True)
        return candidates[:num_results]

    def get_candidates_by_vector(self, query_embedding: List[float], num_results: int = 5) -> List[Dict[str, Any]]:
        """Get candidates using pure vector search with proper Couchbase patterns."""
        print("🔍 Performing vector search...")
        
        # Create search request
        search_req = SearchRequest.create(
            VectorSearch.from_vector_query(
                VectorQuery(
                    field_name="embedding",
                    vector=query_embedding,
                    num_candidates=num_results * 2
                )
            )
        )
        
        try:
            result = self.scope.search(
                self.search_index_name,
                search_req,
                timeout=timedelta(seconds=20)
            )
            rows = list(result.rows())
        except Exception as e:
            print(f"Vector search failed: {e}")
            return []

        candidates = []
        for row in rows:
            try:
                doc = self.collection.get(row.id, timeout=timedelta(seconds=5))
                if doc and doc.value:
                    data = doc.value
                    data["_score"] = row.score
                    data["_id"] = row.id
                    candidates.append(data)
            except Exception as e:
                print(f"Error processing candidate {row.id}: {e}")
                continue
        
        return candidates[:num_results]

    def _calculate_skills_match_score(self, candidate: Dict[str, Any], required_skills: List[str]) -> float:
        """Calculate skills match score for a candidate."""
        candidate_skills = set()
        if candidate.get("skills"):
            candidate_skills.update(skill.lower() for skill in candidate["skills"])
        if candidate.get("technical_skills"):
            candidate_skills.update(skill.lower() for skill in candidate["technical_skills"])
        
        required_skills_set = set(skill.lower() for skill in required_skills)
        
        if not required_skills_set:
            return 0.0
        
        # Calculate match percentage
        matches = len(required_skills_set & candidate_skills)
        match_percentage = matches / len(required_skills_set)
        
        # Bonus for having additional skills
        bonus = min(0.2, len(candidate_skills) * 0.01)
        
        return min(1.0, match_percentage + bonus)

    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from text using LLM analysis."""
        try:
            prompt = f"""
            Extract technical skills from this text. Return ONLY a JSON array of skills.
            
            Text: {text[:2000]}
            
            Return format: ["skill1", "skill2", "skill3"]
            """
            
            response = self._llm_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean response to extract JSON array
            if "[" in content and "]" in content:
                start_bracket = content.find("[")
                end_bracket = content.rfind("]")
                if start_bracket != -1 and end_bracket != -1:
                    json_array = content[start_bracket:end_bracket + 1]
                    try:
                        skills = json.loads(json_array)
                        if isinstance(skills, list):
                            return [str(skill).strip() for skill in skills if skill]
                    except json.JSONDecodeError:
                        pass
            
            # Fallback to regex extraction
            skills_pattern = r'\b(?:React|Angular|Vue|Node\.js|Python|Java|JavaScript|TypeScript|AWS|Azure|Docker|Kubernetes|MongoDB|PostgreSQL|MySQL|Redis|GraphQL|REST|API|Git|CI/CD|DevOps|Agile|Scrum|HTML|CSS|SASS|LESS|Bootstrap|Tailwind|jQuery|Express\.js|Django|Flask|Spring|Hibernate|JPA|Maven|Gradle|npm|yarn|Webpack|Babel|ESLint|Prettier)\b'
            skills = re.findall(skills_pattern, text, re.IGNORECASE)
            return list(set(skills))
            
        except Exception as e:
            print(f"Error extracting skills: {e}")
            # Fallback to regex
            skills_pattern = r'\b(?:React|Angular|Vue|Node\.js|Python|Java|JavaScript|TypeScript|AWS|Azure|Docker|Kubernetes|MongoDB|PostgreSQL|MySQL|Redis|GraphQL|REST|API|Git|CI/CD|DevOps|Agile|Scrum|HTML|CSS|SASS|LESS|Bootstrap|Tailwind|jQuery|Express\.js|Django|Flask|Spring|Hibernate|JPA|Maven|Gradle|npm|yarn|Webpack|Babel|ESLint|Prettier)\b'
            skills = re.findall(skills_pattern, text, re.IGNORECASE)
            return list(set(skills))

    def _extract_years_experience(self, text: str) -> Optional[int]:
        """Extract years of experience from text using LLM analysis."""
        try:
            prompt = f"""
            Extract the required years of experience from this job description. Return ONLY a number.
            
            Text: {text[:2000]}
            
            Return format: 5 (just the number, no text)
            """
            
            response = self._llm_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract number from response
            number_match = re.search(r'(\d+)', content)
            if number_match:
                return int(number_match.group(1))
            
            # Fallback to regex
            experience_pattern = r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)'
            experience_match = re.search(experience_pattern, text, re.IGNORECASE)
            if experience_match:
                return int(experience_match.group(1))
            
            return None
            
        except Exception as e:
            print(f"Error extracting years of experience: {e}")
            # Fallback to regex
            experience_pattern = r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)'
            experience_match = re.search(experience_pattern, text, re.IGNORECASE)
            if experience_match:
                return int(experience_match.group(1))
            return None

    def generate_enhanced_embedding(self, text: str, is_job_description: bool = False) -> List[float]:
        """Generate enhanced embeddings with context awareness."""
        if is_job_description:
            # For job descriptions, enhance the text with structured requirements
            enhanced_text = f"JOB DESCRIPTION: {text}"
        else:
            enhanced_text = f"CANDIDATE PROFILE: {text}"
        
        return self.generate_embedding(enhanced_text)

    def get_candidates_by_hybrid_search(self, job_description: str, num_results: int = 5) -> List[Dict]:
        """Get candidates using hybrid search (FTS + vector search) with fallback to document scan."""
        try:
            print(f"🔍 Performing enhanced hybrid search with Couchbase prefilters...")
            
            # Step 1: Analyze job description
            jd_analysis = self._analyze_job_description(job_description)
            if not jd_analysis:
                print("❌ Failed to analyze job description")
                return []
            
            print(f"✅ Job analysis complete: {jd_analysis.get('required_skills', [])} skills, {jd_analysis.get('years_experience')} years exp")
            
            # Step 2: Try to create working filters
            search_filters = self._create_working_filters(jd_analysis)
            
            # Step 3: If filters fail, try document scan approach
            if not search_filters:
                print(f"⚠️ Regular filters failed - trying document scan approach...")
                search_filters = self._create_working_filters_with_document_scan(jd_analysis)
            
            # Step 4: Perform search with filters
            if search_filters:
                print(f"🔍 Performing search with filters...")
                
                try:
                    from couchbase.search import SearchRequest
                    from datetime import timedelta
                    
                    # Create search request with filters
                    search_req = SearchRequest.create(search_filters)
                    
                    # Perform search
                    result = self.scope.search(
                        self.search_index_name,
                        search_req,
                        timeout=timedelta(seconds=10)
                    )
                    
                    # Get all results
                    all_candidates = []
                    for row in result.rows():
                        doc_id = row.id
                        if doc_id.startswith("candidate::"):
                            # Get the actual document
                            try:
                                doc = self.collection.get(doc_id).content_as[dict]
                                all_candidates.append(doc)
                            except Exception as e:
                                print(f"   ⚠️ Could not retrieve document {doc_id}: {e}")
                                continue
                    
                    print(f"   📊 Search returned {len(all_candidates)} candidates")
                    
                    # Step 5: If search returned 0 results, fall back to document scan approach
                    if len(all_candidates) == 0:
                        print(f"   ⚠️ Search returned 0 results - falling back to document scan approach...")
                        search_filters = self._create_working_filters_with_document_scan(jd_analysis)
                        
                        if search_filters:
                            # Try the document scan approach
                            search_req = SearchRequest.create(search_filters)
                            result = self.scope.search(
                                self.search_index_name,
                                search_req,
                                timeout=timedelta(seconds=10)
                            )
                            
                            # Get all results from document scan
                            all_candidates = []
                            for row in result.rows():
                                doc_id = row.id
                                if doc_id.startswith("candidate::"):
                                    try:
                                        doc = self.collection.get(doc_id).content_as[dict]
                                        all_candidates.append(doc)
                                    except Exception as e:
                                        print(f"   ⚠️ Could not retrieve document {doc_id}: {e}")
                                        continue
                            
                            print(f"   📊 Document scan returned {len(all_candidates)} candidates")
                    
                    # Step 6: Apply client-side filtering if using document scan approach
                    if isinstance(search_filters, MatchAllQuery):
                        print(f"   🔍 Applying client-side filtering...")
                        filtered_candidates = self._apply_client_side_filters(all_candidates, jd_analysis)
                        all_candidates = filtered_candidates
                        print(f"   📊 After client-side filtering: {len(all_candidates)} candidates")
                    
                    # Step 7: Perform vector search on filtered candidates
                    if all_candidates:
                        print(f"   🔍 Performing vector search on {len(all_candidates)} candidates...")
                        
                        # Create embeddings for job description
                        job_embedding = self._create_embedding(job_description)
                        if not job_embedding:
                            print("   ❌ Failed to create job description embedding")
                            return all_candidates[:num_results]
                        
                        print(f"   ✅ Job description embedding created (length: {len(job_embedding)})")
                        
                        # Calculate similarity scores
                        scored_candidates = []
                        for i, candidate in enumerate(all_candidates):
                            candidate_embedding = candidate.get("_default", {}).get("embedding")
                            if not candidate_embedding:
                                # Try root level
                                candidate_embedding = candidate.get("embedding")
                            
                            if candidate_embedding:
                                print(f"      Candidate {i+1}: Has embedding (length: {len(candidate_embedding)})")
                                try:
                                    similarity = self._calculate_cosine_similarity(job_embedding, candidate_embedding)
                                    scored_candidates.append({
                                        "candidate": candidate,
                                        "score": similarity
                                    })
                                    print(f"      ✅ Similarity calculated: {similarity:.4f}")
                                except Exception as e:
                                    print(f"      ❌ Error calculating similarity: {e}")
                            else:
                                print(f"      Candidate {i+1}: No embedding found")
                        
                        print(f"   📊 Successfully scored {len(scored_candidates)} candidates")
                        
                        if scored_candidates:
                            # Sort by similarity score
                            scored_candidates.sort(key=lambda x: x["score"], reverse=True)
                            
                            # Return top candidates
                            top_candidates = scored_candidates[:num_results]
                            print(f"   ✅ Vector search completed - returning top {len(top_candidates)} candidates")
                            
                            # Attach scores to candidate objects for web app display
                            final_candidates = []
                            for item in top_candidates:
                                candidate = item["candidate"]
                                score = item["score"]
                                
                                # Extract candidate data from _default structure
                                candidate_data = candidate.get("_default", {})
                                if not candidate_data:
                                    candidate_data = candidate  # Use root level if _default is empty
                                
                                # Create final candidate object with scores
                                final_candidate = {
                                    "name": candidate_data.get("name", "Unknown"),
                                    "email": candidate_data.get("email", ""),
                                    "phone": candidate_data.get("phone", ""),
                                    "location": candidate_data.get("location", ""),
                                    "skills": candidate_data.get("skills", []),
                                    "technical_skills": candidate_data.get("technical_skills", []),
                                    "summary": candidate_data.get("summary", ""),
                                    "experience": candidate_data.get("experience", ""),
                                    "education": candidate_data.get("education", ""),
                                    "years_experience": candidate_data.get("years_experience", 0),
                                    # Attach scores for web app
                                    "_score": score,  # Vector similarity score
                                    "_combined_score": score,  # For compatibility with web app
                                    "_vector_score": score,  # Alternative score field
                                }
                                
                                final_candidates.append(final_candidate)
                            
                            return final_candidates
                        else:
                            print(f"   ❌ No candidates with valid embeddings found")
                            # Return top candidates without vector scoring
                            print(f"   🔄 Falling back to top candidates without vector scoring")
                            
                            # Format candidates for web app display
                            final_candidates = []
                            for candidate in all_candidates[:num_results]:
                                candidate_data = candidate.get("_default", {})
                                if not candidate_data:
                                    candidate_data = candidate  # Use root level if _default is empty
                                
                                # Create final candidate object with default scores
                                final_candidate = {
                                    "name": candidate_data.get("name", "Unknown"),
                                    "email": candidate_data.get("email", ""),
                                    "phone": candidate_data.get("phone", ""),
                                    "location": candidate_data.get("location", ""),
                                    "skills": candidate_data.get("skills", []),
                                    "technical_skills": candidate_data.get("technical_skills", []),
                                    "summary": candidate_data.get("summary", ""),
                                    "experience": candidate_data.get("experience", ""),
                                    "education": candidate_data.get("education", ""),
                                    "years_experience": candidate_data.get("years_experience", 0),
                                    # Default scores for fallback
                                    "_score": 0.5,  # Default vector score
                                    "_combined_score": 0.5,  # Default combined score
                                    "_vector_score": 0.5,  # Default vector score
                                }
                                
                                final_candidates.append(final_candidate)
                            
                            return final_candidates
                    else:
                        print("   ❌ No candidates found after filtering")
                        return []
                        
                except Exception as e:
                    print(f"   ❌ Error performing search with filters: {e}")
                    # Fall back to pure vector search
                    print(f"   🔄 Falling back to pure vector search...")
                    return self._get_candidates_by_vector_search_only(job_description, num_results)
            else:
                print(f"❌ No working filters could be created")
                # Fall back to pure vector search
                print(f"🔄 Falling back to pure vector search...")
                return self._get_candidates_by_vector_search_only(job_description, num_results)
                
        except Exception as e:
            print(f"❌ Error in hybrid search: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _analyze_job_description(self, job_description: str) -> Optional[Dict[str, Any]]:
        """Analyze job description and return structured data."""
        try:
            return self.analyze_job_description(job_description)
        except Exception as e:
            print(f"Error analyzing job description: {e}")
            return None

    def _create_embedding(self, text: str) -> Optional[List[float]]:
        """Create embedding for a given text."""
        try:
            return self.generate_embedding(text)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _get_candidates_by_vector_search_only(self, job_description: str, num_results: int) -> List[Dict]:
        """Get candidates using pure vector search only."""
        print("🔍 Performing vector search...")
        
        # Create search request
        search_req = SearchRequest.create(
            VectorSearch.from_vector_query(
                VectorQuery(
                    field_name="embedding",
                    vector=[0.0] * 1024,  # Neutral vector
                    num_candidates=num_results * 2
                )
            )
        )
        
        try:
            result = self.scope.search(
                self.search_index_name,
                search_req,
                timeout=timedelta(seconds=20)
            )
            rows = list(result.rows())
        except Exception as e:
            print(f"Vector search failed: {e}")
            return []

        candidates = []
        for row in rows:
            try:
                doc = self.collection.get(row.id, timeout=timedelta(seconds=5))
                if doc and doc.value:
                    data = doc.value
                    data["_score"] = row.score
                    data["_id"] = row.id
                    candidates.append(data)
            except Exception as e:
                print(f"Error processing candidate {row.id}: {e}")
                continue
        
        return candidates[:num_results]

    def _validate_filters_before_use(self, search_filters: Any) -> bool:
        """Validate that filters will actually return results before using them."""
        try:
            print(f"🔍 Validating filters before use...")
            
            # Test the filters with a simple search
            from couchbase.search import SearchRequest
            from datetime import timedelta
            
            search_req = SearchRequest.create(search_filters)
            
            result = self.scope.search(
                self.search_index_name,
                search_req,
                timeout=timedelta(seconds=10)
            )
            rows = list(result.rows())
            
            if len(rows) > 0:
                print(f"   ✅ Filters validated - will return {len(rows)} results")
                return True
            else:
                print(f"   ❌ Filters validated - will return 0 results (too restrictive)")
                return False
                
        except Exception as e:
            print(f"   ❌ Error validating filters: {e}")
            return False

    def _create_working_filters(self, jd_analysis: Dict[str, Any]) -> Optional[Any]:
        """Create working filters for skills and experience."""
        print(f"🔍 Creating working filters with validation...")
        try:
            from couchbase.search import MatchQuery, NumericRangeQuery, ConjunctionQuery
            from datetime import timedelta
            
            filters = []
            
            # Create skills filter
            required_skills = jd_analysis.get("required_skills", [])
            if required_skills:
                print(f"   🔍 Creating skills filter for: {required_skills}")
                skill_queries = []
                
                for skill in required_skills:
                    # Look for skills at root level, not under _default
                    skill_query = MatchQuery("skills", skill)
                    skill_queries.append(skill_query)
                    print(f"       Added skill query for '{skill}' in skills")
                    
                    # Also check technical_skills at root level
                    tech_skill_query = MatchQuery("technical_skills", skill)
                    skill_queries.append(tech_skill_query)
                    print(f"       Added skill query for '{skill}' in technical_skills")
                
                if skill_queries:
                    # Use OR logic for skills (candidate can have any of the required skills)
                    skills_filter = DisjunctionQuery(*skill_queries)
                    filters.append(skills_filter)
                    print(f"   ✅ Created skills filter with {len(skill_queries)} skill queries")
            
            # Create experience filter
            required_years = jd_analysis.get("years_experience")
            if required_years:
                print(f"   🔍 Creating experience filter for: {required_years} years")
                # Look for years_experience at root level, not under _default
                min_years = max(1, int(required_years * 0.5))  # Allow 50% of required years
                experience_filter = NumericRangeQuery.min("years_experience", min_years)
                filters.append(experience_filter)
                print(f"   ✅ Created experience filter: years_experience >= {min_years}")
            
            if filters:
                # Combine filters with AND logic
                if len(filters) > 1:
                    print(f"   🔍 Combining {len(filters)} filters with AND logic")
                    combined_filter = ConjunctionQuery(*filters)
                else:
                    combined_filter = filters[0]
                
                print(f"   ✅ Created {len(filters)} working filters")
                return combined_filter
            else:
                print(f"   ⚠️  No filters created")
                return None
                
        except Exception as e:
            print(f"   ❌ Error creating working filters: {e}")
            return None

    def _find_working_skills_field(self, skill: str) -> Optional[str]:
        """Find a field name that actually works for skills search."""
        from datetime import timedelta
        
        field_names = [
            "skills", 
            "technical_skills", 
            "_default.skills", 
            "_default.technical_skills",
            "_default.skill",
            "_default.technology",
            "skill",
            "technology"
        ]
        
        for field in field_names:
            try:
                from couchbase.search import MatchQuery
                print(f"       Testing field '{field}' with skill '{skill}'...")
                
                # Debug: Print the exact query being created
                query = MatchQuery(field, skill)
                print(f"         Created query: {query}")
                
                # Fix: Use the correct Couchbase search request syntax
                search_req = SearchRequest.create(query)
                print(f"         Created search request: {search_req}")
                
                result = self.scope.search(
                    self.search_index_name,
                    search_req,
                    timeout=timedelta(seconds=5)
                )
                rows = list(result.rows())
                
                if len(rows) > 0:
                    return field
                    
            except Exception as e:
                print(f"       Field {field} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return None

    def _find_working_experience_field(self, required_years: int) -> Optional[str]:
        """Find a field name that actually works for experience search."""
        from datetime import timedelta
        
        field_names = [
            "years_experience", 
            "_default.years_experience", 
            "_default.experience", 
            "_default.years",
            "experience",
            "years"
        ]
        
        min_exp = max(0, int(required_years * 0.3))
        
        for field in field_names:
            try:
                from couchbase.search import NumericRangeQuery
                # Fix: Use the correct Couchbase search request syntax
                search_req = SearchRequest.create(
                    NumericRangeQuery(field=field, min=min_exp, inclusive_min=True)
                )
                
                result = self.scope.search(
                    self.search_index_name,
                    search_req,
                    timeout=timedelta(seconds=5)
                )
                rows = list(result.rows())
                
                if len(rows) > 0:
                    return field
                    
            except Exception as e:
                print(f"       Field {field} failed: {e}")
                continue
        
        return None

    def _calculate_filter_score(self, candidate: Dict[str, Any], filters: Dict[str, Any]) -> float:
        """Calculate how well a candidate matches the filter criteria."""
        score = 0.0
        max_score = 0.0
        
        # Skills match score
        if "skills" in filters and filters["skills"]:
            required_skills = filters["skills"] if isinstance(filters["skills"], list) else [filters["skills"]]
            candidate_skills = set()
            if candidate.get("skills"):
                candidate_skills.update(skill.lower() for skill in candidate["skills"])
            if candidate.get("technical_skills"):
                candidate_skills.update(skill.lower() for skill in candidate["technical_skills"])
            
            matches = len(set(skill.lower() for skill in required_skills) & candidate_skills)
            skills_score = matches / len(required_skills) if required_skills else 0.0
            score += skills_score * 0.4
            max_score += 0.4
        
        # Experience match score
        if "experience_range" in filters and candidate.get("years_experience"):
            exp_range = filters["experience_range"]
            if isinstance(exp_range, (list, tuple)) and len(exp_range) == 2:
                min_exp, max_exp = exp_range
                candidate_exp = candidate["years_experience"]
                
                if min_exp is not None and max_exp is not None:
                    if min_exp <= candidate_exp <= max_exp:
                        exp_score = 1.0  # Perfect match
                    elif candidate_exp >= min_exp:
                        exp_score = 0.8  # Overqualified
                    elif candidate_exp >= min_exp * 0.7:
                        exp_score = 0.6  # Close match
                    else:
                        exp_score = 0.2  # Underqualified
                elif min_exp is not None:
                    exp_score = 1.0 if candidate_exp >= min_exp else 0.2
                elif max_exp is not None:
                    exp_score = 1.0 if candidate_exp <= max_exp else 0.2
                else:
                    exp_score = 0.5
                
                score += exp_score * 0.3
                max_score += 0.3
        
        # Location match score
        if "location" in filters and candidate.get("location"):
            required_location = filters["location"].lower()
            candidate_location = candidate["location"].lower()
            
            if required_location in candidate_location or candidate_location in required_location:
                location_score = 1.0
            elif any(word in candidate_location for word in required_location.split()):
                location_score = 0.7
            else:
                location_score = 0.0
            
            score += location_score * 0.2
            max_score += 0.2
        
        # Education match score
        if "education" in filters and candidate.get("education"):
            required_education = filters["education"].lower()
            candidate_education = candidate["education"].lower()
            
            if required_education in candidate_education or candidate_education in required_education:
                education_score = 1.0
            else:
                education_score = 0.0
            
            score += education_score * 0.1
            max_score += 0.1
        
        # Normalize score
        return score / max_score if max_score > 0 else 0.0

    def _create_fallback_filters(self, jd_analysis: Dict[str, Any]) -> Optional[Any]:
        """Create fallback filters when working filters fail."""
        print(f"🔍 Creating fallback filters...")
        try:
            from couchbase.search import MatchQuery, DisjunctionQuery
            from datetime import timedelta
            
            # Use the actual field names that exist in the documents
            # From document inspection: skills and technical_skills are at root level
            actual_fields = ["skills", "technical_skills"]
            
            required_skills = jd_analysis.get("required_skills", [])
            if not required_skills:
                print(f"   ⚠️  No required skills specified")
                return None
            
            print(f"   🔍 Creating fallback skills filter for: {required_skills[:3]}")
            
            # Create simple skill queries for the fields that actually exist
            skill_queries = []
            for skill in required_skills[:3]:  # Limit to 3 skills for efficiency
                for field in actual_fields:
                    try:
                        skill_query = MatchQuery(field, skill)
                        skill_queries.append(skill_query)
                        print(f"       Added fallback query for '{skill}' in {field}")
                    except Exception as e:
                        print(f"       Error creating fallback query for '{skill}' in {field}: {e}")
                        continue
            
            if skill_queries:
                # Use OR logic for skills (candidate should have at least one required skill)
                fallback_filter = DisjunctionQuery(*skill_queries)
                print(f"   ✅ Created fallback filter with {len(skill_queries)} skill queries")
                return fallback_filter
            else:
                print(f"   ❌ Could not create any fallback skill queries")
                return None
                
        except Exception as e:
            print(f"   ❌ Error creating fallback filters: {e}")
            return None

    def _test_search_index_basic_functionality(self) -> bool:
        """Test if the search index is working at all with basic queries."""
        from datetime import timedelta
        
        print(f"🔍 Testing search index basic functionality...")
        
        try:
            # Test with MatchAll query first
            from couchbase.search import MatchAllQuery
            search_req = SearchRequest.create(MatchAllQuery())
            
            result = self.scope.search(
                self.search_index_name,
                search_req,
                timeout=timedelta(seconds=10)
            )
            rows = list(result.rows())
            print(f"   MatchAll query returned: {len(rows)} results")
            
            if len(rows) == 0:
                print(f"   ❌ CRITICAL: Search index returned 0 results with MatchAll query!")
                print(f"   🔍 This means the search index is not working at all")
                return False
            else:
                print(f"   ✅ Search index is working - MatchAll returns {len(rows)} results")
                
                # Test with a simple text search
                from couchbase.search import MatchQuery
                search_req = SearchRequest.create(MatchQuery("_all", "developer"))
                
                result = self.scope.search(
                    self.search_index_name,
                    search_req,
                    timeout=timedelta(seconds=10)
                )
                text_rows = list(result.rows())
                print(f"   Text search for 'developer' returned: {len(text_rows)} results")
                
                if len(text_rows) > 0:
                    print(f"   ✅ Text search is working")
                    return True
                else:
                    print(f"   ⚠️ Text search returned 0 results - may be field mapping issue")
                    return True  # Index works, but field mapping may be wrong
                    
        except Exception as e:
            print(f"   ❌ Search index test failed: {e}")
            return False

    def test_filters_directly(self, job_description: str = None) -> bool:
        """Test filters directly to diagnose issues."""
        print(f"🧪 Testing filters directly...")
        
        if not job_description:
            job_description = "React developer with TypeScript experience"
        
        try:
            # Test the new hybrid search approach that includes fallback
            print(f"🔍 Testing hybrid search with fallback to document scan...")
            candidates = self.get_candidates_by_hybrid_search(job_description, num_results=5)
            
            if candidates:
                print(f"✅ Hybrid search successful! Found {len(candidates)} candidates")
                print(f"\n📋 Top candidates:")
                for i, candidate in enumerate(candidates[:3], 1):
                    # Data is stored at root level, not under _default
                    name = candidate.get("name", "Unknown")
                    skills = candidate.get("skills", [])[:3]
                    experience = candidate.get("years_experience", 0)
                    print(f"   {i}. {name} - Skills: {skills} - Experience: {experience} years")
                return True
            else:
                print(f"❌ Hybrid search failed - no candidates found")
                return False
                
        except Exception as e:
            print(f"❌ Error testing filters directly: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _create_working_filters_flexible(self, jd_analysis: Dict[str, Any]) -> Optional[Any]:
        """Create working filters using flexible approaches that work with existing search indexes."""
        print(f"🔍 Creating flexible working filters...")
        
        try:
            # Since the search index doesn't have specific field mappings,
            # we'll use more flexible approaches that work with the existing index
            
            # Approach 1: Try using the _all field with key terms
            required_skills = jd_analysis.get("required_skills", [])
            if required_skills:
                print(f"   🔍 Trying flexible skills search with _all field...")
                
                # Use the first skill and try different search strategies
                first_skill = required_skills[0]
                
                # Strategy 1: Simple text search in _all field
                try:
                    from couchbase.search import MatchQuery
                    from datetime import timedelta
                    
                    # Search for the skill in the _all field
                    search_req = SearchRequest.create(MatchQuery("_all", first_skill))
                    
                    result = self.scope.search(
                        self.search_index_name,
                        search_req,
                        timeout=timedelta(seconds=10)
                    )
                    rows = list(result.rows())
                    
                    if len(rows) > 0:
                        print(f"   ✅ Flexible _all search works with '{first_skill}' ({len(rows)} results)")
                        return MatchQuery("_all", first_skill)
                    else:
                        print(f"   ❌ _all search failed with '{first_skill}' (0 results)")
                        
                except Exception as e:
                    print(f"   ❌ _all search error: {e}")
                
                # Strategy 2: Try wildcard search
                try:
                    from couchbase.search import WildcardQuery
                    
                    # Search for documents containing the skill (partial match)
                    wildcard_query = WildcardQuery("_all", f"*{first_skill}*")
                    search_req = SearchRequest.create(wildcard_query)
                    
                    result = self.scope.search(
                        self.search_index_name,
                        search_req,
                        timeout=timedelta(seconds=10)
                    )
                    rows = list(result.rows())
                    
                    if len(rows) > 0:
                        print(f"   ✅ Wildcard search works with '*{first_skill}*' ({len(rows)} results)")
                        return wildcard_query
                    else:
                        print(f"   ❌ Wildcard search failed with '*{first_skill}*' (0 results)")
                        
                except Exception as e:
                    print(f"   ❌ Wildcard search error: {e}")
                
                # Strategy 3: Try searching for common terms that should exist
                print(f"   🔍 Trying common terms search...")
                common_terms = ["developer", "engineer", "programmer", "software", "technology"]
                
                for term in common_terms:
                    try:
                        search_req = SearchRequest.create(MatchQuery("_all", term))
                        
                        result = self.scope.search(
                            self.search_index_name,
                            search_req,
                            timeout=timedelta(seconds=10)
                        )
                        rows = list(result.rows())
                        
                        if len(rows) > 0:
                            print(f"   ✅ Common term search works with '{term}' ({len(rows)} results)")
                            return MatchQuery("_all", term)
                        else:
                            print(f"   ❌ Common term search failed with '{term}' (0 results)")
                            
                    except Exception as e:
                        print(f"   ❌ Common term search error with '{term}': {e}")
                        continue
            
            # If no skills approach worked, try a very simple approach
            print(f"   🔍 Trying very simple approach...")
            try:
                from couchbase.search import MatchAllQuery
                
                # Just return a MatchAll query - this will work but won't filter
                print(f"   ⚠️ Using MatchAll query (no filtering)")
                return MatchAllQuery()
                
            except Exception as e:
                print(f"   ❌ MatchAll approach failed: {e}")
            
            print(f"   ❌ No flexible approaches worked")
            return None
            
        except Exception as e:
            print(f"   ❌ Error creating flexible filters: {e}")
            return None

    def _test_search_index_capabilities(self) -> Dict[str, Any]:
        """Test what fields the search index actually supports and can search on."""
        print(f"🔍 Testing search index capabilities...")
        
        capabilities = {
            "supports_all_field": False,
            "supports_specific_fields": False,
            "working_fields": [],
            "working_terms": []
        }
        
        try:
            from datetime import timedelta
            
            # Test 1: Test _all field with common terms
            print(f"   Testing _all field capabilities...")
            test_terms = ["developer", "React", "JavaScript", "Python", "experience"]
            
            for term in test_terms:
                try:
                    from couchbase.search import MatchQuery
                    search_req = SearchRequest.create(MatchQuery("_all", term))
                    
                    result = self.scope.search(
                        self.search_index_name,
                        search_req,
                        timeout=timedelta(seconds=5)
                    )
                    rows = list(result.rows())
                    
                    if len(rows) > 0:
                        print(f"     ✅ _all field works with '{term}' ({len(rows)} results)")
                        capabilities["supports_all_field"] = True
                        capabilities["working_terms"].append(term)
                    else:
                        print(f"     ❌ _all field failed with '{term}' (0 results)")
                        
                except Exception as e:
                    print(f"     ❌ _all field error with '{term}': {e}")
            
            # Test 2: Test specific field names that exist in documents
            print(f"   Testing specific field capabilities...")
            test_fields = [
                ("_default.skills", "React"),
                ("_default.technical_skills", "JavaScript"), 
                ("_default.years_experience", 3),
                ("skills", "React"),
                ("technical_skills", "JavaScript")
            ]
            
            for field_name, test_value in test_fields:
                try:
                    if isinstance(test_value, str):
                        from couchbase.search import MatchQuery
                        search_req = SearchRequest.create(MatchQuery(field_name, test_value))
                    else:
                        from couchbase.search import NumericRangeQuery
                        search_req = SearchRequest.create(
                            NumericRangeQuery(field=field_name, min=test_value, inclusive_min=True)
                        )
                    
                    result = self.scope.search(
                        self.search_index_name,
                        search_req,
                        timeout=timedelta(seconds=5)
                    )
                    rows = list(result.rows())
                    
                    if len(rows) > 0:
                        print(f"     ✅ Field '{field_name}' works with '{test_value}' ({len(rows)} results)")
                        capabilities["supports_specific_fields"] = True
                        capabilities["working_fields"].append(field_name)
                    else:
                        print(f"     ❌ Field '{field_name}' failed with '{test_value}' (0 results)")
                        
                except Exception as e:
                    print(f"     ❌ Field '{field_name}' error with '{test_value}': {e}")
            
            # Test 3: Test wildcard search
            print(f"   Testing wildcard search capabilities...")
            try:
                from couchbase.search import WildcardQuery
                wildcard_query = WildcardQuery("_all", "*React*")
                search_req = SearchRequest.create(wildcard_query)
                
                result = self.scope.search(
                    self.search_index_name,
                    search_req,
                    timeout=timedelta(seconds=5)
                )
                rows = list(result.rows())
                
                if len(rows) > 0:
                    print(f"     ✅ Wildcard search works with '*React*' ({len(rows)} results)")
                else:
                    print(f"     ❌ Wildcard search failed with '*React*' (0 results)")
                    
            except Exception as e:
                print(f"     ❌ Wildcard search error: {e}")
            
            print(f"   📊 Search index capabilities summary:")
            print(f"     - Supports _all field: {capabilities['supports_all_field']}")
            print(f"     - Supports specific fields: {capabilities['supports_specific_fields']}")
            print(f"     - Working fields: {capabilities['working_fields']}")
            print(f"     - Working terms: {capabilities['working_terms']}")
            
            return capabilities
            
        except Exception as e:
            print(f"   ❌ Error testing search index capabilities: {e}")
            return capabilities

    def _create_working_filters_with_document_scan(self, jd_analysis: Dict[str, Any]) -> Optional[Any]:
        """Create working filters by scanning documents directly since the search index doesn't support FTS."""
        print(f"🔍 Creating filters using document scan approach...")
        
        try:
            # Since the search index doesn't support FTS, we'll use a different approach
            # We'll create a filter that always returns results, but then filter the results client-side
            
            from couchbase.search import MatchAllQuery
            from datetime import timedelta
            
            # Return MatchAll query - this will return all documents
            # The actual filtering will happen in the hybrid search method
            print(f"   ✅ Using MatchAll query for document retrieval")
            print(f"   💡 Will apply client-side filtering for skills and experience")
            
            return MatchAllQuery()
            
        except Exception as e:
            print(f"   ❌ Error creating document scan filters: {e}")
            return None

    def _apply_client_side_filters(self, candidates: List[Dict], jd_analysis: Dict[str, Any]) -> List[Dict]:
        """Apply client-side filtering to candidates based on job requirements."""
        print(f"🔍 Applying client-side filters to {len(candidates)} candidates...")
        
        try:
            required_skills = jd_analysis.get("required_skills", [])
            required_years = jd_analysis.get("years_experience")
            
            print(f"   🔍 Required skills: {required_skills}")
            print(f"   🔍 Required experience: {required_years} years")
            
            filtered_candidates = []
            
            for i, candidate in enumerate(candidates):
                # Extract candidate data from the _default structure
                candidate_data = candidate.get("_default", {})
                
                print(f"   🔍 Candidate {i+1}: {candidate_data.get('name', 'Unknown')}")
                print(f"      Document keys: {list(candidate.keys())}")
                print(f"      _default keys: {list(candidate_data.keys())}")
                print(f"      Skills: {candidate_data.get('skills', [])[:3]}")
                print(f"      Technical skills: {candidate_data.get('technical_skills', [])[:3]}")
                print(f"      Experience: {candidate_data.get('years_experience', 'Unknown')}")
                
                # Check if data might be at root level instead of _default
                if not candidate_data.get('skills') and not candidate_data.get('technical_skills'):
                    print(f"      🔍 Checking root level for skills...")
                    root_skills = candidate.get('skills', [])
                    root_tech_skills = candidate.get('technical_skills', [])
                    print(f"      Root skills: {root_skills[:3]}")
                    print(f"      Root technical skills: {root_tech_skills[:3]}")
                    
                    # Use root level data if _default is empty
                    if root_skills or root_tech_skills:
                        candidate_data = candidate
                        print(f"      ✅ Using root level data instead of _default")
                
                # Check skills match
                skills_match = False
                if required_skills:
                    candidate_skills = candidate_data.get("skills", []) + candidate_data.get("technical_skills", [])
                    candidate_skills = [skill.lower() for skill in candidate_skills if skill]
                    
                    print(f"      Combined skills (lowercase): {candidate_skills[:5]}")
                    print(f"      Required skills (lowercase): {[skill.lower() for skill in required_skills]}")
                    
                    # Check if candidate has at least one required skill
                    for required_skill in required_skills:
                        if required_skill.lower() in candidate_skills:
                            skills_match = True
                            print(f"      ✅ Skills match: '{required_skill.lower()}' found in candidate skills")
                            break
                    
                    if not skills_match:
                        print(f"      ❌ No skills match found")
                else:
                    skills_match = True  # No skills requirement
                    print(f"      ✅ No skills requirement")
                
                # Check experience match
                experience_match = False
                if required_years and isinstance(required_years, (int, float)):
                    candidate_years = candidate_data.get("years_experience", 0)
                    min_required = required_years * 0.5
                    
                    print(f"      Required min experience: {min_required} years")
                    print(f"      Candidate experience: {candidate_years} years")
                    
                    if isinstance(candidate_years, (int, float)) and candidate_years >= min_required:
                        experience_match = True
                        print(f"      ✅ Experience match: {candidate_years} >= {min_required}")
                    else:
                        print(f"      ❌ Experience mismatch: {candidate_years} < {min_required}")
                else:
                    experience_match = True  # No experience requirement
                    print(f"      ✅ No experience requirement")
                
                # Candidate matches if both skills and experience match
                if skills_match and experience_match:
                    filtered_candidates.append(candidate)
                    print(f"      🎉 Candidate MATCHES requirements!")
                else:
                    print(f"      ❌ Candidate filtered out: Skills match: {skills_match}, Experience match: {experience_match}")
                
                print(f"      ---")
            
            print(f"   📊 Client-side filtering: {len(filtered_candidates)}/{len(candidates)} candidates match requirements")
            return filtered_candidates
            
        except Exception as e:
            print(f"   ❌ Error applying client-side filters: {e}")
            return candidates  # Return all candidates if filtering fails

