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
from couchbase.search import SearchRequest, MatchQuery, ConjunctionQuery, DisjunctionQuery, MatchAllQuery, NumericRangeQuery


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
        self.collection = self.bucket.collection("_default")
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
            return self._fallback_jd_analysis(job_description)

    def _fallback_jd_analysis(self, job_description: str) -> Dict[str, Any]:
        """Fallback analysis when LLM fails."""
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

    def get_candidates_by_vector(self, query_embedding: List[float], num_results: int = 5) -> List[Dict[str, Any]]:
        """Get candidates using pure vector search."""
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

    def search_candidates_by_skills(self, required_skills: List[str], num_results: int = 5) -> List[Dict[str, Any]]:
        """Search candidates by specific skills."""
        if not required_skills:
            return []
        
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
            enhanced_text = f"JOB DESCRIPTION: {text}"
        else:
            enhanced_text = f"CANDIDATE PROFILE: {text}"
        
        return self.generate_embedding(enhanced_text)

    def get_candidates_by_hybrid_search(self, job_description: str, num_results: int = 5) -> List[Dict]:
        """Get candidates using hybrid search (FTS + vector search) with fallback to document scan."""
        try:
            # Step 1: Analyze job description
            jd_analysis = self.analyze_job_description(job_description)
            if not jd_analysis:
                print("❌ Failed to analyze job description")
                return []
            
            # Step 2: Try to create working filters
            search_filters = self._create_working_filters(jd_analysis)
            
            # Step 3: If filters fail, try document scan approach
            if not search_filters:
                search_filters = self._create_working_filters_with_document_scan(jd_analysis)
            
            # Step 4: Perform search with filters
            if search_filters:
                try:
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
                            try:
                                doc = self.collection.get(doc_id).content_as[dict]
                                all_candidates.append(doc)
                            except Exception as e:
                                continue
                    
                    # Step 5: If search returned 0 results, fall back to document scan approach
                    if len(all_candidates) == 0:
                        search_filters = self._create_working_filters_with_document_scan(jd_analysis)
                        
                        if search_filters:
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
                                        continue
                    
                    # Step 6: Apply client-side filtering if using document scan approach
                    if isinstance(search_filters, MatchAllQuery):
                        filtered_candidates = self._apply_client_side_filters(all_candidates, jd_analysis)
                        all_candidates = filtered_candidates
                    
                    # Step 7: Perform vector search on filtered candidates
                    if all_candidates:
                        # Create embeddings for job description
                        job_embedding = self.generate_embedding(job_description)
                        if not job_embedding:
                            return all_candidates[:num_results]
                        
                        # Calculate similarity scores
                        scored_candidates = []
                        for candidate in all_candidates:
                            candidate_embedding = candidate.get("embedding")
                            
                            if candidate_embedding:
                                try:
                                    similarity = self._calculate_cosine_similarity(job_embedding, candidate_embedding)
                                    scored_candidates.append({
                                        "candidate": candidate,
                                        "score": similarity
                                    })
                                except Exception as e:
                                    continue
                        
                        if scored_candidates:
                            # Sort by similarity score
                            scored_candidates.sort(key=lambda x: x["score"], reverse=True)
                            
                            # Return top candidates
                            top_candidates = scored_candidates[:num_results]
                            
                            # Attach scores to candidate objects for web app display
                            final_candidates = []
                            for item in top_candidates:
                                candidate = item["candidate"]
                                score = item["score"]
                                
                                # Create final candidate object with scores
                                final_candidate = {
                                    "name": candidate.get("name", "Unknown"),
                                    "email": candidate.get("email", ""),
                                    "phone": candidate.get("phone", ""),
                                    "location": candidate.get("location", ""),
                                    "skills": candidate.get("skills", []),
                                    "technical_skills": candidate.get("technical_skills", []),
                                    "summary": candidate.get("summary", ""),
                                    "experience": candidate.get("experience", ""),
                                    "education": candidate.get("education", ""),
                                    "years_experience": candidate.get("years_experience", 0),
                                    # Attach scores for web app
                                    "_score": score,
                                    "_combined_score": score,
                                    "_vector_score": score,
                                }
                                
                                final_candidates.append(final_candidate)
                            
                            return final_candidates
                        else:
                            # Return top candidates without vector scoring
                            final_candidates = []
                            for candidate in all_candidates[:num_results]:
                                # Create final candidate object with default scores
                                final_candidate = {
                                    "name": candidate.get("name", "Unknown"),
                                    "email": candidate.get("email", ""),
                                    "phone": candidate.get("phone", ""),
                                    "location": candidate.get("location", ""),
                                    "skills": candidate.get("skills", []),
                                    "technical_skills": candidate.get("technical_skills", []),
                                    "summary": candidate.get("summary", ""),
                                    "experience": candidate.get("experience", ""),
                                    "education": candidate.get("education", ""),
                                    "years_experience": candidate.get("years_experience", 0),
                                    # Default scores for fallback
                                    "_score": 0.5,
                                    "_combined_score": 0.5,
                                    "_vector_score": 0.5,
                                }
                                
                                final_candidates.append(final_candidate)
                            
                            return final_candidates
                    else:
                        return []
                        
                except Exception as e:
                    print(f"Error performing search with filters: {e}")
                    return self._get_candidates_by_vector_search_only(job_description, num_results)
            else:
                print(f"No working filters could be created")
                return self._get_candidates_by_vector_search_only(job_description, num_results)
                
        except Exception as e:
            print(f"Error in hybrid search: {e}")
            return []

    def _create_working_filters(self, jd_analysis: Dict[str, Any]) -> Optional[Any]:
        """Create working filters for skills and experience."""
        try:
            filters = []
            
            # Create skills filter
            required_skills = jd_analysis.get("required_skills", [])
            if required_skills:
                skill_queries = []
                
                for skill in required_skills:
                    # Look for skills at root level
                    skill_query = MatchQuery("skills", skill)
                    skill_queries.append(skill_query)
                    
                    # Also check technical_skills at root level
                    tech_skill_query = MatchQuery("technical_skills", skill)
                    skill_queries.append(tech_skill_query)
                
                if skill_queries:
                    # Use OR logic for skills
                    skills_filter = DisjunctionQuery(*skill_queries)
                    filters.append(skills_filter)
            
            # Create experience filter
            required_years = jd_analysis.get("years_experience")
            if required_years:
                min_years = max(1, int(required_years * 0.5))
                experience_filter = NumericRangeQuery.min("years_experience", min_years)
                filters.append(experience_filter)
            
            if filters:
                # Combine filters with AND logic
                if len(filters) > 1:
                    combined_filter = ConjunctionQuery(*filters)
                else:
                    combined_filter = filters[0]
                
                return combined_filter
            else:
                return None
                
        except Exception as e:
            print(f"Error creating working filters: {e}")
            return None

    def _create_working_filters_with_document_scan(self, jd_analysis: Dict[str, Any]) -> Optional[Any]:
        """Create filters using document scan approach."""
        try:
            return MatchAllQuery()
        except Exception as e:
            print(f"Error creating document scan filters: {e}")
            return None

    def _apply_client_side_filters(self, candidates: List[Dict], jd_analysis: Dict[str, Any]) -> List[Dict]:
        """Apply client-side filtering to candidates based on job requirements."""
        try:
            required_skills = jd_analysis.get("required_skills", [])
            required_years = jd_analysis.get("years_experience")
            
            filtered_candidates = []
            
            for candidate in candidates:
                # Extract candidate data from the _default structure
                candidate_data = candidate.get("_default", {})
                
                # Check if data might be at root level instead of _default
                if not candidate_data.get('skills') and not candidate_data.get('technical_skills'):
                    root_skills = candidate.get('skills', [])
                    root_tech_skills = candidate.get('technical_skills', [])
                    
                    # Use root level data if _default is empty
                    if root_skills or root_tech_skills:
                        candidate_data = candidate
                
                # Check skills match
                skills_match = False
                if required_skills:
                    candidate_skills = candidate_data.get("skills", []) + candidate_data.get("technical_skills", [])
                    candidate_skills = [skill.lower() for skill in candidate_skills if skill]
                    
                    # Check if candidate has at least one required skill
                    for required_skill in required_skills:
                        if required_skill.lower() in candidate_skills:
                            skills_match = True
                            break
                else:
                    skills_match = True  # No skills requirement
                
                # Check experience match
                experience_match = False
                if required_years and isinstance(required_years, (int, float)):
                    candidate_years = candidate_data.get("years_experience", 0)
                    min_required = required_years * 0.5
                    
                    if isinstance(candidate_years, (int, float)) and candidate_years >= min_required:
                        experience_match = True
                else:
                    experience_match = True  # No experience requirement
                
                # Candidate matches if both skills and experience match
                if skills_match and experience_match:
                    filtered_candidates.append(candidate)
            
            return filtered_candidates
            
        except Exception as e:
            print(f"Error applying client-side filters: {e}")
            return candidates

    def _get_candidates_by_vector_search_only(self, job_description: str, num_results: int) -> List[Dict]:
        """Get candidates using pure vector search only."""
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

    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

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
