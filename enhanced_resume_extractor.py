#!/usr/bin/env python3
"""
Enhanced Resume Extractor - A powerful AI-powered resume parsing tool that extracts 
structured candidate information and stores it in Couchbase with high accuracy.
Built with Nebius AI for intelligent parsing and analysis.
"""

import os
import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import tempfile
import shutil

from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI

from cb_connection import CouchbaseConnection


class EnhancedResumeExtractor:
    """Enhanced resume extractor using Nebius LLM for accurate parsing."""
    
    def __init__(self):
        """Initialize the resume extractor."""
        load_dotenv()
        
        # Initialize Nebius LLM client
        self.llm_client = OpenAI(
            base_url=os.getenv("NEBIUS_API_BASE"),
            api_key=os.getenv("NEBIUS_API_KEY"),
        )
        
        # Initialize Couchbase connection
        try:
            self.cb = CouchbaseConnection()
            print("✅ Connected to Couchbase successfully")
        except Exception as e:
            print(f"❌ Failed to connect to Couchbase: {e}")
            raise
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF file with better error handling."""
        try:
            reader = PdfReader(pdf_path)
            texts = []
            
            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        texts.append(text.strip())
                    else:
                        print(f"  ⚠️  Page {page_num + 1} had no extractable text")
                except Exception as e:
                    print(f"  ⚠️  Error extracting text from page {page_num + 1}: {e}")
                    continue
            
            if not texts:
                raise ValueError("No text could be extracted from the PDF")
            
            return "\n\n".join(texts)
            
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {e}")
    
    def extract_resume_data(self, resume_text: str, filename: str) -> Dict[str, Any]:
        """
        Extract structured resume data using enhanced LLM prompts.
        This method uses multiple specialized prompts for different sections.
        """
        print(f"  🤖 Extracting resume data with enhanced LLM analysis...")
        
        try:
            # Step 1: Extract basic personal information
            personal_info = self._extract_personal_info(resume_text)
            
            # Step 2: Extract skills and technical information
            skills_info = self._extract_skills_info(resume_text)
            
            # Step 3: Extract work experience
            work_experience = self._extract_work_experience(resume_text)
            
            # Step 4: Extract education and certifications
            education_info = self._extract_education_info(resume_text)
            
            # Step 5: Calculate total years of experience
            total_experience = self._calculate_total_experience(work_experience)
            
            # Step 6: Generate summary
            summary = self._generate_summary(resume_text, personal_info, skills_info, work_experience)
            
            # Combine all extracted information
            resume_data = {
                "source": "resume",
                "filename": filename,
                "extracted_at": datetime.now().isoformat(),
                
                # Personal Information
                "name": personal_info.get("name", "Unknown"),
                "email": personal_info.get("email", ""),
                "phone": personal_info.get("phone", ""),
                "location": personal_info.get("location", ""),
                
                # Skills and Expertise
                "skills": skills_info.get("skills", []),
                "technical_skills": skills_info.get("technical_skills", []),
                "soft_skills": skills_info.get("soft_skills", []),
                "languages": skills_info.get("languages", []),
                
                # Experience
                "years_experience": total_experience,
                "total_experience_years": total_experience,
                "work_history": work_experience,
                
                # Education and Certifications
                "education": education_info.get("education", ""),
                "certifications": education_info.get("certifications", []),
                
                # Professional Links
                "github": personal_info.get("github", ""),
                "linkedin": personal_info.get("linkedin", ""),
                "portfolio": personal_info.get("portfolio", ""),
                
                # Summary and Analysis
                "summary": summary,
                "experience": f"{total_experience} years of professional experience",
                
                # Metadata
                "parsing_confidence": self._calculate_parsing_confidence(
                    personal_info, skills_info, work_experience, education_info
                )
            }
            
            # Validate and clean the data
            resume_data = self._validate_and_clean_data(resume_data)
            
            print(f"  ✅ Resume data extraction complete")
            print(f"     - Name: {resume_data.get('name', 'Unknown')}")
            print(f"     - Experience: {resume_data.get('years_experience', 0)} years")
            print(f"     - Skills: {len(resume_data.get('skills', []))} skills")
            print(f"     - Confidence: {resume_data.get('parsing_confidence', 0):.1f}%")
            
            return resume_data
            
        except Exception as e:
            print(f"  ❌ Resume data extraction failed: {e}")
            # Return fallback data
            return self._create_fallback_data(resume_text, filename)
    
    def _extract_personal_info(self, resume_text: str) -> Dict[str, Any]:
        """Extract personal information using specialized LLM prompt."""
        prompt = f"""
        Extract ONLY the personal information from this resume. Return a JSON object with these exact fields:

        {{
            "name": "Full legal name (not titles, not descriptions)",
            "email": "Email address if found",
            "phone": "Phone number if found",
            "location": "City, State/Country if found",
            "github": "GitHub profile URL if found",
            "linkedin": "LinkedIn profile URL if found",
            "portfolio": "Portfolio/website URL if found"
        }}

        IMPORTANT RULES:
        - For name: Extract ONLY the actual person's name, not section headers like "ABOUT ME" or "EXPERIENCE"
        - For name: If you see "EXPERIENCE [Name]" extract just the name part
        - For name: If you see "Web Developer Resume" or similar, look for the actual person's name elsewhere
        - For email: Extract complete email addresses only
        - For phone: Extract complete phone numbers only
        - For location: Extract city and state/country, not just "San Francisco" without state
        - Return ONLY valid JSON, no explanations or markdown

        Resume text:
        {resume_text[:3000]}

        JSON response:
        """

        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            return self._parse_json_response(content)
            
        except Exception as e:
            print(f"    ⚠️  Personal info extraction failed: {e}")
            return self._fallback_personal_info(resume_text)
    
    def _extract_skills_info(self, resume_text: str) -> Dict[str, Any]:
        """Extract skills information using specialized LLM prompt."""
        prompt = f"""
        Extract skills from this resume. Return a JSON object with these exact fields:

        {{
            "skills": ["skill1", "skill2", "skill3"],
            "technical_skills": ["tech1", "tech2", "tech3"],
            "soft_skills": ["soft1", "soft2", "soft3"],
            "languages": ["language1", "language2"]
        }}

        IMPORTANT RULES:
        - Extract actual skills mentioned in the resume
        - Technical skills: programming languages, frameworks, tools, technologies
        - Soft skills: communication, leadership, problem-solving, etc.
        - Languages: programming languages and human languages
        - Avoid generic terms like "development" or "coding"
        - Return ONLY valid JSON, no explanations

        Resume text:
        {resume_text[:3000]}

        JSON response:
        """

        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800
            )
            
            content = response.choices[0].message.content.strip()
            return self._parse_json_response(content)
            
        except Exception as e:
            print(f"    ⚠️  Skills extraction failed: {e}")
            return self._fallback_skills_info(resume_text)
    
    def _extract_work_experience(self, resume_text: str) -> List[Dict[str, Any]]:
        """Extract work experience using specialized LLM prompt."""
        prompt = f"""
        Extract work experience from this resume. Return a JSON array with job objects:

        [
            {{
                "company": "Company name",
                "title": "Job title",
                "duration": "Duration (e.g., 2020-2023)",
                "years": 3,
                "description": "Brief job description",
                "technologies": ["tech1", "tech2", "tech3"],
                "achievements": ["achievement1", "achievement2"]
            }}
        ]

        IMPORTANT RULES:
        - Extract actual job positions with company names and titles
        - Calculate years for each position
        - Include technologies mentioned for each role
        - Focus on quantifiable achievements
        - Return ONLY valid JSON array, no explanations

        Resume text:
        {resume_text[:4000]}

        JSON response:
        """

        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            result = self._parse_json_response(content)
            
            # Ensure it's a list
            if isinstance(result, list):
                return result
            elif isinstance(result, dict) and "work_history" in result:
                return result["work_history"]
            else:
                return []
            
        except Exception as e:
            print(f"    ⚠️  Work experience extraction failed: {e}")
            return self._fallback_work_experience(resume_text)
    
    def _extract_education_info(self, resume_text: str) -> Dict[str, Any]:
        """Extract education and certification information."""
        prompt = f"""
        Extract education and certifications from this resume. Return a JSON object:

        {{
            "education": "Education summary (degree, institution, year)",
            "certifications": ["cert1", "cert2", "cert3"]
        }}

        IMPORTANT RULES:
        - Extract actual degrees, institutions, and years
        - Include relevant certifications
        - Return ONLY valid JSON, no explanations

        Resume text:
        {resume_text[:2000]}

        JSON response:
        """

        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=600
            )
            
            content = response.choices[0].message.content.strip()
            return self._parse_json_response(content)
            
        except Exception as e:
            print(f"    ⚠️  Education extraction failed: {e}")
            return self._fallback_education_info(resume_text)
    
    def _calculate_total_experience(self, work_history: List[Dict[str, Any]]) -> int:
        """Calculate total years of experience from work history."""
        total_years = 0
        
        for job in work_history:
            years = job.get("years", 0)
            if isinstance(years, (int, float)):
                total_years += years
            elif isinstance(years, str):
                # Try to extract years from duration strings
                try:
                    # Look for patterns like "2020-2023" or "2 years"
                    duration = job.get("duration", "")
                    if "-" in duration:
                        # Extract years from date range
                        years_match = re.search(r'(\d{4})', duration)
                        if years_match:
                            total_years += 1  # Assume at least 1 year
                    elif "year" in duration.lower():
                        year_match = re.search(r'(\d+)', duration)
                        if year_match:
                            total_years += int(year_match.group(1))
                except:
                    pass
        
        return int(total_years) if total_years > 0 else 0
    
    def _generate_summary(self, resume_text: str, personal_info: Dict, skills_info: Dict, work_experience: List) -> str:
        """Generate a professional summary based on extracted information."""
        prompt = f"""
        Generate a professional summary (2-3 sentences) based on this information:

        Name: {personal_info.get('name', 'Professional')}
        Skills: {', '.join(skills_info.get('skills', [])[:5])}
        Experience: {len(work_experience)} positions
        Years: {sum(job.get('years', 0) for job in work_experience)} years

        Resume context: {resume_text[:1000]}

        Create a professional summary that highlights:
        - Professional identity and expertise
        - Key skills and experience level
        - Career focus or specialization

        Return ONLY the summary text, no markdown or formatting.
        """

        try:
            response = self.llm_client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"    ⚠️  Summary generation failed: {e}")
            return f"Experienced professional with expertise in {', '.join(skills_info.get('skills', [])[:3])}"
    
    def _parse_json_response(self, content: str) -> Any:
        """Parse JSON response from LLM with robust error handling."""
        try:
            # Clean the response
            content = content.strip()
            
            # Remove markdown code fences
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            
            if content.endswith("```"):
                content = content[:-3]
            
            # Find JSON content
            start_brace = content.find("{")
            start_bracket = content.find("[")
            
            if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
                # Object
                end_brace = content.rfind("}")
                if end_brace > start_brace:
                    json_content = content[start_brace:end_brace + 1]
                    return json.loads(json_content)
            elif start_bracket != -1:
                # Array
                end_bracket = content.rfind("]")
                if end_bracket > start_bracket:
                    json_content = content[start_bracket:end_bracket + 1]
                    return json.loads(json_content)
            
            raise ValueError("No valid JSON found in response")
            
        except Exception as e:
            print(f"    ⚠️  JSON parsing failed: {e}")
            return {}
    
    def _calculate_parsing_confidence(self, personal_info: Dict, skills_info: Dict, work_experience: List, education_info: Dict) -> float:
        """Calculate confidence score for the parsing results."""
        confidence = 0.0
        max_score = 0.0
        
        # Personal info confidence
        if personal_info.get("name") and personal_info["name"] != "Unknown":
            confidence += 25
        max_score += 25
        
        if personal_info.get("email"):
            confidence += 15
        max_score += 15
        
        if personal_info.get("phone"):
            confidence += 10
        max_score += 10
        
        # Skills confidence
        if skills_info.get("skills") and len(skills_info["skills"]) > 0:
            confidence += 20
        max_score += 20
        
        # Work experience confidence
        if work_experience and len(work_experience) > 0:
            confidence += 20
        max_score += 20
        
        # Education confidence
        if education_info.get("education"):
            confidence += 10
        max_score += 10
        
        return (confidence / max_score) * 100 if max_score > 0 else 0.0
    
    def _validate_and_clean_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean the extracted data."""
        # Ensure required fields exist
        required_fields = ["name", "email", "skills", "years_experience"]
        for field in required_fields:
            if field not in data:
                data[field] = "" if field == "email" else [] if field in ["skills", "technical_skills", "soft_skills"] else 0
        
        # Clean name field
        if data.get("name"):
            name = data["name"]
            # Remove common prefixes/suffixes
            name = re.sub(r'^(EXPERIENCE|ABOUT ME|Web Developer Resume|Resume)\s*', '', name, flags=re.IGNORECASE)
            name = re.sub(r'\s*\|.*$', '', name)  # Remove everything after |
            data["name"] = name.strip()
        
        # Ensure skills are lists
        for skill_field in ["skills", "technical_skills", "soft_skills", "languages", "certifications"]:
            if skill_field in data and not isinstance(data[skill_field], list):
                data[skill_field] = []
        
        # Ensure work_history is a list
        if not isinstance(data.get("work_history"), list):
            data["work_history"] = []
        
        return data
    
    def _create_fallback_data(self, resume_text: str, filename: str) -> Dict[str, Any]:
        """Create fallback data when LLM extraction fails."""
        print(f"    ⚠️  Using fallback extraction for {filename}")
        
        # Basic regex extraction
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
        
        # Extract skills using regex
        skills_pattern = r'\b(?:React|Angular|Vue|Node\.js|Python|Java|JavaScript|TypeScript|AWS|Azure|Docker|Kubernetes|MongoDB|PostgreSQL|MySQL|Redis|GraphQL|REST|API|Git|CI/CD|DevOps|Agile|Scrum|HTML|CSS|SASS|LESS|Bootstrap|Tailwind|jQuery|Express\.js|Django|Flask|Spring|Hibernate|JPA|Maven|Gradle|npm|yarn|Webpack|Babel|ESLint|Prettier)\b'
        skills = re.findall(skills_pattern, resume_text, re.IGNORECASE)
        
        return {
            "source": "resume",
            "filename": filename,
            "extracted_at": datetime.now().isoformat(),
            "name": name,
            "email": email,
            "phone": phone,
            "location": "",
            "skills": list(set(skills)),
            "technical_skills": list(set(skills)),
            "soft_skills": [],
            "languages": [],
            "years_experience": 0,
            "total_experience_years": 0,
            "work_history": [],
            "education": "",
            "certifications": [],
            "github": "",
            "linkedin": "",
            "portfolio": "",
            "summary": f"Professional with skills in {', '.join(skills[:3]) if skills else 'software development'}",
            "experience": "Experience not specified",
            "parsing_confidence": 30.0
        }
    
    def _fallback_personal_info(self, resume_text: str) -> Dict[str, Any]:
        """Fallback personal info extraction."""
        lines = resume_text.split('\n')
        name = lines[0].strip() if lines else "Unknown"
        
        # Clean name
        name = re.sub(r'^(EXPERIENCE|ABOUT ME|Web Developer Resume|Resume)\s*', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s*\|.*$', '', name)
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, resume_text)
        email = email_match.group(0) if email_match else ""
        
        # Extract phone
        phone_pattern = r'[\+]?[1-9][\d]{0,15}'
        phone_match = re.search(phone_pattern, resume_text)
        phone = phone_match.group(0) if phone_match else ""
        
        return {
            "name": name.strip(),
            "email": email,
            "phone": phone,
            "location": "",
            "github": "",
            "linkedin": "",
            "portfolio": ""
        }
    
    def _fallback_skills_info(self, resume_text: str) -> Dict[str, Any]:
        """Fallback skills extraction."""
        skills_pattern = r'\b(?:React|Angular|Vue|Node\.js|Python|Java|JavaScript|TypeScript|AWS|Azure|Docker|Kubernetes|MongoDB|PostgreSQL|MySQL|Redis|GraphQL|REST|API|Git|CI/CD|DevOps|Agile|Scrum|HTML|CSS|SASS|LESS|Bootstrap|Tailwind|jQuery|Express\.js|Django|Flask|Spring|Hibernate|JPA|Maven|Gradle|npm|yarn|Webpack|Babel|ESLint|Prettier)\b'
        skills = re.findall(skills_pattern, resume_text, re.IGNORECASE)
        
        return {
            "skills": list(set(skills)),
            "technical_skills": list(set(skills)),
            "soft_skills": [],
            "languages": []
        }
    
    def _fallback_work_experience(self, resume_text: str) -> List[Dict[str, Any]]:
        """Fallback work experience extraction."""
        work_history = []
        
        # Look for company patterns
        company_patterns = [
            r'(?:at|with|worked at|employed by)\s+([A-Z][A-Za-z\s&.,]+(?:Inc|Corp|LLC|Ltd|Company|Tech|Solutions|Systems))',
            r'([A-Z][A-Za-z\s&.,]+(?:Inc|Corp|LLC|Ltd|Company|Tech|Solutions|Systems))',
        ]
        
        for pattern in company_patterns:
            companies = re.findall(pattern, resume_text, re.IGNORECASE)
            for company in companies[:3]:  # Limit to 3 companies
                if company.strip() and len(company.strip()) > 2:
                    work_history.append({
                        "company": company.strip(),
                        "title": "Software Engineer",
                        "duration": "Unknown",
                        "years": 1,
                        "description": "Software development role",
                        "technologies": [],
                        "achievements": []
                    })
        
        return work_history
    
    def _fallback_education_info(self, resume_text: str) -> Dict[str, Any]:
        """Fallback education extraction."""
        return {
            "education": "",
            "certifications": []
        }
    
    def process_resume(self, pdf_path: str, filename: str) -> Dict[str, Any]:
        """Process a single resume PDF file."""
        print(f"📄 Processing: {filename}")
        
        try:
            # Extract text from PDF
            resume_text = self.extract_pdf_text(pdf_path)
            print(f"  📝 Extracted {len(resume_text)} characters from PDF")
            
            # Extract structured data
            resume_data = self.extract_resume_data(resume_text, filename)
            
            # Generate embedding
            print(f"  🔍 Generating embedding...")
            text_for_embedding = "\n".join([
                resume_data.get("name", ""),
                resume_data.get("summary", ""),
                resume_data.get("experience", ""),
                resume_data.get("education", ""),
                ", ".join(resume_data.get("skills", [])),
                ", ".join(resume_data.get("technical_skills", [])),
                str(resume_data.get("years_experience", "")),
                " ".join([job.get("description", "") for job in resume_data.get("work_history", [])])
            ])
            
            embedding = self.cb.generate_embedding(text_for_embedding)
            resume_data["embedding"] = embedding
            
            print(f"  ✅ Successfully processed: {filename}")
            return resume_data
            
        except Exception as e:
            print(f"  ❌ Error processing {filename}: {e}")
            # Return fallback data
            return self._create_fallback_data("", filename)
    
    def process_resumes_directory(self, resumes_dir: str) -> List[Dict[str, Any]]:
        """Process all resumes in a directory."""
        if not os.path.exists(resumes_dir):
            print(f"❌ Resumes directory '{resumes_dir}' not found")
            return []
        
        pdf_files = [f for f in os.listdir(resumes_dir) if f.lower().endswith(".pdf")]
        
        if not pdf_files:
            print(f"No PDF files found in '{resumes_dir}' directory.")
            return []
        
        print(f"Found {len(pdf_files)} PDF files to process...")
        
        processed_resumes = []
        
        for i, filename in enumerate(pdf_files, 1):
            print(f"\n📄 Processing {i}/{len(pdf_files)}: {filename}")
            file_path = os.path.join(resumes_dir, filename)
            
            try:
                resume_data = self.process_resume(file_path, filename)
                processed_resumes.append(resume_data)
                
                # Store in Couchbase
                doc_id = f"candidate::{os.path.splitext(filename)[0]}"
                self.cb.upsert_candidate(doc_id, resume_data)
                print(f"  💾 Stored in Couchbase: {doc_id}")
                
            except Exception as e:
                print(f"  ❌ Failed to process {filename}: {e}")
                continue
        
        return processed_resumes


def main():
    """Main function to run the enhanced resume extractor."""
    load_dotenv()
    
    # Check environment
    if not os.getenv("NEBIUS_API_KEY"):
        print("❌ NEBIUS_API_KEY not found in environment")
        return
    
    resumes_dir = os.getenv("RESUME_DIR", "resumes")
    
    try:
        # Initialize extractor
        extractor = EnhancedResumeExtractor()
        
        # Process resumes
        processed_resumes = extractor.process_resumes_directory(resumes_dir)
        
        print(f"\n🎉 Resume extraction complete!")
        print(f"✅ Successfully processed {len(processed_resumes)} resumes")
        
        # Summary statistics
        if processed_resumes:
            total_skills = sum(len(r.get("skills", [])) for r in processed_resumes)
            avg_confidence = sum(r.get("parsing_confidence", 0) for r in processed_resumes) / len(processed_resumes)
            
            print(f"\n📊 Summary Statistics:")
            print(f"   Total candidates: {len(processed_resumes)}")
            print(f"   Total skills extracted: {total_skills}")
            print(f"   Average parsing confidence: {avg_confidence:.1f}%")
            
            # Show top candidates by confidence
            top_candidates = sorted(processed_resumes, key=lambda x: x.get("parsing_confidence", 0), reverse=True)[:3]
            print(f"\n🏆 Top Candidates by Parsing Confidence:")
            for i, candidate in enumerate(top_candidates, 1):
                print(f"   {i}. {candidate.get('name', 'Unknown')} - {candidate.get('parsing_confidence', 0):.1f}%")
        
    except Exception as e:
        print(f"❌ Resume extraction failed: {e}")


if __name__ == "__main__":
    main() 