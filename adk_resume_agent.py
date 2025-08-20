import os
import json
import asyncio
from typing import Dict, List, Any
import datetime
import re # Added for fallback_resume_analysis

from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from pypdf import PdfReader

from cb_connection import CouchbaseConnection


load_dotenv()


def extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    texts: List[str] = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(texts)


def build_candidate_doc(parsed: Dict[str, Any], embedding: List[float]) -> Dict[str, Any]:
    return {
        "source": "resume",
        "name": parsed.get("name"),
        "email": parsed.get("email"),
        "phone": parsed.get("phone"),
        "location": parsed.get("location"),
        "skills": parsed.get("skills", []),
        "experience": parsed.get("experience"),
        "education": parsed.get("education"),
        "summary": parsed.get("summary"),
        "years_experience": parsed.get("years_experience", 0),
        "total_experience_years": parsed.get("total_experience_years", parsed.get("years_experience", 0)),
        "work_history": parsed.get("work_history", []),
        "technical_skills": parsed.get("technical_skills", []),
        "soft_skills": parsed.get("soft_skills", []),
        "certifications": parsed.get("certifications", []),
        "languages": parsed.get("languages", []),
        "github": parsed.get("github", ""),
        "linkedin": parsed.get("linkedin", ""),
        "embedding": embedding,
        "created_at": datetime.datetime.now().isoformat(),
        "updated_at": datetime.datetime.now().isoformat()
    }


async def run_adk_resume_ingestion_async(resume_dir: str = "resumes") -> List[str]:
    """Parse resumes via ADK LLM and insert into Couchbase.

    Returns list of upserted document IDs.
    """
    # Base LLM config
    base_model = LiteLlm(
        model="nebius/Qwen/Qwen2.5-72B-Instruct",
        api_base=os.getenv("NEBIUS_API_BASE"),
        api_key=os.getenv("NEBIUS_API_KEY"),
    )

    # Define the parsing agent instruction
    parse_instruction = (
        "You are a resume parsing assistant. You will receive resume text content. "
        "Extract a clean JSON with: name, email, phone, location, skills (array), "
        "experience (concise text), education (concise text), summary (2-3 sentences), "
        "years_experience (number - total years across all jobs), work_history (array of job objects with company, title, duration, description, years, technologies), "
        "technical_skills (array of technical skills), soft_skills (array of soft skills), "
        "certifications (array), languages (array), github (string), linkedin (string). "
        "CRITICAL: Return ONLY valid JSON. Do not include any thinking, explanations, or markdown formatting. "
        "The response must start with { and end with }. Example format:\n"
        "{\n"
        '  "name": "John Doe",\n'
        '  "email": "john@example.com",\n'
        '  "phone": "+1-555-0123",\n'
        '  "location": "San Francisco, CA",\n'
        '  "skills": ["Python", "JavaScript", "AWS"],\n'
        '  "experience": "5 years as Software Engineer...",\n'
        '  "education": "BS Computer Science, Stanford University",\n'
        '  "summary": "Experienced software engineer...",\n'
        '  "years_experience": 5,\n'
        '  "work_history": [\n'
        '    {\n'
        '      "company": "Tech Corp",\n'
        '      "title": "Senior Software Engineer",\n'
        '      "duration": "2020-2023",\n'
        '      "years": 3,\n'
        '      "description": "Led development of web applications using React and Node.js",\n'
        '      "technologies": ["React", "Node.js", "Python"]\n'
        '    }\n'
        '  ],\n'
        '  "technical_skills": ["React", "Node.js", "Python", "AWS"],\n'
        '  "soft_skills": ["Leadership", "Communication", "Problem Solving"],\n'
        '  "certifications": ["AWS Certified Developer"],\n'
        '  "languages": ["English", "Spanish"],\n'
        '  "github": "github.com/johndoe",\n'
        '  "linkedin": "linkedin.com/in/johndoe"\n'
        "}\n\n"
        "IMPORTANT: Do not use <think> tags or any other formatting. Output pure JSON only. "
        "For years_experience, calculate the total years across all work experience. "
        "For each job in work_history, include the 'years' field with the duration in years."
    )

    # This agent consumes resume plain text and outputs JSON text
    parse_agent = LlmAgent(
        name="ResumeParserAgent",
        model=base_model,
        instruction=parse_instruction,
        output_key="parsed_resume_json",
    )

    pipeline = SequentialAgent(name="ResumeIngestionPipeline", sub_agents=[parse_agent])

    session_service = InMemorySessionService()
    runner = Runner(agent=pipeline, app_name="resume_ingestion", session_service=session_service)

    cb = CouchbaseConnection()

    if not os.path.exists(resume_dir):
        print(f"Resume directory '{resume_dir}' does not exist. Creating it.")
        os.makedirs(resume_dir, exist_ok=True)
        return []

    upserted_ids: List[str] = []
    pdf_files = [f for f in os.listdir(resume_dir) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDF files found in '{resume_dir}' directory.")
        return []
    
    print(f"Found {len(pdf_files)} PDF files to process...")
    
    for i, filename in enumerate(pdf_files, 1):
        print(f"Processing {i}/{len(pdf_files)}: {filename}")
        file_path = os.path.join(resume_dir, filename)

        try:
            resume_text = extract_pdf_text(file_path)
            if not resume_text.strip():
                print(f"  Warning: No text extracted from {filename}")
                continue
                
            content = types.Content(role="user", parts=[
                types.Part(text=f"File: {filename}\n\n{resume_text}")
            ])

            # Create session first
            session_id = f"s_{filename.replace(' ', '_').replace('(', '').replace(')', '')}"
            await session_service.create_session(
                app_name="resume_ingestion",
                user_id="u",
                session_id=session_id
            )

            events = runner.run(user_id="u", session_id=session_id, new_message=content)
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            continue

        # Try to get the response
        parsed_json_text = None
        for e in events:
            if e.is_final_response():
                parsed_json_text = e.content.parts[0].text
                print(f"  Raw agent response: {parsed_json_text[:200]}...")
                break

        # If no response, try one more time with a simpler prompt
        if not parsed_json_text:
            print(f"  No response from ADK agent, trying retry...")
            try:
                retry_content = types.Content(role="user", parts=[
                    types.Part(text=f"Parse this resume into JSON: {resume_text[:1000]}")
                ])
                retry_events = runner.run(user_id="u", session_id=session_id, new_message=retry_content)
                for e in retry_events:
                    if e.is_final_response():
                        parsed_json_text = e.content.parts[0].text
                        break
            except Exception as retry_error:
                print(f"  Retry also failed: {retry_error}")
                continue

        if not parsed_json_text:
            continue

        # Robust JSON extraction (handles various response formats)
        parsed: Dict[str, Any]
        cleaned = (parsed_json_text or "").strip()
        
        # Remove thinking/explanation text before JSON
        if "<think>" in cleaned.lower():
            # Find the start of JSON after thinking
            json_start = cleaned.find("{")
            if json_start != -1:
                cleaned = cleaned[json_start:]
            else:
                print(f"  Warning: No JSON found in response for {filename}")
                continue
        
        # Remove markdown code fences
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 3:
                cleaned = "\n".join(lines[1:-1])
        
        # Strip potential json language identifier
        if cleaned.lower().startswith("json\n"):
            cleaned = cleaned[5:]
        
        # Find the first { and last } to extract just the JSON
        start_brace = cleaned.find("{")
        end_brace = cleaned.rfind("}")
        
        if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
            cleaned = cleaned[start_brace:end_brace + 1]
        
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"  Failed to parse JSON for {filename}: {str(e)}")
            print(f"  Raw response preview: {parsed_json_text[:200]}...")
            print(f"  Cleaned JSON preview: {cleaned[:200]}...")
            
            # Fallback: Try direct LLM call for JSON parsing
            print(f"  Attempting fallback JSON parsing...")
            try:
                fallback_response = base_model.generate_content(
                    f"Parse this resume text into JSON format with keys: name, email, phone, location, skills (array), experience, education, summary, years_experience, work_history, technical_skills, soft_skills. Return ONLY valid JSON:\n\n{resume_text[:2000]}"
                )
                fallback_json = fallback_response.text.strip()
                
                # Clean the fallback response
                if fallback_json.startswith("```"):
                    lines = fallback_json.splitlines()
                    if len(lines) >= 3:
                        fallback_json = "\n".join(lines[1:-1])
                
                start_brace = fallback_json.find("{")
                end_brace = fallback_json.rfind("}")
                if start_brace != -1 and end_brace != -1:
                    fallback_json = fallback_json[start_brace:end_brace + 1]
                    parsed = json.loads(fallback_json)
                    print(f"  ✅ Fallback parsing successful for {filename}")
                else:
                    print(f"  ❌ Fallback parsing also failed for {filename}")
                    continue
            except Exception as fallback_error:
                print(f"  ❌ Fallback parsing failed: {fallback_error}")
                continue

        # Validate required fields
        required_fields = ["name", "email"]
        missing_fields = [field for field in required_fields if not parsed.get(field)]
        if missing_fields:
            print(f"  Warning: Missing required fields for {filename}: {missing_fields}")
            # Try to extract basic info if name is missing
            if not parsed.get("name"):
                # Extract first line as potential name
                lines = resume_text.split('\n')
                for line in lines[:5]:  # Check first 5 lines
                    line = line.strip()
                    if line and len(line) > 2 and len(line) < 100:
                        parsed["name"] = line
                        break
        
        # Build enhanced text for embeddings
        text_for_embedding = "\n".join(
            [
                parsed.get("name", ""),
                parsed.get("summary", ""),
                parsed.get("experience", ""),
                parsed.get("education", ""),
                ", ".join(parsed.get("skills", [])),
                ", ".join(parsed.get("technical_skills", [])),
                str(parsed.get("years_experience", "")),
                # Include work history descriptions
                " ".join([job.get("description", "") for job in parsed.get("work_history", [])])
            ]
        )
        embedding = cb.generate_embedding(text_for_embedding)

        doc = build_candidate_doc(parsed, embedding)
        doc_id = f"candidate::{os.path.splitext(filename)[0]}"
        cb.upsert_candidate(doc_id, doc)
        upserted_ids.append(doc_id)
        print(f"  ✅ Successfully processed and stored: {filename}")

    return upserted_ids


def run_adk_resume_ingestion(resume_dir: str = "resumes") -> List[str]:
    """Synchronous wrapper for resume ingestion."""
    return asyncio.run(run_adk_resume_ingestion_async(resume_dir))


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
        
        # Extract skills using regex
        skills_pattern = r'\b(?:React|Angular|Vue|Node\.js|Python|Java|JavaScript|TypeScript|AWS|Azure|Docker|Kubernetes|MongoDB|PostgreSQL|MySQL|Redis|GraphQL|REST|API|Git|CI/CD|DevOps|Agile|Scrum|HTML|CSS|SASS|LESS|Bootstrap|Tailwind|jQuery|Express\.js|Django|Flask|Spring|Hibernate|JPA|Maven|Gradle|npm|yarn|Webpack|Babel|ESLint|Prettier)\b'
        skills = re.findall(skills_pattern, resume_text, re.IGNORECASE)
        
        # Extract years of experience
        experience_pattern = r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)'
        experience_match = re.search(experience_pattern, resume_text, re.IGNORECASE)
        years_experience = int(experience_match.group(1)) if experience_match else 0
        
        # Extract work history patterns
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
                        "title": "Software Engineer",  # Default title
                        "duration": "Unknown",
                        "years": 1,
                        "description": "Software development role",
                        "technologies": skills[:3] if skills else []
                    })
        
        return {
            "name": name,
            "email": email,
            "phone": phone,
            "location": "",
            "years_experience": years_experience,
            "total_experience_years": years_experience,
            "skills": list(set(skills)),
            "technical_skills": list(set(skills)),
            "soft_skills": [],
            "experience": f"{years_experience} years of experience" if years_experience else "Experience not specified",
            "education": "",
            "summary": f"Software professional with {years_experience} years of experience" if years_experience else "Software professional",
            "work_history": work_history,
            "certifications": [],
            "languages": [],
            "github": "",
            "linkedin": ""
        }