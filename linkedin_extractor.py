#!/usr/bin/env python3
"""
LinkedIn Profile Extractor for HR Sourcing System
Uses BrightData MCP server to extract LinkedIn profiles and store them in Couchbase.
"""

import os
import json
import logging
import asyncio
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import nest_asyncio
nest_asyncio.apply()


# Import your existing modules
from cb_connection import CouchbaseConnection
from mcp_server import get_mcp_server, wait_for_initialization
from job_agent import run_analysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LinkedInExtractor:
    """Extracts LinkedIn profiles using BrightData MCP server and stores them in Couchbase."""
    
    def __init__(self, profile_timeout: int = 500, delay_between_profiles: int = 10):
        """Initialize the LinkedIn extractor.
        
        Args:
            profile_timeout: Timeout in seconds for profile extraction (default: 60)
            delay_between_profiles: Delay in seconds between profile processing (default: 5)
        """
        load_dotenv()
        
        # Configuration
        self.profile_timeout = profile_timeout
        self.delay_between_profiles = delay_between_profiles
        
        # Initialize Couchbase connection
        try:
            self.cb_connection = CouchbaseConnection()
            logger.info("✅ Connected to Couchbase")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Couchbase: {e}")
            raise
        
        # Initialize MCP server
        self.mcp_server = None
        
    async def initialize_mcp(self):
        """Initialize the MCP server for BrightData."""
        logger.info("🔧 Initializing MCP server...")
        try:
            await wait_for_initialization()
            self.mcp_server = get_mcp_server()
            if self.mcp_server:
                logger.info("✅ MCP server initialized successfully")
                return True
            else:
                logger.error("❌ Failed to get MCP server instance")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize MCP server: {e}")
            return False
    
    def load_linkedin_urls(self, excel_file_path: str) -> List[Dict[str, str]]:
        """Load LinkedIn URLs from the Excel file."""
        logger.info(f"📊 Loading LinkedIn URLs from: {excel_file_path}")
        
        try:
            # Read the Excel file
            df = pd.read_excel(excel_file_path)
            
            # Look for the LinkedIn URL column
            linkedin_column = None
            for col in df.columns:
                if 'linkedin' in col.lower() or 'url' in col.lower():
                    linkedin_column = col
                    break
            
            if not linkedin_column:
                logger.error("❌ No LinkedIn URL column found in Excel file")
                logger.info(f"Available columns: {list(df.columns)}")
                return []
            
            logger.info(f"✅ Found LinkedIn column: {linkedin_column}")
            
            # Extract URLs and basic info
            urls_data = []
            for index, row in df.iterrows():
                linkedin_url = row[linkedin_column]
                
                # Skip empty URLs
                if pd.isna(linkedin_url) or not str(linkedin_url).strip():
                    continue
                
                # Clean the URL
                linkedin_url = str(linkedin_url).strip()
                if not linkedin_url.startswith('http'):
                    linkedin_url = f"https://{linkedin_url}"
                
                # Extract other available information
                candidate_info = {
                    'linkedin_url': linkedin_url,
                    'row_index': index,
                    'source_file': excel_file_path
                }
                
                # Try to get additional fields if available
                for col in df.columns:
                    if col != linkedin_column and not pd.isna(row[col]):
                        candidate_info[col.lower().replace(' ', '_')] = str(row[col]).strip()
                
                urls_data.append(candidate_info)
            
            logger.info(f"✅ Loaded {len(urls_data)} LinkedIn URLs")
            return urls_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load LinkedIn URLs: {e}")
            return []
    
    async def extract_linkedin_profile(self, linkedin_url: str) -> Optional[Dict[str, Any]]:
        """Extract LinkedIn profile data using the job agent system."""
        logger.info(f"🔍 Extracting profile from: {linkedin_url}")
        
        try:
            # Use the existing job agent to analyze the LinkedIn profile with configurable timeout
            profile_data = await asyncio.wait_for(
                run_analysis(self.mcp_server, linkedin_url),
                timeout=float(self.profile_timeout)  # Use configurable timeout
            )
            
            if not profile_data:
                logger.warning(f"⚠️  No profile data extracted from: {linkedin_url}")
                return None
            
            # Log the raw response for debugging
            logger.info(f"📄 Raw profile data received (length: {len(str(profile_data))})")
            logger.info(f"📄 Profile data type: {type(profile_data)}")
            
            # Parse the markdown response to extract structured data
            structured_data = self._parse_profile_response(profile_data, linkedin_url)
            
            logger.info(f"✅ Successfully extracted profile for: {linkedin_url}")
            return structured_data
            
        except asyncio.TimeoutError:
            logger.error(f"⏰ Timeout ({self.profile_timeout}s) while extracting profile from: {linkedin_url}")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to extract profile from {linkedin_url}: {e}")
            return None
    
    def _parse_profile_response(self, profile_response: str, linkedin_url: str) -> Dict[str, Any]:
        """Parse the markdown response from the job agent into structured data."""
        logger.info("🔧 Parsing profile response...")
        
        # Initialize structured data with defaults
        structured_data = {
            "name": "Unknown",
            "email": "",
            "phone": "",
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
            "linkedin": linkedin_url,
            "source": "linkedin_extraction",
            "extraction_date": datetime.now().isoformat(),
            "parsing_confidence": 0.0
        }
        
        try:
            # Extract profile summary section
            if "## 👤 Profile Summary" in profile_response:
                summary_start = profile_response.find("## 👤 Profile Summary")
                summary_end = profile_response.find("##", summary_start + 1)
                if summary_end == -1:
                    summary_end = len(profile_response)
                
                summary_text = profile_response[summary_start:summary_end]
                structured_data["summary"] = self._extract_text_after_header(summary_text, "## 👤 Profile Summary")
            
            # Extract skills section
            if "## 🎯 Your Top Skills:" in profile_response:
                skills_start = profile_response.find("## 🎯 Your Top Skills:")
                skills_end = profile_response.find("##", skills_start + 1)
                if skills_end == -1:
                    skills_end = len(profile_response)
                
                skills_text = profile_response[skills_start:skills_end]
                skills_list = self._extract_list_items(skills_text)
                structured_data["skills"] = skills_list
                structured_data["technical_skills"] = skills_list  # Assume top skills are technical
            
            # Extract suggested roles for additional context
            if "## 💡 Suggested Roles:" in profile_response:
                roles_start = profile_response.find("## 💡 Suggested Roles:")
                roles_end = profile_response.find("##", roles_start + 1)
                if roles_end == -1:
                    roles_end = len(profile_response)
                
                roles_text = profile_response[roles_start:roles_end]
                structured_data["experience"] = self._extract_text_after_header(roles_text, "## 💡 Suggested Roles:")
            
            # Extract job matches for additional context
            if "## 💼 Current Job Matches:" in profile_response:
                matches_start = profile_response.find("## 💼 Current Job Matches:")
                matches_end = profile_response.find("##", matches_start + 1)
                if matches_end == -1:
                    matches_end = len(profile_response)
                
                matches_text = profile_response[matches_start:matches_end]
                # Extract company names and job titles
                companies = self._extract_companies_from_matches(matches_text)
                if companies:
                    structured_data["work_history"] = [{"company": company, "title": "Current Opportunity", "duration": "Present", "years": 0, "description": "", "technologies": []} for company in companies]
            
            # Try to extract name from the first line of summary
            if structured_data["summary"]:
                first_line = structured_data["summary"].split('\n')[0].strip()
                if first_line and not first_line.startswith('##'):
                    structured_data["name"] = first_line
            
            # Calculate parsing confidence based on extracted data
            confidence_score = self._calculate_parsing_confidence(structured_data)
            structured_data["parsing_confidence"] = confidence_score
            
            logger.info(f"✅ Parsed profile with confidence: {confidence_score:.2f}")
            return structured_data
            
        except Exception as e:
            logger.error(f"❌ Error parsing profile response: {e}")
            logger.error(f"❌ Response type: {type(profile_response)}")
            logger.error(f"❌ Response length: {len(str(profile_response))}")
            structured_data["parsing_confidence"] = 0.0
            return structured_data
    
    def _parse_expected_markdown(self, profile_response: str, structured_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the expected markdown format from the job agent."""
        try:
            # Extract profile summary section
            if "## 👤 Profile Summary" in profile_response:
                summary_start = profile_response.find("## 👤 Profile Summary")
                summary_end = profile_response.find("##", summary_start + 1)
                if summary_end == -1:
                    summary_end = len(profile_response)
                
                summary_text = profile_response[summary_start:summary_end]
                structured_data["summary"] = self._extract_text_after_header(summary_text, "## 👤 Profile Summary")
            
            # Extract skills section
            if "## 🎯 Your Top Skills:" in profile_response:
                skills_start = profile_response.find("## 🎯 Your Top Skills:")
                skills_end = profile_response.find("##", skills_start + 1)
                if skills_end == -1:
                    skills_end = len(profile_response)
                
                skills_text = profile_response[skills_start:skills_end]
                skills_list = self._extract_list_items(skills_text)
                structured_data["skills"] = skills_list
                structured_data["technical_skills"] = skills_list
            
            # Extract suggested roles
            if "## 💡 Suggested Roles:" in profile_response:
                roles_start = profile_response.find("## 💡 Suggested Roles:")
                roles_end = profile_response.find("##", roles_start + 1)
                if roles_end == -1:
                    roles_end = len(profile_response)
                
                roles_text = profile_response[roles_start:roles_end]
                structured_data["experience"] = self._extract_text_after_header(roles_text, "## 💡 Suggested Roles:")
            
            # Extract job matches
            if "## 💼 Current Job Matches:" in profile_response:
                matches_start = profile_response.find("## 💼 Current Job Matches:")
                matches_end = profile_response.find("##", matches_start + 1)
                if matches_end == -1:
                    matches_end = len(profile_response)
                
                matches_text = profile_response[matches_start:matches_end]
                companies = self._extract_companies_from_matches(matches_text)
                if companies:
                    structured_data["work_history"] = [
                        {"company": company, "title": "Current Opportunity", "duration": "Present", "years": 0, "description": "", "technologies": []} 
                        for company in companies
                    ]
            
            # Try to extract name from summary
            if structured_data["summary"]:
                first_line = structured_data["summary"].split('\n')[0].strip()
                if first_line and not first_line.startswith('##'):
                    structured_data["name"] = first_line
            
            return structured_data
            
        except Exception as e:
            logger.error(f"❌ Error in expected markdown parsing: {e}")
            return structured_data
    
    def _parse_alternative_format(self, profile_response: str, structured_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse alternative LinkedIn profile formats."""
        try:
            # Look for common LinkedIn profile patterns
            lines = profile_response.split('\n')
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                # Look for name patterns
                if not structured_data["name"] or structured_data["name"] == "Unknown":
                    if "linkedin.com/in/" in profile_response:
                        # Try to extract name from URL
                        url_parts = profile_response.split("linkedin.com/in/")
                        if len(url_parts) > 1:
                            name_part = url_parts[1].split('/')[0].split('?')[0]
                            if name_part and len(name_part) > 2:
                                structured_data["name"] = name_part.replace('-', ' ').replace('_', ' ').title()
                
                # Look for skills in the text
                if "skill" in line.lower() or "technology" in line.lower() or "tech" in line.lower():
                    # Extract potential skills from this line
                    potential_skills = self._extract_potential_skills(line)
                    if potential_skills:
                        structured_data["skills"].extend(potential_skills)
                        structured_data["technical_skills"].extend(potential_skills)
            
            # Use the entire response as summary if no structured summary found
            if not structured_data["summary"]:
                structured_data["summary"] = profile_response[:1000]  # Limit to first 1000 chars
            
            return structured_data
            
        except Exception as e:
            logger.error(f"❌ Error in alternative format parsing: {e}")
            return structured_data
    
    def _parse_basic_text(self, profile_response: str, structured_data: Dict[str, Any]) -> Dict[str, Any):
        """Parse basic text format as fallback."""
        try:
            # Use the entire response as summary
            structured_data["summary"] = profile_response[:1000]
            
            # Try to extract any recognizable patterns
            if "linkedin.com/in/" in profile_response:
                # Extract name from URL
                url_parts = profile_response.split("linkedin.com/in/")
                if len(url_parts) > 1:
                    name_part = url_parts[1].split('/')[0].split('?')[0]
                    if name_part and len(name_part) > 2:
                        structured_data["name"] = name_part.replace('-', ' ').replace('_', ' ').title()
            
            # Look for any technical terms that might be skills
            tech_terms = ["react", "python", "javascript", "java", "node", "aws", "azure", "docker", "kubernetes"]
            found_skills = []
            for term in tech_terms:
                if term.lower() in profile_response.lower():
                    found_skills.append(term.title())
            
            if found_skills:
                structured_data["skills"] = found_skills
                structured_data["technical_skills"] = found_skills
            
            return structured_data
            
        except Exception as e:
            logger.error(f"❌ Error in basic text parsing: {e}")
            return structured_data
    
    def _extract_potential_skills(self, text: str) -> List[str]:
        """Extract potential skills from text."""
        skills = []
        tech_terms = [
            "react", "angular", "vue", "node.js", "python", "java", "javascript", "typescript",
            "aws", "azure", "docker", "kubernetes", "mongodb", "postgresql", "mysql", "redis",
            "graphql", "rest", "api", "git", "ci/cd", "devops", "agile", "scrum", "html", "css"
        ]
        
        for term in tech_terms:
            if term.lower() in text.lower():
                skills.append(term.title())
        
        return skills
    
    def _extract_text_after_header(self, text: str, header: str) -> str:
        """Extract text content after a markdown header."""
        try:
            # Remove the header line
            content = text.replace(header, "").strip()
            # Remove any leading/trailing whitespace and newlines
            content = content.strip()
            return content
        except:
            return ""
    
    def _extract_list_items(self, text: str) -> List[str]:
        """Extract list items from markdown text."""
        items = []
        try:
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('- ') or line.startswith('* '):
                    item = line[2:].strip()
                    if item:
                        items.append(item)
        except:
            pass
        return items
    
    def _extract_companies_from_matches(self, text: str) -> List[str]:
        """Extract company names from job matches section."""
        companies = []
        try:
            lines = text.split('\n')
            for line in lines:
                if "**Company:**" in line:
                    company = line.split("**Company:**")[1].strip()
                    if company:
                        companies.append(company)
        except:
            pass
        return companies
    
    def _calculate_parsing_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence score for the parsed data."""
        score = 0.0
        max_score = 100.0
        
        # Name (20 points)
        if data.get("name") and data["name"] != "Unknown":
            score += 20
        
        # Skills (25 points)
        if data.get("skills") and len(data["skills"]) > 0:
            score += min(25, len(data["skills"]) * 5)
        
        # Summary (20 points)
        if data.get("summary") and len(data["summary"]) > 50:
            score += 20
        
        # Work history (15 points)
        if data.get("work_history") and len(data["work_history"]) > 0:
            score += 15
        
        # Experience (10 points)
        if data.get("experience") and len(data["experience"]) > 20:
            score += 10
        
        # LinkedIn URL (10 points)
        if data.get("linkedin"):
            score += 10
        
        return min(100.0, score)
    
    def generate_embedding_for_profile(self, profile_data: Dict[str, Any]) -> List[float]:
        """Generate embedding for the LinkedIn profile data."""
        try:
            # Create a text representation of the profile
            profile_text = f"""
            Name: {profile_data.get('name', '')}
            Summary: {profile_data.get('summary', '')}
            Skills: {', '.join(profile_data.get('skills', []))}
            Experience: {profile_data.get('experience', '')}
            Work History: {', '.join([f"{job.get('company', '')} - {job.get('title', '')}" for job in profile_data.get('work_history', [])])}
            Education: {profile_data.get('education', '')}
            """
            
            # Generate embedding using the Couchbase connection
            embedding = self.cb_connection.generate_embedding(profile_text)
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Failed to generate embedding: {e}")
            return [0.0] * 1024  # Return zero vector as fallback
    
    def store_profile_in_couchbase(self, profile_data: Dict[str, Any], original_info: Dict[str, str]) -> bool:
        """Store the extracted profile in Couchbase."""
        try:
            # Generate document ID
            doc_id = f"linkedin_{original_info.get('row_index', datetime.now().timestamp())}"
            
            # Generate embedding
            embedding = self.generate_embedding_for_profile(profile_data)
            profile_data["embedding"] = embedding
            
            # Add metadata
            profile_data["_id"] = doc_id
            profile_data["source_file"] = original_info.get("source_file", "")
            profile_data["extraction_method"] = "linkedin_mcp"
            
            # Store in Couchbase
            self.cb_connection.upsert_candidate(doc_id, profile_data)
            
            logger.info(f"✅ Stored profile in Couchbase with ID: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store profile in Couchbase: {e}")
            return False
    
    async def process_linkedin_profiles(self, excel_file_path: str, max_profiles: Optional[int] = None) -> Dict[str, Any]:
        """Process LinkedIn profiles from Excel file and store them in Couchbase."""
        logger.info("🚀 Starting LinkedIn profile processing...")
        
        # Initialize MCP server
        if not await self.initialize_mcp():
            logger.error("❌ Failed to initialize MCP server. Exiting.")
            return {"success": False, "error": "MCP server initialization failed"}
        
        # Load LinkedIn URLs
        urls_data = self.load_linkedin_urls(excel_file_path)
        if not urls_data:
            logger.error("❌ No LinkedIn URLs found to process.")
            return {"success": False, "error": "No LinkedIn URLs found"}
        
        # Limit processing if specified
        if max_profiles:
            urls_data = urls_data[:max_profiles]
            logger.info(f"📊 Processing limited to {max_profiles} profiles")
        
        # Process each profile
        results = {
            "total_profiles": len(urls_data),
            "successful_extractions": 0,
            "failed_extractions": 0,
            "successful_storage": 0,
            "failed_storage": 0,
            "profiles": []
        }
        
        for i, url_info in enumerate(urls_data, 1):
            linkedin_url = url_info["linkedin_url"]
            logger.info(f"📊 Processing profile {i}/{len(urls_data)}: {linkedin_url}")
            print(f"🔄 Processing {i}/{len(urls_data)}: {linkedin_url[:50]}...")
            
            try:
                # Extract profile data with timeout
                logger.info(f"⏱️  Starting profile extraction (timeout: {self.profile_timeout}s)")
                profile_data = await self.extract_linkedin_profile(linkedin_url)
                
                if profile_data:
                    results["successful_extractions"] += 1
                    
                    # Store in Couchbase
                    if self.store_profile_in_couchbase(profile_data, url_info):
                        results["successful_storage"] += 1
                        profile_data["storage_status"] = "success"
                    else:
                        results["failed_storage"] += 1
                        profile_data["storage_status"] = "failed"
                    
                    # Add to results
                    results["profiles"].append({
                        "url": linkedin_url,
                        "name": profile_data.get("name", "Unknown"),
                        "confidence": profile_data.get("parsing_confidence", 0.0),
                        "skills_count": len(profile_data.get("skills", [])),
                        "status": profile_data.get("storage_status", "unknown")
                    })
                    
                else:
                    results["failed_extractions"] += 1
                    results["profiles"].append({
                        "url": linkedin_url,
                        "name": "Failed",
                        "confidence": 0.0,
                        "skills_count": 0,
                        "status": "extraction_failed"
                    })
                
            except Exception as e:
                logger.error(f"❌ Error processing {linkedin_url}: {e}")
                results["failed_extractions"] += 1
                results["profiles"].append({
                    "url": linkedin_url,
                    "name": "Error",
                    "confidence": 0.0,
                    "skills_count": 0,
                    "status": "error"
                })
            
            # Add configurable delay to avoid rate limiting
            await asyncio.sleep(self.delay_between_profiles)
        
        # Log summary
        logger.info("🎉 LinkedIn profile processing completed!")
        logger.info(f"📊 Summary: {results['successful_extractions']}/{results['total_profiles']} extracted, {results['successful_storage']}/{results['total_profiles']} stored")
        
        return results


async def main():
    """Main function to run the LinkedIn extractor."""
    print("🚀 LinkedIn Profile Extractor")
    print("=" * 50)
    
    # Check if Excel file exists
    excel_file = "profiles/linkedin.xlsx"
    if not os.path.exists(excel_file):
        print(f"❌ Excel file not found: {excel_file}")
        print("Please ensure the file exists in the profiles/ directory.")
        return
    
    try:
        # Initialize extractor with custom timeout settings
        extractor = LinkedInExtractor(
            profile_timeout=500,  # 60 seconds timeout for profile extraction
            delay_between_profiles=50 # 5 seconds delay between profiles
        )
        
        print(f"⏱️  Profile extraction timeout: {extractor.profile_timeout} seconds")
        print(f"⏳ Delay between profiles: {extractor.delay_between_profiles} seconds")
        
        # Process profiles
        results = await extractor.process_linkedin_profiles(excel_file, max_profiles=5)  # Start with 5 for testing
        
        # Display results
        print("\n📋 Processing Results:")
        print("=" * 50)
        print(f"Total profiles: {results['total_profiles']}")
        print(f"Successful extractions: {results['successful_extractions']}")
        print(f"Failed extractions: {results['failed_extractions']}")
        print(f"Successful storage: {results['successful_storage']}")
        print(f"Failed storage: {results['failed_storage']}")
        
        if results['profiles']:
            print("\n📊 Individual Profile Results:")
            for profile in results['profiles']:
                status_emoji = "✅" if profile['status'] == 'success' else "❌"
                print(f"{status_emoji} {profile['name']} - Confidence: {profile['confidence']:.1f}% - Skills: {profile['skills_count']}")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        logger.error(f"Main execution error: {e}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 
 