# agentic-hr-sourcing

 **Implementation Plan**
 
---

## **📥 Phase 1: Ingestion from BambooHR + LinkedIn**

### 🟦 **Source A: BambooHR Zip File**

#### Tasks:

1. **CSV Parsing Agent**

   * Parse CSV inside the BambooHR ZIP archive.
   * Extract resume URLs and metadata.
2. **Resume Downloader**

   * Programmatically download each PDF using links from the CSV.
3. **Resume Parser Agent**

   * Extract candidate name, contact, skills, education, work history.
   * Generate embeddings using LLM service (OpenAI or Capella AI).
4. **Store in Couchbase**

   * Save parsed structured data + raw resume in unified collection `candidates`.
   * Add metadata like `source: "bamboohr"`.

---

### 🟩 **Source B: LinkedIn via HR Search + BrightData MCP**

#### Inputs:

* Skills (entered by HR)
* Location (entered by HR)

#### Tasks:

1. **LinkedIn Search Trigger**

   * Input form for HR to specify skillset + location.
   * Construct LinkedIn search query using standard search URL structure.
2. **Candidate URL Collector**

   * Extract LinkedIn profile URLs from search results (use MCP if needed here too).
3. **BrightData MCP Scraper**

   * Scrape each candidate’s LinkedIn profile using MCP:

     * Name, Role, Experience, Skills, Location, Education
   * Parse unstructured data into structured JSON.
4. **Generate Embeddings + Store in Couchbase**

   * Embed relevant fields using LLM services.
   * Store in same `candidates` collection with metadata like `source: "linkedin"`.

---

## ✅ **📊 Phase 2: Unified Couchbase Collection Design**

### 🔸Collection: `candidates`

Each document schema (sample):

```json
{
  "source": "bamboohr" | "linkedin",
  "name": "Jane Doe",
  "email": "jane@example.com",
  "location": "San Francisco",
  "skills": ["React", "Node.js", "AWS"],
  "experience": "...",
  "education": "...",
  "resume_url": "...",
  "linkedin_url": "...",
  "vector_embedding": [0.1, 0.24, ...],
}
```
---

## ✅ **🔁 Phase 3: Agent Pipeline Using LangGraph**

Revised agent flow

1. **Job Posted**
2. **Job Analysis Agent**

   * Extract skills → send to sourcing agents.
3. **Candidate Sourcing Agent (2-pronged)**

   * \[BambooHR] → Resume URLs → Parsing Agent
   * \[LinkedIn] → URLs → MCP Scraper Agent
4. **Resume Parsing Agent**

   * Extract candidate data
   * Generate embeddings
5. **Matching Agent**

   * Compare against job vectors using Couchbase Vector Search
6. **Decision Router**

   * Score candidates → Auto-approve or human review

7. All of this will be enclosed in a Streamlit UI where the HR can manually add a new job description or upload a pdf containing the JD and we perform vector search to share the most relevant user profiles for the given JD.

8. HR can also tweak criteria for JD to tweak the search results. 

---

## ✅ **🧠 Phase 4: LLM + Vector Search**

### Two Options 

#### 🅰️ Option A: External LLM Stack

* LangChain → OpenAI / Azure OpenAI → Couchbase Vector Search → Couchbase Storage

#### 🅱️ Option B: Capella AI

* LangChain → Capella AI Model Service → Couchbase Vector Search → Couchbase Storage

> Use Capella AI to reduce latency & infra complexity where possible.

---
