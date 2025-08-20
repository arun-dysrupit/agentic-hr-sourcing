1. Create a .env file in the root directory and add the following variables:

# Couchbase Capella Configuration
CB_CONNECTION_STRING=
CB_USERNAME=
CB_PASSWORD=
CB_BUCKET=
CB_COLLECTION=
CB_SCOPE=
CB_SEARCH_INDEX=

# Bright Data MCP 
BRIGHT_DATA_API_KEY=
BROWSER_AUTH=

# Nebius AI API KEY 
NEBIUS_API_BASE=
NEBIUS_API_KEY=


2. Run the following command to install the dependencies:

```bash
pip install -r requirements.txt
```

3. Create a folder called "resumes" in the root directory and put the resumes in it. In order to ingest the resumes, you need to run the following command:

```bash
python ingest_resumes.py
```

4. In order to run the HR agent, you need to run the following command:

```bash
python hr_app.py
```

In the HR agent, you can upload a job description and the resumes will be searched for the best matches. There are two options: 

1. Pure vector search
2. Hybrid search (vector search + filters) 

5. First ensure you have a spreadsheet with the Linkedin URLs kept in the profiles folder. In order to do Linkedin scraping, you need to run the following command:

```bash
python linkedin_extractor.py
```

6. In order to do job description analysis, you need to run the following command:

```bash