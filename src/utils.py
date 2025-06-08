from llm import get_llm_output
import pymupdf

def handle_upload(cv_files, jd_files):
    """Main handler: extract, index, and feedback."""
    if not cv_files or not jd_files:
        return "❌ Please upload both CV and Job Description PDFs."
    # Extract texts
    cv_text = extract_text_from_pdfs(cv_files)
    jd_text = extract_text_from_pdfs(jd_files)
    # # Index into RAG vector DB
    # chunk_and_index(cv_text, "cv_collection")
    # chunk_and_index(jd_text, "jd_collection")
    # Generate feedback
    feedback = generate_feedback(cv_text, jd_text)
    return feedback

def extract_text_from_pdfs(files):
    """Extract full text from a list of PDF files."""
    if not files:
        return "No files provided."
    if not isinstance(files, list):
        files = [files]
    text = ""
    for f in files or []:
        with pymupdf.open(f) as pdf:
            text += "".join(page.get_text() for page in pdf)
    return text

# function to structure CV text into a clean format
def structure_cv(cv_text):
    system_prompt = """
    You are a professional resume/CV formatter. 
    Reformat the candidate's resume text into a clean, modern structure with these sections. Leave 'no information' for any section that is not applicable:
    1. Header: Name, Contact Info(email, phone, LinkedIn)
    2. Professional Summary: 2–3 sentence career overview of the candidate's background and goals
    3. Experience: List of jobs with role, company, dates, bullet achievements
    4. Education: Degrees, institutions, years
    5. Projects (if any): title, description, technologies used
    6. Skills: grouped by category (e.g., Languages, Tools, Frameworks)
    7. Certifications (if any)

    Output the resume in Markdown format. Use consistent bullet points and headers.
    """
    # Chia nhỏ nếu CV quá dài (ở đây giả định cv_text < 4000 tokens)
    prompt = f"""
    Raw Resume Text:
    \"\"\"
    {cv_text}
    \"\"\"

    Please reformat as described.
    """
    return get_llm_output(
        temperature=0.3,
        max_tokens=1500,
        system_role=system_prompt,
        prompt=prompt
    )

# function to structure job description text into a clean format
def structure_jd(jd_text):
    system_prompt = """
You are a professional job description formatter. 
Reformat the candidate's job description text into a clean, modern structure with these sections. Leave 'no information' for any section that is not applicable:
1. Header: Job Title, Company Name
2. Summary: 2–3 sentence overview of the role
3. Responsibilities: List of key responsibilities and tasks
4. Requirements: Skills and qualifications needed
5. Preferred Qualifications: Additional skills that are a plus
6. Benefits: Perks and benefits offered
7. Application Process: How to apply

Output the job description in Markdown format. Use consistent bullet points and headers.
"""
    # Chia nhỏ nếu JD quá dài (ở đây giả định jd_text < 4000 tokens)
    prompt = f"""
Raw Job Description Text:
\"\"\"
{jd_text}
\"\"\"

Please reformat as described.
"""
    return get_llm_output(
        temperature=0.3,
        max_tokens=1200,
        system_role=system_prompt,
        prompt=prompt
    )

# Generate feedback on how to improve the CV based on the job description
def generate_feedback(cv_text, jd_text):
    """
    Use LLM to generate structured, comprehensive feedback on the resume:
    - Đưa ra các tiêu chí đánh giá (Assessment Criteria)
    - Đánh giá từng mục (Review) dựa trên tiêu chí
    - Gợi ý cải tiến (Improvement Suggestions) để tăng tỉ lệ đậu
    """
    system_prompt = """
You are a senior career advisor and HR specialist. 
Your task is to evaluate the candidate’s resume against the Job Description, 
using the following structured framework:

1. Assessment Criteria:
   a. Relevance of Experience – How well past roles map to required responsibilities.
   b. Skills Match – Presence and prominence of key technical and soft skills.
   c. Projects - Relevance and impact of projects related to the JD.
   d. Achievement Impact – Use of quantifiable results and action verbs.
   e. Keywords & ATS Optimization – Inclusion of terminology from the JD.
   f. Clarity & Readability – Logical sectioning, bullet usage, concise language.
   g. Professional Branding – Strong summary, consistent formatting, contact info.

2. For each criterion:
   - Provide a **brief review**: highlight strengths and gaps.
   - Offer **2–3 short actionable suggestions** to improve that aspect of the resume.

3. Finally, summarize **top 3 priority actions** the candidate should take 
   to significantly increase their chance of being shortlisted.

Return your output in Markdown, with sections:
## Assessment
- **Relevance of Experience**  
  - Review: …  
  - Suggestions:
    1. …
    2. …

(Repeat for each criterion)

## Top 3 Priority Actions
1. …
2. …
3. …
"""

    user_prompt = f"""
Job Description:
\"\"\"
{jd_text}
\"\"\"

Candidate Resume:
\"\"\"
{cv_text}
\"\"\"
"""

    return get_llm_output(
        temperature=0.3,
        max_tokens=1200,
        system_role=system_prompt,
        prompt=user_prompt
    )