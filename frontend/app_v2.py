import streamlit as st
import os
from pypdf import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import re
import numpy as np
from faster_whisper import WhisperModel
import io

# --- Whisper Model Initialization ---
# Use st.cache_resource to load the model only once and keep it in memory.
@st.cache_resource
def get_whisper_model():
    """Loads the Whisper model into memory."""
    # Using the "base" model as requested for efficiency.
    return WhisperModel("base", device="cpu", compute_type="int8")


model = get_whisper_model()

# --- Helper Functions ---
def load_document(uploaded_file):
    """Loads text from an uploaded file (TXT or PDF)."""
    text = ""
    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension == ".txt":
            text = uploaded_file.read().decode("utf-8")
        elif file_extension == ".pdf":
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return None
    return text

def generate_question(llm, memory, interview_prompt, resume, job_description, human_input):
    """Generates the next interview question."""
    chat_history = memory.buffer
    formatted_prompt = interview_prompt.format(
        resume=resume,
        job_description=job_description,
        chat_history=chat_history,
        human_input=human_input
    )
    response = llm.invoke(formatted_prompt)
    return response.content

def calculate_score(llm, resume_text, jd_text, conversation_history):
    """Calculates the final score and assessment."""
    scoring_template = """You are an AI evaluating a candidate's interview performance.
Given the resume, job description, and the full interview conversation history,
provide a qualitative assessment of the candidate's suitability for the role.
Consider the following:
- How well the candidate's skills and experience align with the job requirements.
- The relevance and depth of the candidate's answers.
- How effectively the candidate addressed the interviewer's questions.
- Overall communication clarity and confidence.

Provide a brief summary of their strengths and weaknesses based on the interview,
and a concluding assessment of their fit for the position.
Additionally, provide a numerical score from 1 to 10 (where 1 is poor and 10 is excellent)
at the very beginning of your response, formatted as "SCORE: [number]/10".

Resume:
{resume}

Job Description:
{job_description}

Conversation History:
{history}

Candidate Assessment:
"""
    scoring_prompt = PromptTemplate(
        input_variables=["resume", "job_description", "history"],
        template=scoring_template,
    )
    chain = (
        {"resume": RunnablePassthrough(), "job_description": RunnablePassthrough(), "history": RunnablePassthrough()}
        | scoring_prompt
        | llm
        | StrOutputParser()
    )
    full_assessment_text = chain.invoke({
        "resume": resume_text,
        "job_description": jd_text,
        "history": conversation_history
    })
    score_match = re.search(r"SCORE:\s*(\d+)/10", full_assessment_text)
    score = int(score_match.group(1)) if score_match else None
    assessment_text = full_assessment_text
    if score_match:
        assessment_text = re.sub(r"SCORE:\s*\d+/10\s*", "", full_assessment_text, 1).strip()
    return score, assessment_text

# --- Transcription Function ---
def transcribe_audio(audio_file_input):
    """
    Transcribes an audio file input using the pre-loaded Whisper model.
    """
    if not audio_file_input:
        return ""
    
    # Read the bytes from the uploaded file object
    audio_bytes = audio_file_input.read()
    
    # Wrap the bytes in a buffer
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "temp_audio.wav" # Whisper needs a file extension.

    # Transcribe using faster-whisper
    segments, _ = model.transcribe(
        audio_file,
        language="en",
        beam_size=1,
    )
    
    transcript = " ".join(segment.text for segment in segments)
    return transcript

# --- Streamlit App ---

st.set_page_config(page_title="AI Interviewer", layout="wide")

st.title("AI-Powered Technical Interviewer")
st.write("Upload a resume and job description, to start the interview")

# --- API Key Input ---
if "GOOGLE_API_KEY" not in os.environ:
    api_key = st.text_input("Enter your Google API Key:", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    else:
        st.warning("Please enter your Google API Key to proceed.")
        st.stop()

# --- Initialization ---
try:
    # Using gemini-1.5-flash for a balance of speed and capability
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
except Exception as e:
    st.error(f"Failed to initialize the language model. Please check your API key. Error: {e}")
    st.stop()

interview_prompt_template = """
You are an expert AI Interviewer for a high-stakes technical interview. Your role is to assess the candidate’s qualifications, skills, and thinking based on the job description and resume.

**Strictly follow these rules:**
1. Ask *one* clear, concise, and relevant interview question at a time.
2. Use the resume to probe deeply into specific experiences or skills the candidate claims.
3. Use the job description to test alignment and required competencies.
4. Use chat history to ask meaningful follow-ups and avoid repeating topics.
5. Critically evaluate the candidate’s latest response (`Candidate Response`) and demand specificity, reasoning, or examples when appropriate.

Avoid small talk. Be professional and direct.

---
**Resume:**
{resume}

**Job Description:**
{job_description}

**Conversation So Far:**
{chat_history}

**Candidate Response:**
{human_input}
---

Based on all the above, ask your next best interview question or follow-up that rigorously tests their skills, depth of knowledge, or experience:
"""
interview_prompt = PromptTemplate(
    input_variables=["resume", "job_description", "chat_history", "human_input"],
    template=interview_prompt_template,
)

# --- State Management ---
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'interview_started' not in st.session_state:
    st.session_state.interview_started = False
if 'interview_finished' not in st.session_state:
    st.session_state.interview_finished = False
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = ""
if 'jd_text' not in st.session_state:
    st.session_state.jd_text = ""
# --- ADDED: State for managing the transcript review process ---
if 'transcript' not in st.session_state:
    st.session_state.transcript = None

# --- UI Layout ---
col1, col2 = st.columns(2)

with col1:
    st.header("Inputs")
    resume_file = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf", "txt"])
    jd_file = st.file_uploader("Upload Job Description (PDF or TXT)", type=["pdf", "txt"])

    if resume_file and not st.session_state.resume_text:
        st.session_state.resume_text = load_document(resume_file)
        if st.session_state.resume_text:
            st.success("Resume loaded successfully!")
    if jd_file and not st.session_state.jd_text:
        st.session_state.jd_text = load_document(jd_file)
        if st.session_state.jd_text:
            st.success("Job Description loaded successfully!")

    if st.session_state.resume_text:
        with st.expander("View Resume"):
            st.text(st.session_state.resume_text)
    if st.session_state.jd_text:
        with st.expander("View Job Description"):
            st.text(st.session_state.jd_text)

    start_button = st.button("Start Interview", disabled=not (st.session_state.resume_text and st.session_state.jd_text))

if start_button and not st.session_state.interview_started:
    st.session_state.interview_started = True
    with st.spinner("AI is preparing the first question..."):
        ai_question = generate_question(llm, st.session_state.memory, interview_prompt, st.session_state.resume_text, st.session_state.jd_text, "The candidate is ready. Please begin the interview.")
        st.session_state.chat_history.append(("AI", ai_question))
        st.session_state.memory.save_context({"input": ""}, {"output": ai_question})
    st.rerun()

with col2:
    st.header("Interview Chat")

    if st.session_state.interview_started and not st.session_state.interview_finished:
        for role, text in st.session_state.chat_history:
            with st.chat_message("assistant" if role == "AI" else "user"):
                st.write(text)

        # --- MODIFIED: Two-step process for audio input and submission ---
        st.write("---")
        
        # If there's no pending transcript to review, show the recorder.
        if st.session_state.transcript is None:
            audio_input = st.audio_input("Record your answer:", key="audio_input")
            if audio_input:
                with st.spinner("Transcribing your answer..."):
                    transcript = transcribe_audio(audio_input)
                    st.session_state.transcript = transcript # Store transcript for review
                st.rerun() # Rerun to show the review UI
        
        # If there IS a pending transcript, show the review UI.
        else:
            st.subheader("Review your answer")
            # Display the transcript in an editable text area
            edited_transcript = st.text_area(
                "Edit your transcribed answer below, then submit.", 
                value=st.session_state.transcript,
                height=150
            )

            if st.button("Submit Answer"):
                final_transcript = edited_transcript
                if final_transcript:
                    # Add the user's final answer to history
                    st.session_state.chat_history.append(("Human", final_transcript))
                    
                    # Generate the AI's next question
                    with st.spinner("AI is thinking..."):
                        ai_question = generate_question(
                            llm, 
                            st.session_state.memory, 
                            interview_prompt, 
                            st.session_state.resume_text, 
                            st.session_state.jd_text, 
                            final_transcript
                        )
                    
                    st.session_state.chat_history.append(("AI", ai_question))
                    st.session_state.memory.save_context({"input": final_transcript}, {"output": ai_question})
                    
                    # --- CRITICAL FIX: Clear the transcript to prevent loop and show recorder again ---
                    st.session_state.transcript = None
                    
                    st.rerun()
                else:
                    st.warning("Your answer cannot be empty.")

        st.write("---")
        if st.button("End Interview Manually", key="end_interview"):
            st.session_state.interview_finished = True
            st.rerun()

    elif st.session_state.interview_finished:
        st.info("Interview has ended. Generating your performance report...")
        full_conversation = "\n".join([f"{role}: {text}" for role, text in st.session_state.chat_history])
        
        with st.spinner("Calculating score and generating feedback..."):
            score, assessment = calculate_score(llm, st.session_state.resume_text, st.session_state.jd_text, full_conversation)

        st.header("Interview Report")
        if score is not None:
            st.subheader(f"Overall Score: {score}/10")
        else:
            st.warning("Could not extract a numerical score.")
        
        st.subheader("Qualitative Assessment")
        st.write(assessment)

        if st.button("Start New Interview"):
            # Reset state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    else:
        st.info("Please upload a resume and job description and click 'Start Interview'.")
