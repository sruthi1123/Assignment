
# Home Loan Application Chatbot

This is a smart Streamlit-based chatbot designed to collect all relevant data for a home loan application through a dynamic, multi-turn conversation. The system guides users by asking only the missing or incomplete details, adapting its prompts based on previous responses. It uses LangChain, LangGraph, and a local LLM model served via Ollama to intelligently extract structured fields from natural language input.

---

## Features

- Conversational form filling for home loan application
- Multi-level, multi-branch dynamic flow
- Adapt prompts based on user inputs
- Intelligent field extraction using multiple LLM chains:
  - Employment & income details (salaried/business)
  - Property type and values (new/resale)
  - Credit score and default history
- Partial input handling across all agents
- Live chat history tracking in the sidebar
- Final structured summary of the collected loan application details
---

## Tech Stack

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Ollama](https://ollama.ai/) running locally (`llama3`)
- Python 3.11+

---

## 📦 Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/sruthi1123/Assignment.git
   cd loan-chatbot
   ```

2. **Install Dependencies**
   Create and activate a virtual environment (optional), then install:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Ollama**
   Make sure you have [Ollama installed](https://ollama.com/download) and run:
   ```bash
   ollama run llama3
   ```

4. **Start the Streamlit App**
   ```bash
   streamlit run App.py
   ```

---

## Example Prompts to Use

- "I’m working at Infosys earning 10L per annum."
- "The property is new from Prestige group, costs 1.2 Cr."
- "Credit score is 750, no defaults in last year."

---


