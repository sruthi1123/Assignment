import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Dict, Any
import re
import json
import pdfkit
import base64
import tempfile

# Define LangGraph state
class AppState(TypedDict, total=False):
    user_input: str
    city: Optional[str]
    age: Optional[str]
    marital_status: Optional[str]
    email: Optional[str]
    mobile: Optional[str]
    employment: Optional[str]
    employer: Optional[str]
    income: Optional[float]
    mode: Optional[str]
    company_name: Optional[str]
    turnover: Optional[str]
    profit: Optional[str]
    property_type: Optional[str]
    property: Optional[Dict[str, Any]]
    credit_score: Optional[int]
    has_defaults: Optional[bool]
    default_within_12_months: Optional[bool]
    filtered_lenders: Optional[list]
    final_offer: Optional[Dict[str, Any]]

def extract_json(raw):
    try:
        # Step 1: If the response is a dict with "text" field, extract from it
        if isinstance(raw, dict) and "text" in raw:
            raw = raw["text"]

        # Step 2: Extract the first JSON-looking object from raw text
        match = re.search(r'{[\s\S]*?}', raw)
        if match:
            json_text = match.group(0)
            parsed = json.loads(json_text)

            # Normalize keys
            normalized = {}
            for k, v in parsed.items():
                key = k.replace(" ", "_").replace("-", "_").lower()
                if key == "propertytype":
                    key = "property_type"
                normalized[key] = v
            return normalized
    except Exception as e:
        print("Failed to extract JSON:", e)

    return {}


# LLM setup
llm = ChatOpenAI(openai_api_base="http://localhost:11434/v1",
                 openai_api_key="ollama",
                 model="llama3",
                 temperature=0)

def get_llm_chain(template_text):
    prompt = PromptTemplate(input_variables=["text"], template=template_text)
    return LLMChain(llm=llm, prompt=prompt)

# LLM Chains
personal_info_chain = get_llm_chain("""
Extract the following details from the text as JSON:
- city (required)
- age (optional)
- marital_status (optional)
- email (optional)
- mobile (optional)

Return ONLY a valid JSON like:
{{
  "city": "Hyderabad",
  "age": "29",
  "marital_status": "single",
  "email": "abc@gmail.com",
  "mobile": "9876543210"
}}

Text: {text}
""")

income_type_chain = get_llm_chain('''From the following text, extract employment type ONLY if clearly and explicitly mentioned.

Return JSON like:
{{"employment": "salaried"}}
OR
{{"employment": "business"}}

If employment type is not explicitly stated, return {{}}

Do not infer based on company names or job titles. Do not guess.

Text: {text}
''')

salaried_chain = get_llm_chain('''From the following text, extract only the fields that are **clearly present**:
- employer
- income (as a single numeric value where L is lakh and Cr is crore)
- mode (monthly or fixed+variable)

Return a JSON with only the found fields. Do not make up values. Do not guess income based on company names. Donot guess mode until explicitly stated.

If any field is missing, just skip that field in the output.

Text: {text}
''')
business_chain = get_llm_chain('''From the following text, extract business-related details if clearly mentioned:
- company_name
- turnover (as text)
- profit (as text or number)

Return a JSON with only available values. Do not guess or estimate.

Skip fields that are not present in the text.

Text: {text}
''')
property_type_chain = get_llm_chain("Extract property type (new/resale) as JSON from: {text}")
new_property_chain = get_llm_chain('''Extract builder_name and market_value from the text ONLY IF explicitly stated.

Return JSON like:
{{"builder_name": "MyHome Constructions", "market_value": "80L"}}

Skip any field that is not directly stated. Do NOT make assumptions, estimate, or generate values based on common knowledge.

If none are found, return {{}}

Text: {text}
''')
resale_property_chain = get_llm_chain('''Extract previous_owner, age_of_property, market_value as JSON from: {text}

Return a JSON like:
{{"previous_owner": "Mr. Reddy", "age_of_property": "10 years", "market_value": "65L"}}

Do not guess or make assumptions. Skip any field not found in the input.
''')
credit_chain = get_llm_chain('''Extract ONLY the following from the text if they are **clearly mentioned**:
- credit_score (as a number)
- has_defaults (true/false)
- default_within_12_months (true/false)

Return JSON like:
{{"credit_score": 720, "has_defaults": false, "default_within_12_months": false}}

Do NOT guess or hallucinate. If fields are missing or ambiguous, leave them out.

If nothing is present, return {{}}

Text: {text}
''')


llm_agents = {
    "PersonalInfoAgent": personal_info_chain,
    "IncomeTypeAgent": income_type_chain,
    "SalariedAgent": salaried_chain,
    "BusinessAgent": business_chain,
    "PropertyTypeAgent": property_type_chain,
    "NewPropertyAgent": new_property_chain,
    "ResalePropertyAgent": resale_property_chain,
    "CreditAgent": credit_chain
}

def run_smart_agents(state):
    text = state.get("user_input", "")
    updated = False

    # Run IncomeTypeAgent early
    result = llm_agents["IncomeTypeAgent"].run({"text": text})
    #print("[IncomeTypeAgent]", result)
    parsed = extract_json(result)
    if parsed.get("employment") and not state.get("employment"):
        state["employment"] = parsed["employment"]
        updated = True

    # Run SalariedAgent if applicable
    if state.get("employment") == "salaried":
        result = llm_agents["SalariedAgent"].run({"text": text})
        #st.write("[SalariedAgent]", result)
        parsed = extract_json(result)

        # Handle income formatting
        income = parsed.get("income")
        if isinstance(income, str):
            try:
                parsed["income"] = float(income.replace(",", "").replace("L", "00000").strip())
            except:
                parsed["income"] = None

        for key in ["employer", "income", "mode"]:
            if parsed.get(key) and not state.get(key):
                state[key] = parsed[key]
                updated = True

    # Run BusinessAgent if applicable
    if state.get("employment") == "business":
        result = llm_agents["BusinessAgent"].run({"text": text})
        #st.write("[BusinessAgent]", result)
        parsed = extract_json(result)
        for key in ["company_name", "turnover", "profit"]:
            if parsed.get(key) and not state.get(key):
                state[key] = parsed[key]
                updated = True

    # Always run PersonalInfoAgent
    cleaned_text = text.replace("\n", " ").strip()
    result = llm_agents["PersonalInfoAgent"].invoke({"text": cleaned_text})

    #st.write("[PersonalInfoAgent]", result)
    # Optional for debugging
    #print("Raw LLM output (PersonalInfoAgent):", result)

    parsed = extract_json(result)

    # Required: city
    if parsed.get("city") and not state.get("city"):
        state["city"] = parsed["city"]
        updated = True

    # Optional: age, marital_status, email, mobile
    for key in ["age", "marital_status", "email", "mobile"]:
        if not state.get(key):
            value = parsed.get(key, "no mention")
            state[key] = value
            updated = True


    # Run PropertyTypeAgent
    result = llm_agents["PropertyTypeAgent"].run({"text": text})
    #st.write("[PropertyTypeAgent]", result)
    parsed = extract_json(result)

    # Fix 1: if list is returned, take first item
    if isinstance(parsed, list) and len(parsed) > 0:
        parsed = parsed[0]

    # Fix 2: normalize keys
    key_mapping = {
        "Property Type": "property_type"
    }
    parsed = {key_mapping.get(k, k.lower()): v for k, v in parsed.items()}

    if parsed:
        if "property_type" in parsed and not state.get("property_type"):
            prop_type = parsed["property_type"]
            if isinstance(prop_type, str):
                state["property_type"] = prop_type.lower()
                updated = True

        # Save property subfields if not already present
        property_fields = ["builder_name", "market_value", "previous_owner", "age_of_property"]
        property_data = state.get("property", {}) or {}
        for key in property_fields:
            if parsed.get(key) and not property_data.get(key):
                property_data[key] = parsed[key]
                updated = True
        if property_data:
            state["property"] = property_data

    # 🔁 Trigger NewPropertyAgent or ResalePropertyAgent if property_type is set
    if state.get("property_type") == "new":
        result = llm_agents["NewPropertyAgent"].run({"text": text})
        #st.write("[NewPropertyAgent]", result)
        parsed = extract_json(result)
        if parsed:
            if "property" not in state or not isinstance(state["property"], dict):
                state["property"] = {}
            for key in ["builder_name", "market_value"]:
                if parsed.get(key) and not state["property"].get(key):
                    state["property"][key] = parsed[key]
                    updated = True

    elif state.get("property_type") == "resale":
        result = llm_agents["ResalePropertyAgent"].run({"text": text})
        #st.write("[ResalePropertyAgent]", result)
        parsed = extract_json(result)
        if parsed:
            if "property" not in state or not isinstance(state["property"], dict):
                state["property"] = {}
            for key in ["previous_owner", "age_of_property", "market_value"]:
                if parsed.get(key) and not state["property"].get(key):
                    state["property"][key] = parsed[key]
                    updated = True

    # CreditAgent for credit info (partial updates handled)
    result = llm_agents["CreditAgent"].run({"text": text})
    #st.write("[CreditAgent]", result)
    parsed = extract_json(result)
    for key in ["credit_score", "has_defaults", "default_within_12_months"]:
        if key in parsed and parsed[key] is not None and state.get(key) is None:
            state[key] = parsed[key]
            updated = True

    # ✅ Return updated state at the end
    return state or {}


def get_missing_prompt(state):
    if not state.get("city"):
        return "Which city are you planning to buy the property in?"

    if not state.get("employment"):
        return "Are you salaried or a business owner?"

    if state.get("employment") == "salaried":
        missing = []
        if not state.get("employer"): missing.append("employer")
        if not state.get("income"): missing.append("annual income")
        if not state.get("mode"): missing.append("income mode (monthly/fixed+variable)")
        if missing: return f"Please provide: {', '.join(missing)}."

    if state.get("employment") == "business":
        missing = []
        if not state.get("company_name"): missing.append("company name")
        if not state.get("turnover"): missing.append("turnover")
        if not state.get("profit"): missing.append("profit")
        if missing: return f"Please provide: {', '.join(missing)}."

    if not state.get("property_type"):
        return "Is the property new or resale?"

    prop = state.get("property", {})
    if state.get("property_type") == "new":
        if not prop.get("builder_name") or not prop.get("market_value"):
            return "Please provide builder name and market value."
    elif state.get("property_type") == "resale":
        missing = []
        if not prop.get("previous_owner"): missing.append("previous owner")
        if not prop.get("age_of_property"): missing.append("age of property")
        if not prop.get("market_value"): missing.append("market value")
        if missing: return f"Please provide: {', '.join(missing)} for resale property."

    # Credit subfield checks
    has_score = state.get("credit_score") is not None
    has_defaults = state.get("has_defaults") is not None
    has_recent_defaults = state.get("default_within_12_months") is not None

    if not has_score and not has_defaults and not has_recent_defaults:
        return "What is your credit score and any defaults in past year?"

    credit_missing = []
    if not has_score:
        credit_missing.append("credit score")
    if not has_defaults:
        credit_missing.append("any past defaults")
    if not has_recent_defaults:
        credit_missing.append("defaults in the past 12 months")

    if credit_missing:
        return f"Please provide: {', '.join(credit_missing)}."


    return None

# Streamlit UI
st.set_page_config("Home Loan Chatbot")
st.title("🏡 Home Loan Application Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "app_state" not in st.session_state:
    st.session_state.app_state = {}

user_message = st.chat_input("Describe your job, property, or credit details to continue...")

if user_message:
    st.session_state.chat_history.append(("user", user_message))
    state = st.session_state.app_state
    state["user_input"] = user_message
    updated_state = run_smart_agents(state)
    prompt = get_missing_prompt(updated_state)

    if not prompt:
        updated_state["filtered_lenders"] = ["HDFC", "ICICI"]
        updated_state["final_offer"] = {
            "amount": 4000000,
            "emi": 42000,
            "roi": 8.1,
            "tenure": 20
        }
        offer = updated_state["final_offer"]
        response = f"\U0001f389 Here's your loan offer:\n\n- Lenders: {', '.join(updated_state['filtered_lenders'])}\n- Amount: ₹{offer['amount']}\n- EMI: ₹{offer['emi']}/month\n- Rate of Interest: {offer['roi']}%\n- Tenure: {offer['tenure']} years"
    else:
        response = prompt

    st.session_state.app_state = updated_state
    st.session_state.chat_history.append(("bot", response))

# Chat history
for speaker, msg in st.session_state.chat_history:
    with st.chat_message("🧑" if speaker == "user" else "🤖"):
        st.markdown(msg)

with st.sidebar:
    st.markdown("## 📄Loan Application Overview")

    state = st.session_state.app_state

    st.subheader("👤 Personal Info")
    st.write("**City:**", state.get("city", "Not provided"))
    st.write("Age:", state.get("age"))
    st.write("Marital Status:", state.get("marital_status"))
    st.write("Email:", state.get("email"))
    st.write("Mobile:", state.get("mobile"))

    st.write("**Employment Type:**", state.get("employment", "Not provided"))

    if state.get("employment") == "salaried":
        st.write("**Employer:**", state.get("employer", "Not provided"))
        st.write("**Income:**", f"₹{state.get('income'):,}" if state.get("income") else "Not provided")
        st.write("**Mode:**", state.get("mode", "Not provided"))
    elif state.get("employment") == "business":
        st.write("**Company Name:**", state.get("company_name", "Not provided"))
        st.write("**Turnover:**", state.get("turnover", "Not provided"))
        st.write("**Profit:**", state.get("profit", "Not provided"))

    st.subheader("🏠 Property Info")
    st.write("**Property Type:**", state.get("property_type", "Not provided"))
    property_info = state.get("property", {})
    for key, label in {
        "builder_name": "Builder Name",
        "market_value": "Market Value",
        "previous_owner": "Previous Owner",
        "age_of_property": "Age of Property"
    }.items():
        if property_info.get(key):
            st.write(f"**{label}:**", property_info.get(key))

    st.subheader("💳 Credit Info")
    st.write("**Credit Score:**", state.get("credit_score", "Not provided"))
    st.write("**Has Defaults:**", "Yes" if state.get("has_defaults") else "No" if state.get("has_defaults") is not None else "Not provided")
    st.write("**Defaults in Last 12 Months:**", "Yes" if state.get("default_within_12_months") else "No" if state.get("default_within_12_months") is not None else "Not provided")

    if state.get("final_offer"):
        st.subheader("🎯 Final Loan Offer")
        offer = state["final_offer"]
        st.write("**Lenders:**", ", ".join(state.get("filtered_lenders", [])))
        st.write("**Loan Amount:**", f"₹{offer.get('amount', 0):,}")
        st.write("**EMI:**", f"₹{offer.get('emi', 0):,}/month")
        st.write("**Interest Rate:**", f"{offer.get('roi', 0)}%")
        st.write("**Tenure:**", f"{offer.get('tenure', 0)} years")
