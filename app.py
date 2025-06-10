import streamlit as st
import toml
from openai import OpenAI
import requests
import google.generativeai as genai
import anthropic
from pathlib import Path
import os
import re
import pandas as pd
import io
import time
from typing import Optional, Dict, List, Tuple


# Load secrets from either Streamlit Cloud or local `.streamlit/secrets.toml`
def get_api_keys():
    try:
        return st.secrets["api_keys"]
    except Exception:
        try:
            return toml.load(Path(".streamlit/secrets.toml"))["api_keys"]
        except Exception as e:
            st.error(f"Error loading API keys: {str(e)}")
            return {}


api_keys = get_api_keys()


# Helper function to estimate tokens (rough approximation)
def estimate_tokens(text):
    """Rough token estimation: ~4 characters per token for most languages"""
    return len(text) // 4


# Helper function to create model-specific prompts based on token limits
def create_model_prompt(base_prompt, full_text, vocabulary_terms, vocab_access_method, model_name):
    """Create model-specific prompts respecting token limits - prioritizes vocabulary over text length"""

    # Token limits - leave room for response
    model_limits = {
        "gpt-4": 7192,
        "claude": 200000,
        "gemini": 1048576,
        "deepseek": 30768
    }

    # Response token reservations
    response_reserves = {
        "gpt-4": 2000,
        "claude": 4000,
        "gemini": 4000,
        "deepseek": 2000
    }

    # Determine model type and limits
    if "gpt" in model_name.lower() or model_name == "chatgpt":
        limit = model_limits["gpt-4"]
        response_reserve = response_reserves["gpt-4"]
    elif "claude" in model_name.lower():
        limit = model_limits["claude"]
        response_reserve = response_reserves["claude"]
    elif "gemini" in model_name.lower():
        limit = model_limits["gemini"]
        response_reserve = response_reserves["gemini"]
    elif "deepseek" in model_name.lower():
        limit = model_limits["deepseek"]
        response_reserve = response_reserves["deepseek"]
    else:
        limit = 8192
        response_reserve = 2000

    # Base instruction (same for all models)
    base_instruction = """You are a subject indexer specializing in LGBTQI+ literature analysis. Your task is to analyze the provided literary work and suggest relevant subject terms from the QueerLit controlled vocabulary.

Please analyze ONLY the literary text provided (ignore any metadata) and:
1. Identify specific LGBTQI+ themes, characters, relationships, or content in the text
2. Suggest appropriate QueerLit vocabulary terms that would apply (use exact terms from the vocabulary when possible)
3. Provide brief justification for each suggested term based on textual evidence
4. If no exact vocabulary match exists, suggest the closest appropriate terms or describe what terms might be needed

Base your analysis solely on the literary content, not on external knowledge about the author or work."""

    # Create vocabulary information (SAME for all models - this is important for fair comparison)
    if not vocabulary_terms:
        vocab_info = "Focus on standard LGBTQI+ terminology."
    elif vocab_access_method == "Sample Terms (Fast)":
        vocab_sample = vocabulary_terms[:30]  # Same sample size for all
        vocab_text = ", ".join(vocab_sample)
        vocab_info = f"Some QueerLit vocabulary terms: {vocab_text}..."
    elif vocab_access_method == "Full List (Comprehensive)":
        # Include ALL vocabulary terms for all models
        vocab_text = ", ".join(vocabulary_terms)
        vocab_info = f"QueerLit vocabulary includes: {vocab_text}"
    elif vocab_access_method == "Categorized (Organized)":
        # Same categorization for all models
        identity_terms = [t for t in vocabulary_terms[:50] if
                          any(keyword in t.lower() for keyword in ['person', 'identitet', 'sexual', 'gender'])][:10]
        relationship_terms = [t for t in vocabulary_terms[:50] if
                              any(keyword in t.lower() for keyword in ['kärlek', 'relation', 'familj'])][:10]
        theme_terms = [t for t in vocabulary_terms[:50] if
                       any(keyword in t.lower() for keyword in ['tema', 'ämne', 'område'])][:10]

        vocab_info = f"""QueerLit vocabulary includes:
Identity/Gender: {', '.join(identity_terms)}
Relationships: {', '.join(relationship_terms)}
Themes: {', '.join(theme_terms)}
(Full vocabulary contains {len(vocabulary_terms)} terms total)"""

    # Calculate tokens for fixed parts (instruction + vocabulary)
    fixed_content = f"{base_instruction}\n\n{vocab_info}\n\n--- LITERARY TEXT TO ANALYZE ---\n"
    fixed_tokens = estimate_tokens(fixed_content)

    # Calculate available tokens for the text
    available_for_prompt = limit - response_reserve
    available_for_text = available_for_prompt - fixed_tokens - 100  # 100 token safety buffer

    # Convert available tokens to characters (roughly 4 chars per token)
    max_text_chars = max(available_for_text * 4, 500)  # At least 500 chars

    # Truncate text if needed
    if len(full_text) > max_text_chars:
        # Try to truncate at a sentence boundary
        truncated_text = full_text[:max_text_chars]

        # Find last period, exclamation, or question mark
        last_sentence = max(
            truncated_text.rfind('.'),
            truncated_text.rfind('!'),
            truncated_text.rfind('?')
        )

        if last_sentence > max_text_chars * 0.8:  # If we found a sentence ending in the last 20%
            truncated_text = truncated_text[:last_sentence + 1]

        truncated_text += "\n\n[TEXT TRUNCATED DUE TO MODEL TOKEN LIMITS - Showing first ~{:,} of {:,} characters]".format(
            len(truncated_text), len(full_text)
        )

        # Show info about truncation
        percentage_shown = (len(truncated_text) / len(full_text)) * 100
        st.info(
            f"ℹ️ {model_name}: Showing {percentage_shown:.1f}% of text to fit token limits while preserving full vocabulary")

        text_to_analyze = truncated_text
    else:
        text_to_analyze = full_text

    # Assemble final prompt
    final_prompt = f"{fixed_content}{text_to_analyze}"

    # Final safety check
    final_tokens = estimate_tokens(final_prompt)
    if final_tokens > available_for_prompt:
        # Emergency truncation - remove more text
        excess_tokens = final_tokens - available_for_prompt
        chars_to_remove = excess_tokens * 4 + 200  # Extra safety margin
        if len(text_to_analyze) > chars_to_remove:
            text_to_analyze = text_to_analyze[:-chars_to_remove] + "\n[TRUNCATED]"
            final_prompt = f"{fixed_content}{text_to_analyze}"

    return final_prompt


# Helper function to extract vocabulary terms from text
def extract_vocabulary_terms_from_text(text, vocabulary_terms):
    """Extract QueerLit vocabulary terms mentioned in the model's response text"""
    if not vocabulary_terms or not text:
        return []

    found_terms = []
    text_lower = text.lower()

    for term in vocabulary_terms:
        term_lower = term.lower()

        if term_lower in text_lower:
            pattern = r'\b' + re.escape(term_lower) + r'\b'
            if re.search(pattern, text_lower):
                found_terms.append(term)

        if '(hbtqi)' in term_lower:
            term_without_suffix = term_lower.replace('(hbtqi)', '').strip()
            pattern = r'\b' + re.escape(term_without_suffix) + r'\b'
            if re.search(pattern, text_lower) and term not in found_terms:
                found_terms.append(term)

    return list(set(found_terms))


# Helper function to calculate metrics
def calculate_metrics(predicted_terms, ground_truth_terms):
    """Calculate precision, recall, and F1 score"""
    if not ground_truth_terms:
        return {"precision": 0, "recall": 0, "f1": 0, "tp": 0, "fp": 0, "fn": 0}

    predicted_set = set([term.lower().strip() for term in predicted_terms])
    gt_set = set([term.lower().strip() for term in ground_truth_terms])

    true_positives = len(predicted_set.intersection(gt_set))
    false_positives = len(predicted_set - gt_set)
    false_negatives = len(gt_set - predicted_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "predicted_terms": list(predicted_set),
        "ground_truth_terms": list(gt_set),
        "matched_terms": list(predicted_set.intersection(gt_set))
    }


# Helper function to load QueerLit vocabulary from TTL files
@st.cache_data
def load_queerlit_vocabulary(ttl_directory="QLITTTLS"):
    """Load and parse QueerLit vocabulary terms from TTL files"""
    vocabulary_terms = []
    ttl_path = Path(ttl_directory)

    if not ttl_path.exists():
        return None, "TTL directory not found. Please create 'QLITTTLS' directory with TTL files."

    try:
        for ttl_file in ttl_path.glob("*.ttl"):
            with open(ttl_file, 'r', encoding='utf-8') as f:
                content = f.read()

                label_patterns = [
                    r'rdfs:label\s+"([^"]+)"',
                    r'skos:prefLabel\s+"([^"]+)"',
                    r'<[^>]*>\s+rdfs:label\s+"([^"]+)"',
                    r'<[^>]*>\s+skos:prefLabel\s+"([^"]+)"'
                ]

                for pattern in label_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    vocabulary_terms.extend(matches)

        vocabulary_terms = sorted(list(set(vocabulary_terms)))
        return vocabulary_terms, f"Loaded {len(vocabulary_terms)} terms from {len(list(ttl_path.glob('*.ttl')))} TTL files"

    except Exception as e:
        return None, f"Error loading TTL files: {str(e)}"


# Helper function to parse MARC and extract QLIT terms
def parse_file_content(file_content):
    """Separates MARC metadata from full text and extracts existing QLIT terms"""
    lines = file_content.split('\n')

    separator_indices = [i for i, line in enumerate(lines) if '---' in line and len(line) > 50]

    if separator_indices:
        marc_section = '\n'.join(lines[:separator_indices[0]])
        full_text = '\n'.join(lines[separator_indices[0]:])
    else:
        marc_end = 0
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in ['title:', 'author:', 'stockholm', 'förlag']):
                marc_end = i + 10
                break

        marc_section = '\n'.join(lines[:marc_end])
        full_text = '\n'.join(lines[marc_end:])

    existing_qlit_terms = []
    peripheral_terms = []

    for line in marc_section.split('\n'):
        # Extract main QLIT terms from 650 fields
        if '650' in line and 'qlit' in line.lower():
            if 'https://queerlit.dh.gu.se/qlit/v1/' in line:
                parts = line.split('a ')
                if len(parts) > 1:
                    term = parts[1].split('0 ')[0].strip()
                    if term:  # Removed the requirement for parentheses
                        existing_qlit_terms.append(term)

        # Extract peripheral terms from 590 fields
        elif '590' in line and 'a ' in line:
            # Handle various possible formats
            if '\ta ' in line:  # Tab separator
                parts = line.split('\ta ')
            else:
                parts = line.split('a ')

            if len(parts) > 1:
                # Extract the term, handling potential 'qlit' suffix
                term_part = parts[1].strip()
                if ' qlit' in term_part:
                    term = term_part.split(' qlit')[0].strip()
                else:
                    term = term_part.strip()

                if term:
                    peripheral_terms.append(term)

    # Combine all terms for evaluation purposes
    all_terms = existing_qlit_terms + peripheral_terms

    return marc_section, full_text, existing_qlit_terms, peripheral_terms, all_terms




# NEW: Enhanced error handling with retry logic
def call_model_with_retry(model_func, prompt, api_key, model_name, max_retries=3):
    """Call model with exponential backoff retry"""
    for attempt in range(max_retries):
        try:
            result = model_func(prompt, api_key)
            if not result.startswith("Error"):
                return result
            # If it's an error response, try again
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Error calling {model_name} after {max_retries} attempts: {str(e)}"
            time.sleep(2 ** attempt)  # Exponential backoff
    return f"Error: Max retries exceeded for {model_name}"


# ----------- Model Wrappers with Enhanced Error Handling -----------

def call_claude(prompt, api_key):
    try:
        if not api_key:
            return "Error calling Claude: No API key provided"
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except anthropic.APIError as e:
        return f"Error calling Claude API: {str(e)}"
    except Exception as e:
        return f"Error calling Claude: {str(e)}"


def call_chatgpt(prompt, api_key):
    try:
        if not api_key:
            return "Error calling ChatGPT: No API key provided"
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling ChatGPT: {str(e)}"


def call_deepseek(prompt, api_key):
    try:
        if not api_key:
            return "Error calling DeepSeek: No API key provided"
        headers = {"Authorization": f"Bearer {api_key}"}
        json_data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=json_data,
            timeout=30  # Add timeout
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "Error calling DeepSeek: Request timed out"
    except requests.exceptions.RequestException as e:
        return f"Error calling DeepSeek: {str(e)}"
    except Exception as e:
        return f"Error calling DeepSeek: {str(e)}"


def call_gemini(prompt, api_key):
    try:
        if not api_key:
            return "Error calling Gemini: No API key provided"
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        if hasattr(response, 'text') and response.text:
            return response.text
        elif hasattr(response, 'parts') and response.parts:
            return ''.join([part.text for part in response.parts if hasattr(part, 'text')])
        else:
            return "Error: No response generated from Gemini"
    except Exception as e:
        return f"Error calling Gemini: {str(e)}"


# NEW: Process single file function for better memory management
def process_single_file(file, base_prompt, vocabulary_terms, vocab_access_method, api_keys):
    """Process a single file and return results"""
    try:
        # Read file content
        file.seek(0)  # Reset file pointer
        file_content = file.read().decode('utf-8')

        # Parse file content
        marc_section, full_text, existing_qlit_terms, peripheral_terms, all_terms = parse_file_content(file_content)

        # Create model-specific prompts
        prompts = {
            "claude": create_model_prompt(base_prompt, full_text, vocabulary_terms, vocab_access_method, "claude"),
            "chatgpt": create_model_prompt(base_prompt, full_text, vocabulary_terms, vocab_access_method, "chatgpt"),
            "deepseek": create_model_prompt(base_prompt, full_text, vocabulary_terms, vocab_access_method, "deepseek"),
            "gemini": create_model_prompt(base_prompt, full_text, vocabulary_terms, vocab_access_method, "gemini")
        }

        # Call models with retry logic
        results = {
            "claude": call_model_with_retry(call_claude, prompts["claude"], api_keys.get("anthropic", ""), "Claude"),
            "chatgpt": call_model_with_retry(call_chatgpt, prompts["chatgpt"], api_keys.get("openai", ""), "ChatGPT"),
            "deepseek": call_model_with_retry(call_deepseek, prompts["deepseek"], api_keys.get("deepseek", ""),
                                              "DeepSeek"),
            "gemini": call_model_with_retry(call_gemini, prompts["gemini"], api_keys.get("gemini", ""), "Gemini")
        }

        return {
            "filename": file.name,
            "marc_section": marc_section,
            "full_text": full_text,
            "existing_qlit_terms": existing_qlit_terms,
            "results": results
        }

    except Exception as e:
        return {
            "filename": file.name,
            "error": f"Error processing file: {str(e)}"
        }


# Helper function to safely render markdown with HTML
def safe_render_response(text, label="View Response"):
    """Safely render model response in a collapsible container"""
    if text.startswith("Error"):
        return f'''
        <div style="
            color: #721c24;
            background-color: #f8d7da;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #f5c6cb;
            font-size: 14px;
            line-height: 1.6;
            font-family: system-ui, sans-serif;
            font-weight: bold;">
            {text}
        </div>
        '''

    # Convert markdown-like formatting to HTML
    text = text.replace('\n', '<br>')
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)

    return f'''
    <details style="margin-bottom: 1em;">
        <summary style="
            color: #212529;
            font-weight: bold;
            font-size: 15px;
            cursor: pointer;
            padding: 8px;
            background-color: #e9ecef;
            border: 1px solid #ced4da;
            border-radius: 6px;
            font-family: system-ui, sans-serif;">{label}</summary>
        <div style="
            color: #212529;
            background-color: #fefefe;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #dee2e6;
            font-size: 14px;
            line-height: 1.6;
            font-family: system-ui, sans-serif;
            margin-top: 10px;">
            {text}
        </div>
    </details>
    '''

# ----------- Streamlit UI -----------

st.set_page_config(page_title="AI Model Comparator for Subject Indexing", layout="wide")
st.title("🤖 AI Model Comparator for Subject Indexing")

# Check if API keys are loaded
if not api_keys:
    st.error("❌ No API keys found. Please configure your API keys in `.streamlit/secrets.toml`")
    st.stop()

# Try to load QueerLit vocabulary
vocabulary_terms, vocab_status = load_queerlit_vocabulary()

# Show vocabulary status
if vocabulary_terms:
    st.success(f"✅ {vocab_status}")
    with st.expander(f"📚 QueerLit Vocabulary Preview ({len(vocabulary_terms)} terms)", expanded=False):
        preview_terms = vocabulary_terms[:20]
        st.write(", ".join(preview_terms))
        if len(vocabulary_terms) > 20:
            st.write(f"... and {len(vocabulary_terms) - 20} more terms")
else:
    st.warning(f"⚠️ {vocab_status}")
    st.info(
        "💡 **To improve testing**: Download QueerLit TTL files and place them in a 'QLITTTLS' directory for more accurate vocabulary guidance.")

# Model information box
with st.expander("ℹ️ Model Information", expanded=False):
    st.markdown("""
    **Models being compared:**
    - **Claude**: claude-3-5-sonnet-20241022 (Anthropic) - 200K tokens
    - **ChatGPT**: gpt-4 (OpenAI) - 8K tokens
    - **DeepSeek**: deepseek-chat (DeepSeek) - 32K tokens
    - **Gemini**: gemini-1.5-flash (Google) - 1M tokens

    All models use temperature=0.7 for consistent comparison.
    """)

# Test Mode Selection
test_mode = st.radio(
    "Select Mode:",
    ["Custom Prompt Testing", "QueerLit Subject Indexing Task"],
    horizontal=True
)

# Initialize variables
user_prompt = ""
base_prompt = ""
uploaded_files = None
vocab_access_method = "Sample Terms (Fast)"

if test_mode == "Custom Prompt Testing":
    user_prompt = st.text_area(
        "Enter your prompt for all models",
        value="You are a subject indexer. Based on the text, suggest relevant terms from the QueerLit vocabulary. Use exact vocabulary terms when possible.",
        height=150
    )

    run_button_text = "Run on All Models"

elif test_mode == "QueerLit Subject Indexing Task":
    st.subheader("📚 QueerLit Subject Indexing Task")

    # File upload
    uploaded_files = st.file_uploader(
        "Upload MARC/text files for subject indexing",
        type=['txt'],
        accept_multiple_files=True,
        help="Upload one or more files with MARC metadata and full text"
    )

    # Create enhanced prompt based on available vocabulary
    if vocabulary_terms:
        vocab_access_method = st.radio(
            "📚 How to provide vocabulary to models:",
            ["Sample Terms (Fast)", "Full List (Comprehensive)", "Categorized (Organized)"],
            help="Choose how much vocabulary information to include in the prompt"
        )

        default_prompt = f"""You are a subject indexer specializing in LGBTQI+ literature analysis. Your task is to analyze the provided literary work and suggest relevant subject terms from the QueerLit controlled vocabulary.

Please analyze ONLY the literary text provided (ignore any metadata) and:
1. Identify specific LGBTQI+ themes, characters, relationships, or content in the text
2. Suggest appropriate QueerLit vocabulary terms that would apply (use exact terms from the vocabulary when possible)
3. Provide brief justification for each suggested term based on textual evidence
4. If no exact vocabulary match exists, suggest the closest appropriate terms or describe what terms might be needed

Base your analysis solely on the literary content, not on external knowledge about the author or work."""
    else:
        default_prompt = """You are a subject indexer specializing in LGBTQI+ literature analysis. Your task is to analyze the provided literary work and suggest relevant subject terms from controlled vocabularies used for LGBTQI+ literature.

Please analyze ONLY the literary text provided (ignore any metadata) and:
1. Identify specific LGBTQI+ themes, characters, relationships, or content
2. Suggest appropriate controlled vocabulary terms that would apply
3. Provide brief justification for each suggested term based on textual evidence
4. Focus on terms that accurately reflect explicit or implicit LGBTQI+ content

Base your analysis solely on the literary content, not on external knowledge about the author or work."""

    # Make prompt editable
    with st.expander("✏️ Edit QueerLit Analysis Prompt", expanded=False):
        st.markdown("**Customize the prompt sent to all models:**")

        # Quick prompt templates
        col1, col2, col3 = st.columns(3)

        template_choice = None
        with col1:
            if st.button("🔍 Focused Analysis"):
                template_choice = "focused"

        with col2:
            if st.button("📝 Detailed Output"):
                template_choice = "detailed"

        with col3:
            if st.button("🔄 Reset to Default"):
                template_choice = "default"

        # Apply template if selected
        if template_choice == "focused":
            prompt_value = """You are a subject indexer. Analyze the literary text and identify LGBTQI+ content.

Provide ONLY:
1. A list of relevant QueerLit vocabulary terms
2. One sentence explaining each term's relevance

Be concise and focus on explicit LGBTQI+ themes only."""

        elif template_choice == "detailed":
            prompt_value = f"""You are an expert subject indexer specializing in LGBTQI+ literary analysis. Perform a comprehensive analysis of the provided text.

Provide a detailed analysis including:
1. **Explicit LGBTQI+ Content**: Direct references to identities, relationships, themes
2. **Implicit/Coded Content**: Subtle representations, metaphors, coded language
3. **Historical Context**: How the work relates to LGBTQI+ history/movements
4. **Suggested Terms**: Specific vocabulary terms with detailed justifications
5. **Confidence Levels**: Rate each suggestion as High/Medium/Low confidence

Be thorough and scholarly in your analysis."""

        else:  # default or initial load
            prompt_value = default_prompt

        base_prompt = st.text_area(
            "Prompt text:",
            value=prompt_value,
            height=300,
            help="Edit this prompt to test different instructions or approaches"
        )

    if uploaded_files:
        # File selection for analysis
        if len(uploaded_files) > 1:
            selected_file = st.selectbox(
                "Select a file to analyze:",
                uploaded_files,
                format_func=lambda x: x.name
            )
        else:
            selected_file = uploaded_files[0]

        file_content = selected_file.read().decode('utf-8')

        # Parse file content
        marc_section, full_text, existing_qlit_terms, peripheral_terms, all_terms = parse_file_content(file_content)

        # Display file info
        st.info(f"**Selected file:** {selected_file.name} ({len(file_content):,} characters)")

        # Show existing QLIT terms if found
        if existing_qlit_terms or peripheral_terms:
            if existing_qlit_terms:
                st.success(f"**Main QLIT terms (650) found:** {', '.join(existing_qlit_terms)}")
            if peripheral_terms:
                st.info(f"**Peripheral terms (590) found:** {', '.join(peripheral_terms)}")
            st.markdown(f"**Total terms for evaluation:** {len(all_terms)}")
        else:
            st.warning("No existing QLIT terms found in MARC metadata")

        # Display sections
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("📄 MARC Metadata Preview", expanded=False):
                st.text(marc_section[:1000] + "..." if len(marc_section) > 1000 else marc_section)

        with col2:
            with st.expander("📖 Full Text Preview", expanded=False):
                st.text(full_text[:1000] + "..." if len(full_text) > 1000 else full_text)

        # Option for batch processing
        if len(uploaded_files) > 1:
            batch_mode = st.checkbox(
                f"🔄 Batch process all {len(uploaded_files)} files",
                help="Process all uploaded files sequentially (this may take longer)"
            )
        else:
            batch_mode = False

        if not batch_mode:
            # Create optimized prompts for each model
            claude_prompt = create_model_prompt(base_prompt, full_text, vocabulary_terms, vocab_access_method, "claude")
            gpt_prompt = create_model_prompt(base_prompt, full_text, vocabulary_terms, vocab_access_method, "chatgpt")
            deepseek_prompt = create_model_prompt(base_prompt, full_text, vocabulary_terms, vocab_access_method,
                                                  "deepseek")
            gemini_prompt = create_model_prompt(base_prompt, full_text, vocabulary_terms, vocab_access_method, "gemini")

            # Show token estimates
            st.info(f"""**Token estimates:** 
            Claude: ~{estimate_tokens(claude_prompt):,} tokens | 
            GPT-4: ~{estimate_tokens(gpt_prompt):,} tokens | 
            DeepSeek: ~{estimate_tokens(deepseek_prompt):,} tokens | 
            Gemini: ~{estimate_tokens(gemini_prompt):,} tokens""")

            run_button_text = f"Analyze '{selected_file.name}' with All Models"
        else:
            run_button_text = f"Batch Analyze All {len(uploaded_files)} Files"
    else:
        st.info("Please upload a file to begin the QueerLit subject indexing task.")
        run_button_text = "Upload file first"

# Main execution button
if st.button(run_button_text):
    if test_mode == "Custom Prompt Testing" or (test_mode == "QueerLit Subject Indexing Task" and uploaded_files):

        if test_mode == "Custom Prompt Testing":
            # Simple custom prompt testing
            with st.spinner("Running models..."):
                claude_result = call_model_with_retry(call_claude, user_prompt, api_keys.get("anthropic", ""), "Claude")
                gpt_result = call_model_with_retry(call_chatgpt, user_prompt, api_keys.get("openai", ""), "ChatGPT")
                ds_result = call_model_with_retry(call_deepseek, user_prompt, api_keys.get("deepseek", ""), "DeepSeek")
                gem_result = call_model_with_retry(call_gemini, user_prompt, api_keys.get("gemini", ""), "Gemini")

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🤖 Claude")
                st.markdown(safe_render_response(claude_result), unsafe_allow_html=True)

                st.markdown("### 🌊 DeepSeek")
                st.markdown(safe_render_response(ds_result), unsafe_allow_html=True)

            with col2:
                st.markdown("### 🔥 ChatGPT")
                st.markdown(safe_render_response(gpt_result), unsafe_allow_html=True)

                st.markdown("### 💎 Gemini")
                st.markdown(safe_render_response(gem_result), unsafe_allow_html=True)

        else:
            # QueerLit Subject Indexing Task
            if batch_mode and len(uploaded_files) > 1:
                # ENHANCED: Batch processing with better memory management
                st.subheader("🔄 Batch Processing Results")

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Process files one at a time to manage memory
                for i, file in enumerate(uploaded_files):
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f'Processing file {i + 1} of {len(uploaded_files)}: {file.name}')

                    st.markdown(f"### 📄 File {i + 1}: {file.name}")

                    # Process single file
                    file_results = process_single_file(file, base_prompt, vocabulary_terms, vocab_access_method,
                                                       api_keys)

                    if "error" in file_results:
                        st.error(file_results["error"])
                        continue

                    # Display existing QLIT terms
                    if file_results["existing_qlit_terms"] or file_results.get("peripheral_terms", []):
                        if file_results["existing_qlit_terms"]:
                            st.success(f"**Main QLIT terms (650):** {', '.join(file_results['existing_qlit_terms'])}")

                        if file_results.get("peripheral_terms", []):
                            st.info(f"**Peripheral terms (590):** {', '.join(file_results['peripheral_terms'])}")

                        all_terms = file_results.get("all_existing_terms",
                                                     file_results["existing_qlit_terms"] + file_results.get(
                                                         "peripheral_terms", []))
                        st.markdown(f"**Total reference terms:** {len(all_terms)} terms")
                    else:
                        st.warning("No existing QLIT terms found in MARC metadata")

                    # Display results in columns
                    col1, col2 = st.columns(2)

                    with col1:
                        with st.container():
                            st.markdown("#### 🤖 Claude")
                            st.markdown(
                                f'<div style="border: 2px solid #1f77b4; border-radius: 8px; padding: 15px; margin: 5px 0; background-color: #f0f8ff;">{safe_render_response(file_results["results"]["claude"])}</div>',
                                unsafe_allow_html=True)

                        with st.container():
                            st.markdown("#### 🌊 DeepSeek")
                            st.markdown(
                                f'<div style="border: 2px solid #2ca02c; border-radius: 8px; padding: 15px; margin: 5px 0; background-color: #f0fff0;">{safe_render_response(file_results["results"]["deepseek"])}</div>',
                                unsafe_allow_html=True)

                    with col2:
                        with st.container():
                            st.markdown("#### 🔥 ChatGPT")
                            st.markdown(
                                f'<div style="border: 2px solid #ff7f0e; border-radius: 8px; padding: 15px; margin: 5px 0; background-color: #fff8f0;">{safe_render_response(file_results["results"]["chatgpt"])}</div>',
                                unsafe_allow_html=True)

                        with st.container():
                            st.markdown("#### 💎 Gemini")
                            st.markdown(
                                f'<div style="border: 2px solid #d62728; border-radius: 8px; padding: 15px; margin: 5px 0; background-color: #fff0f0;">{safe_render_response(file_results["results"]["gemini"])}</div>',
                                unsafe_allow_html=True)

                    # Clear memory after each file
                    del file_results

                    if i < len(uploaded_files) - 1:
                        st.markdown("---")

                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                st.success(f"✅ Batch processing complete! Processed {len(uploaded_files)} files.")

            else:
                # Single file processing
                with st.spinner("Running analysis with all models..."):
                    # Use retry logic for each model
                    claude_result = call_model_with_retry(call_claude, claude_prompt, api_keys.get("anthropic", ""),
                                                          "Claude")
                    gpt_result = call_model_with_retry(call_chatgpt, gpt_prompt, api_keys.get("openai", ""), "ChatGPT")
                    ds_result = call_model_with_retry(call_deepseek, deepseek_prompt, api_keys.get("deepseek", ""),
                                                      "DeepSeek")
                    gem_result = call_model_with_retry(call_gemini, gemini_prompt, api_keys.get("gemini", ""), "Gemini")

                # Results with comparison to existing QLIT terms
                st.subheader("🏷️ Subject Indexing Results")

                # Show existing QLIT terms at the top
                if existing_qlit_terms:
                    st.markdown("### 📋 Existing QLIT Terms (for comparison)")
                    st.success(f"**Reference terms:** {', '.join(all_terms)}")

                    if vocabulary_terms:
                        found_terms = [term for term in all_terms if any(
                            vocab_term.lower() in term.lower() or term.lower() in vocab_term.lower() for vocab_term in
                            vocabulary_terms)]
                        if found_terms:
                            st.info(f"**Terms found in vocabulary:** {', '.join(found_terms)}")

                st.markdown("---")

                # Extract vocabulary terms and calculate metrics if possible
                if vocabulary_terms and existing_qlit_terms:
                    st.subheader("📊 Automatic Term Extraction & Metrics")

                    model_results = {
                        "Claude": claude_result,
                        "ChatGPT": gpt_result,
                        "DeepSeek": ds_result,
                        "Gemini": gem_result
                    }

                    metrics_data = []

                    for model_name, response_text in model_results.items():
                        if not response_text.startswith("Error"):
                            extracted_terms = extract_vocabulary_terms_from_text(response_text, vocabulary_terms)
                            metrics = calculate_metrics(extracted_terms, existing_qlit_terms)

                            metrics_data.append({
                                "Model": model_name,
                                "Extracted_Terms": extracted_terms,
                                "Matched_Terms": metrics["matched_terms"],
                                "Precision": metrics["precision"],
                                "Recall": metrics["recall"],
                                "F1": metrics["f1"],
                                "TP": metrics["tp"],
                                "FP": metrics["fp"],
                                "FN": metrics["fn"]
                            })

                    # Display metrics table
                    if metrics_data:
                        summary_data = []
                        for data in metrics_data:
                            summary_data.append({
                                "Model": data["Model"],
                                "Precision": f"{data['Precision']:.3f}",
                                "Recall": f"{data['Recall']:.3f}",
                                "F1 Score": f"{data['F1']:.3f}",
                                "Terms Found": len(data["Extracted_Terms"]),
                                "Correct": data["TP"]
                            })

                        df_summary = pd.DataFrame(summary_data)
                        st.dataframe(df_summary, use_container_width=True)

                        # Best model
                        best_model = max(metrics_data, key=lambda x: x["F1"])
                        st.success(f"🏅 **Best F1 Score:** {best_model['Model']} with {best_model['F1']:.3f}")

                        # Detailed breakdown for each model
                        with st.expander("🔍 Detailed Metrics Breakdown", expanded=False):
                            for data in metrics_data:
                                st.markdown(f"**{data['Model']} Analysis:**")
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.write(f"• Precision: {data['Precision']:.3f}")
                                    st.write(f"• Recall: {data['Recall']:.3f}")
                                    st.write(f"• F1 Score: {data['F1']:.3f}")

                                with col2:
                                    st.write(f"• True Positives: {data['TP']}")
                                    st.write(f"• False Positives: {data['FP']}")
                                    st.write(f"• False Negatives: {data['FN']}")

                                if data["Matched_Terms"]:
                                    st.success(f"✅ Correctly identified: {', '.join(data['Matched_Terms'])}")

                                if data["Extracted_Terms"]:
                                    st.info(f"🔍 All extracted terms: {', '.join(data['Extracted_Terms'])}")

                                st.markdown("---")

                elif not vocabulary_terms:
                    st.info(
                        "💡 Load QueerLit vocabulary (TTL files) to enable automatic term extraction and metrics calculation")

                elif not existing_qlit_terms:
                    st.info("💡 This file has no existing QLIT terms for comparison")

                st.markdown("---")

                # Display each model's results
                st.markdown("### 🤖 Model Responses")

                # Create tabs for better organization
                tab1, tab2, tab3, tab4 = st.tabs(["🤖 Claude", "🔥 ChatGPT", "🌊 DeepSeek", "💎 Gemini"])

                with tab1:
                    st.markdown(
                        f'<div style="border: 2px solid #1f77b4; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #f0f8ff;">{safe_render_response(claude_result)}</div>',
                        unsafe_allow_html=True)

                with tab2:
                    st.markdown(
                        f'<div style="border: 2px solid #ff7f0e; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #fff8f0;">{safe_render_response(gpt_result)}</div>',
                        unsafe_allow_html=True)

                with tab3:
                    st.markdown(
                        f'<div style="border: 2px solid #2ca02c; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #f0fff0;">{safe_render_response(ds_result)}</div>',
                        unsafe_allow_html=True)

                with tab4:
                    st.markdown(
                        f'<div style="border: 2px solid #d62728; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #fff0f0;">{safe_render_response(gem_result)}</div>',
                        unsafe_allow_html=True)

    else:
        st.warning("Please configure your inputs before running the analysis.")

# Footer
st.markdown("---")
st.markdown(
    "**Note:** This tool compares AI models for LGBTQI+ literature subject indexing. Results may vary based on model availability and API limits. Tokens funded by the programmer's beer money. Use responsibly.")
