
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from openai import OpenAI
import chromadb
from chromadb.config import Settings
import pandas as pd
import re
import json
import google.generativeai as genai

# --- Page Configuration ---
st.set_page_config(
    page_title="Law Firm News Intelligence",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Law Firm News Intelligence Assistant")
st.info("Ask about legal news - find interesting developments or search specific topics.")

# --- API Client Initialization ---
try:
    if 'openai_client' not in st.session_state:
        st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    
    # Initialize Gemini
    if 'gemini_configured' not in st.session_state:
        genai.configure(api_key=st.secrets["Gemini_API_Key"])
        st.session_state.gemini_configured = True
    
except Exception as e:
    st.error(f"Failed to initialize API clients. Please check your API keys. Error: {e}")
    st.stop()

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Configuration")
    
    provider_choice = st.radio(
        "Choose LLM Provider:",
        ("OpenAI (GPT-4o)", "OpenAI (GPT-4o-mini)", "Google (Gemini 2.5 Pro)", "Google (Gemini 2.5 Flash)"),
        key="llm_provider"
    )
    
    st.divider()
    st.subheader("Search Settings")
    n_results = st.slider("Number of results to retrieve", 3, 10, 5)
    
    st.divider()
    st.markdown("**Query Types:**")
    st.markdown("- *'Find the most interesting news'* - Ranked by legal materiality")
    st.markdown("- *'Find news about [topic]'* - Topic-specific search")
    st.markdown("- *'Find interesting news about [topic]'* - Topic search + ranking")
    st.markdown("---")
    st.caption("**Scoring Models (Dynamic):**")
    st.caption("• GPT-4o when using OpenAI models")
    st.caption("• Gemini 2.5 Flash when using Gemini models")
    st.caption("---")
    st.caption("**Features:**")
    st.caption("✓ Automatic deduplication")
    st.caption("✓ Low-score warnings")
    st.caption("✓ Smart query detection")

# --- Initialize Persistent ChromaDB ---
@st.cache_resource
def initialize_system(csv_file, persist_directory="./chromadb_data"):
    """Initialize persistent ChromaDB and load news data if needed"""
    
    # Create persistent client
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=False
        )
    )
    
    # Try to get existing collection
    try:
        collection = client.get_collection("law_firm_news")
        st.info(f"✅ Loaded existing collection with {collection.count()} documents")
        df = pd.read_csv(csv_file)
        return collection, df, client
    except:
        st.warning("📂 Collection not found. Creating new collection...")
    
    # Read CSV
    df = pd.read_csv(csv_file)
    
    # Create new collection
    collection = client.create_collection(
        name="law_firm_news",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Prepare documents for embedding
    documents = []
    metadatas = []
    ids = []
    
    for idx, row in df.iterrows():
        doc_text = f"Company: {row['company_name']}\nDate: {row['Date']}\nNews: {row['Document']}"
        documents.append(doc_text)
        
        metadatas.append({
            "company": row['company_name'],
            "date": str(row['Date']),
            "url": row['URL'],
            "days_since_2000": int(row['days_since_2000']) if pd.notna(row['days_since_2000']) else 0
        })
        
        ids.append(f"doc_{idx}")
    
    # Add to ChromaDB in batches
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        collection.add(
            documents=batch_docs,
            metadatas=batch_meta,
            ids=batch_ids
        )
    
    st.success(f"✅ Created new collection with {collection.count()} documents")
    return collection, df, client

collection, df, client = initialize_system("/workspaces/HW2/HW 7 Source File/Example_news_info_for_testing.csv")
st.divider()

# --- Dynamic Legal Materiality Scoring ---
def score_with_structured_cot(doc_text, metadata, provider_choice):
    """AI-powered legal materiality scoring - uses GPT-4o or Gemini 2.5 Flash based on selected provider"""
    
    prompt = f"""Analyze this news article's legal materiality for a global law firm.

**NEWS:**
Company: {metadata.get('company', 'Unknown')}
Date: {metadata.get('date', 'Unknown')}
Content: {doc_text}

**THINK STEP-BY-STEP:**
1. What legal issues are present? (litigation, regulatory, M&A, compliance, IP, etc.)
2. What's the litigation risk (0-10)?
3. What's the regulatory impact (0-10)?
4. What are the financial stakes (0-10)?
5. What's the client advisory value (0-10)?
6. What's the overall materiality (0-10)?

**SCORING GUIDE:**
- 9-10: Major litigation/SEC investigation/billion-dollar M&A/systemic risks
- 7-8: Significant regulatory action/patent litigation/notable corporate events
- 5-6: Standard legal matters/routine M&A/contract disputes
- 3-4: Minor matters/policy updates/administrative actions
- 0-2: No legal significance/pure business news

Then respond with ONLY valid JSON in this format:
{{
    "legal_issues": ["issue1", "issue2"],
    "litigation_risk": 0-10,
    "regulatory_impact": 0-10,
    "financial_stakes": 0-10,
    "advisory_value": 0-10,
    "overall_score": 0-10,
    "reasoning": "Brief explanation of why this score"
}}"""

    try:
        # Use Gemini 2.5 Flash for Gemini models
        if "Google" in provider_choice:
            model = genai.GenerativeModel(model_name="gemini-2.5-flash")
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    max_output_tokens=400
                )
            )
            
            response_text = response.text
        
        # Use GPT-4o for OpenAI models
        else:
            response = st.session_state.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=400
            )
            
            response_text = response.choices[0].message.content
        
        # Extract JSON (handling markdown code blocks)
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            return result.get('overall_score', 5), result
        
        return 5, {"reasoning": "Failed to parse scoring response"}
        
    except Exception as e:
        return 5, {"reasoning": f"Scoring error: {str(e)}"}

# --- Query Type Detection with Better Logic ---
def detect_query_type(query):
    """Detect query type with better handling of mixed queries"""
    query_lower = query.lower()
    
    # Check for topic-specific keywords
    has_topic_keywords = any(word in query_lower for word in 
                             ['about', 'regarding', 'concerning', 'from', 'by', 'on'])
    
    # Check for interesting patterns
    interesting_patterns = [
        'interesting', 'important', 'significant', 'notable', 
        'top news', 'latest developments', 'key news', 'major news',
        'what should i know', 'highlights', 'priority', 'most material'
    ]
    has_interesting = any(pattern in query_lower for pattern in interesting_patterns)
    
    # Decision logic
    if has_topic_keywords and has_interesting:
        # "find interesting news about X" - do topic search then rank
        return "topic_then_rank"
    elif has_topic_keywords:
        # Pure topic search
        return "topic_search"
    elif has_interesting:
        # Pure interesting/ranking query
        return "interesting"
    else:
        return "topic_search"

# --- Extract Topic/Company from Query ---
def extract_topic_from_query(query):
    """Extract the main topic/company from a query"""
    query_lower = query.lower()
    
    # Look for "about X" or "regarding X" patterns
    for keyword in ['about', 'regarding', 'concerning', 'on']:
        if keyword in query_lower:
            # Extract everything after the keyword
            parts = query_lower.split(keyword)
            if len(parts) > 1:
                topic = parts[1].strip()
                # Clean up common words
                topic = topic.replace('the', '').replace('news', '').strip()
                return topic if topic else None
    
    return None

# --- Deduplicate Articles ---
def deduplicate_articles(articles):
    """Remove duplicate news stories based on URL"""
    seen_urls = set()
    unique_articles = []
    
    for article in articles:
        url = article['metadata']['url']
        # Also check for very similar URLs (different sources, same story)
        url_base = url.split('?')[0]  # Remove query parameters
        
        if url_base not in seen_urls:
            seen_urls.add(url_base)
            unique_articles.append(article)
    
    return unique_articles

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
if user_input := st.chat_input("Ask about legal news..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Display assistant response
    with st.chat_message("assistant"):
        query_type = detect_query_type(user_input)
        topic = extract_topic_from_query(user_input)
        
        if query_type == "interesting":
            # --- Handle "Most Interesting News" Query with AI Scoring ---
            scoring_model = "Gemini 2.5 Flash" if "Google" in provider_choice else "GPT-4o"
            st.markdown(f"*🔍 Analyzing legal materiality with {scoring_model}...*")
            
            # Get 8 documents for scoring
            results = collection.query(
                query_texts=[user_input],
                n_results=min(8, collection.count())
            )
            
            # Score each document using LLM
            scored_docs = []
            if results['documents'] and results['documents'][0]:
                progress_text = st.empty()
                
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    progress_text.markdown(f"*Scoring article {i+1}/{len(results['documents'][0])}...*")
                    
                    score, analysis = score_with_structured_cot(doc, metadata, provider_choice)
                    scored_docs.append({
                        'doc': doc,
                        'metadata': metadata,
                        'score': score,
                        'analysis': analysis,
                        'distance': results['distances'][0][i] if 'distances' in results else 0
                    })
                
                progress_text.empty()
            
            # Deduplicate articles
            scored_docs = deduplicate_articles(scored_docs)
            
            # Sort by AI-generated materiality score
            scored_docs.sort(key=lambda x: x['score'], reverse=True)
            
            # Take top results
            top_docs = scored_docs[:n_results]
            
            # Check if all scores are low
            avg_score = sum(doc['score'] for doc in top_docs) / len(top_docs) if top_docs else 0
            if avg_score <= 5:
                st.warning("⚠️ Note: All retrieved articles have low legal materiality scores (routine business news)")
            
            # Display scoring results in an expander
            with st.expander(f"📊 View AI Scoring Details ({scoring_model})", expanded=False):
                for i, item in enumerate(top_docs):
                    st.markdown(f"### Article {i+1}: Score {item['score']}/10")
                    analysis = item['analysis']
                    
                    if analysis and isinstance(analysis, dict):
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Litigation Risk", f"{analysis.get('litigation_risk', 'N/A')}/10")
                        col2.metric("Regulatory Impact", f"{analysis.get('regulatory_impact', 'N/A')}/10")
                        col3.metric("Financial Stakes", f"{analysis.get('financial_stakes', 'N/A')}/10")
                        col4.metric("Advisory Value", f"{analysis.get('advisory_value', 'N/A')}/10")
                        
                        if analysis.get('legal_issues'):
                            st.markdown(f"**Legal Issues:** {', '.join(analysis['legal_issues'])}")
                        
                        if analysis.get('reasoning'):
                            st.markdown(f"**Analysis:** {analysis['reasoning']}")
                    
                    st.markdown(f"**Company:** {item['metadata'].get('company', 'Unknown')}")
                    st.markdown(f"**URL:** {item['metadata']['url']}")
                    st.divider()
            
            # Build context for LLM response
            context = "**Top Legal News by AI Materiality Score:**\n\n"
            for i, item in enumerate(top_docs):
                context += f"**Article {i+1}** (Materiality Score: {item['score']}/10):\n"
                context += f"{item['doc']}\n"
                
                # Include analysis
                if item['analysis'] and isinstance(item['analysis'], dict):
                    if item['analysis'].get('legal_issues'):
                        context += f"Key Issues: {', '.join(item['analysis']['legal_issues'])}\n"
                    if item['analysis'].get('reasoning'):
                        context += f"Analysis: {item['analysis']['reasoning']}\n"
                
                context += f"URL: {item['metadata']['url']}\n\n"
            
            system_prompt = f"""You are an expert legal news analyst for a global law firm. 

CRITICAL INSTRUCTIONS:
- Use ONLY the news articles provided below
- DO NOT use general knowledge or say "I cannot access real-time news"
- If multiple articles describe the same event, consolidate into one analysis
- Base your response SOLELY on the provided articles

The following news articles have been ranked by AI-powered legal materiality analysis:

{context}

Present them as a prioritized briefing for senior partners, highlighting:
- Key legal implications and practice areas involved
- Potential client impacts and business opportunities
- Regulatory considerations and compliance issues
- Strategic action items or watch points

If articles are about routine business news with low materiality scores, acknowledge this and provide a brief business summary instead of overstating legal significance."""

        elif query_type == "topic_then_rank":
            # --- Handle "Interesting News About X" - Topic Search then Rank ---
            st.markdown(f"*🔍 Searching for news about '{topic}' and ranking by legal materiality...*")
            
            scoring_model = "Gemini 2.5 Flash" if "Google" in provider_choice else "GPT-4o"
            
            # First, do topic-specific search
            results = collection.query(
                query_texts=[user_input],
                n_results=min(12, collection.count())
            )
            
            # Score the topic-specific results
            scored_docs = []
            if results['documents'] and results['documents'][0]:
                progress_text = st.empty()
                
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    progress_text.markdown(f"*Scoring article {i+1}/{len(results['documents'][0])}...*")
                    
                    score, analysis = score_with_structured_cot(doc, metadata, provider_choice)
                    scored_docs.append({
                        'doc': doc,
                        'metadata': metadata,
                        'score': score,
                        'analysis': analysis,
                        'distance': results['distances'][0][i] if 'distances' in results else 0
                    })
                
                progress_text.empty()
            
            # Deduplicate articles
            scored_docs = deduplicate_articles(scored_docs)
            
            # Sort by score
            scored_docs.sort(key=lambda x: x['score'], reverse=True)
            top_docs = scored_docs[:n_results]
            
            # Check uniqueness
            if len(top_docs) == 1 or (len(top_docs) > 1 and all(doc['score'] == top_docs[0]['score'] for doc in top_docs)):
                st.info(f"ℹ️ Found {len(scored_docs)} article(s) about {topic} (duplicates removed)")
            
            # Check if all scores are low
            avg_score = sum(doc['score'] for doc in top_docs) / len(top_docs) if top_docs else 0
            if avg_score <= 5:
                st.warning("⚠️ Note: All retrieved articles have low legal materiality scores (routine business news)")
            
            # Display scoring results
            with st.expander(f"📊 View AI Scoring Details ({scoring_model})", expanded=False):
                for i, item in enumerate(top_docs):
                    st.markdown(f"### Article {i+1}: Score {item['score']}/10")
                    analysis = item['analysis']
                    
                    if analysis and isinstance(analysis, dict):
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Litigation Risk", f"{analysis.get('litigation_risk', 'N/A')}/10")
                        col2.metric("Regulatory Impact", f"{analysis.get('regulatory_impact', 'N/A')}/10")
                        col3.metric("Financial Stakes", f"{analysis.get('financial_stakes', 'N/A')}/10")
                        col4.metric("Advisory Value", f"{analysis.get('advisory_value', 'N/A')}/10")
                        
                        if analysis.get('legal_issues'):
                            st.markdown(f"**Legal Issues:** {', '.join(analysis['legal_issues'])}")
                        
                        if analysis.get('reasoning'):
                            st.markdown(f"**Analysis:** {analysis['reasoning']}")
                    
                    st.markdown(f"**Company:** {item['metadata'].get('company', 'Unknown')}")
                    st.markdown(f"**URL:** {item['metadata']['url']}")
                    st.divider()
            
            # Build context
            context = f"**News about {topic} (Ranked by Legal Materiality):**\n\n"
            for i, item in enumerate(top_docs):
                context += f"**Article {i+1}** (Score: {item['score']}/10):\n{item['doc']}\n"
                if item['analysis'] and isinstance(item['analysis'], dict):
                    if item['analysis'].get('legal_issues'):
                        context += f"Key Issues: {', '.join(item['analysis']['legal_issues'])}\n"
                    if item['analysis'].get('reasoning'):
                        context += f"Analysis: {item['analysis']['reasoning']}\n"
                context += f"URL: {item['metadata']['url']}\n\n"
            
            system_prompt = f"""You are an expert legal news analyst for a global law firm.

CRITICAL INSTRUCTIONS:
- Use ONLY the news articles provided below about {topic}
- DO NOT use general knowledge or say "I cannot access real-time news"
- If multiple articles describe the same event, consolidate your analysis
- Base your response SOLELY on the provided articles

{context}

Provide a focused analysis of {topic}-related news, highlighting legal implications and business opportunities. If the news is routine business activity with low legal materiality, acknowledge this and provide a concise summary rather than overstating significance."""

        else:
            # --- Handle Topic-Specific Search ---
            results = collection.query(
                query_texts=[user_input],
                n_results=n_results
            )
            
            # Build context from search results
            context = ""
            if results['documents'] and results['documents'][0]:
                context = "**Relevant news articles:**\n\n"
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    context += f"**Article {i+1}:**\n{doc}\n"
                    context += f"URL: {metadata['url']}\n\n"
            
            system_prompt = f"""You are a legal news assistant for a global law firm.

CRITICAL INSTRUCTIONS:
- Use ONLY the news articles provided below
- DO NOT use general knowledge or say "I cannot access real-time news"
- Base your response SOLELY on the provided articles

{context}

Answer the user's question comprehensively with focus on legal implications and business impact. Highlight legal risks, opportunities, and relevant precedents if applicable."""
        
        # Build conversation buffer
        conversation_buffer = []
        for msg in st.session_state.messages:
            conversation_buffer.append({"role": msg["role"], "content": msg["content"]})
        
        # Call LLM
        full_response = ""
        try:
            messages_for_api = [
                {"role": "system", "content": system_prompt},
                *conversation_buffer
            ]
            
            if provider_choice == "OpenAI (GPT-4o)":
                stream = st.session_state.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages_for_api,
                    stream=True,
                )
                full_response = st.write_stream(stream)
            
            elif provider_choice == "OpenAI (GPT-4o-mini)":
                stream = st.session_state.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages_for_api,
                    stream=True,
                )
                full_response = st.write_stream(stream)
            
            elif provider_choice == "Google (Gemini 2.5 Pro)":
                gemini_messages = []
                system_instruction = messages_for_api[0]["content"]
                
                for msg in messages_for_api[1:]:
                    role = "user" if msg["role"] == "user" else "model"
                    gemini_messages.append({"role": role, "parts": [msg["content"]]})
                
                model = genai.GenerativeModel(
                    model_name="gemini-2.5-pro",
                    system_instruction=system_instruction
                )
                
                response = model.generate_content(
                    gemini_messages,
                    stream=True
                )
                
                placeholder = st.empty()
                full_response = ""
                for chunk in response:
                    if chunk.text:
                        full_response += chunk.text
                        placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            
            elif provider_choice == "Google (Gemini 2.5 Flash)":
                gemini_messages = []
                system_instruction = messages_for_api[0]["content"]
                
                for msg in messages_for_api[1:]:
                    role = "user" if msg["role"] == "user" else "model"
                    gemini_messages.append({"role": role, "parts": [msg["content"]]})
                
                model = genai.GenerativeModel(
                    model_name="gemini-2.5-flash",
                    system_instruction=system_instruction
                )
                
                response = model.generate_content(
                    gemini_messages,
                    stream=True
                )
                
                placeholder = st.empty()
                full_response = ""
                for chunk in response:
                    if chunk.text:
                        full_response += chunk.text
                        placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
                
        except Exception as e:
            full_response = f"An error occurred: {e}"
            st.error(full_response)
    
    # Add assistant response to chat history
    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})