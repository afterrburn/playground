from agentuity import AgentRequest, AgentResponse, AgentContext
from agents.rag.process_doc import generate_docs_chunks
from openai import OpenAI
import uuid
client = OpenAI()

vector_db_name = "dev-doc"
vector_dimension = 1536
embeddings_model = "text-embedding-3-small"

def get_batch_embeddings(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        input=texts,
        model=embeddings_model
    )
    embeddings = [embedding.embedding for embedding in response.data]
    return embeddings

def get_embeddings(text: str) -> list[float]:
    response = client.embeddings.create(
        input=text,
        model=embeddings_model
    )
    embedding = response.data[0].embedding
    return embedding

async def run(request: AgentRequest, response: AgentResponse, context: AgentContext):
    query = await request.data.text() or "Hello, OpenAI"
    response_text = await answer_query(context, query)
    return response.text(response_text)

async def get_relevant_chunks(context: AgentContext, query: str):
    # embedding = get_embeddings(query)
    results = await context.vector.search(vector_db_name, query, limit=10)
    return results

async def load_docs(context: AgentContext):
    chunks = generate_docs_chunks("content/CLI")
    for chunk in chunks:
        embedded_chunk = get_embeddings(chunk.page_content)
        document = {
            "key": str(uuid.uuid4()),
            "embeddings": embedded_chunk,
            "metadata": {
                "content": chunk.page_content,
                "source": chunk.metadata.get("source", "")
            }
        }
        ids = await context.vector.upsert(vector_db_name, [document])
        print('Upserted document with ids: ', ids)


async def answer_query(context: AgentContext, query: str):
    results = await get_relevant_chunks(context, query)
    
    # Format the retrieved chunks for the LLM
    context_text = ""
    sources = []
    
    for i, result in enumerate(results):
        content = result.metadata.get('content', '')
        source = result.metadata.get('source', f'Document {i+1}')
        url = result.metadata.get('url', '')  # Assuming you store URLs in metadata
        page_title = result.metadata.get('title', source)  # Page title for better citations
        
        sources.append({
            'title': page_title,
            'source': source,
            'url': url
        })
        
        context_text += f"Source: {page_title}\n"
        if url:
            context_text += f"URL: {url}\n"
        context_text += f"Content: {content}\n"
        context_text += f"---\n"
    
    system_prompt = """You are a developer documentation assistant for the Agentuity platform. Your job is to help developers quickly implement solutions using the retrieved documentation context.

## Core Principles
- **Code First**: Always lead with working code examples when possible
- **Concise**: Keep explanations brief (1-2 sentences max for non-code content)
- **Actionable**: Focus on what the developer needs to DO, not theory
- **Accurate**: Only use information from the provided context

## Response Structure
Follow this format:

1. **Code Example** (if applicable)
   - Provide copy-pastable code
   - Use realistic variable names and common patterns
   - Include necessary imports/setup

2. **Brief Explanation** (1-2 sentences)
   - Explain WHEN to use this approach
   - Highlight the key benefit or purpose

3. **Source Reference**
   - Link to the specific documentation section
   - Use format: `[See: {page_title}]({url})` if URL available
   - Otherwise use: `Source: {page_title}`

4. **Warnings/Gotchas** (optional)
   - Only include if there are common pitfalls
   - Use ⚠️ emoji prefix
   - Keep to one line

## Instructions
- If the context contains code examples, adapt them to the user's specific question
- If multiple approaches exist, show the most common/recommended one first
- If the context doesn't fully answer the question, say "Based on the available documentation..." and provide what you can
- Always include source references
- Keep total response under 150 words unless code examples require more space
- If you don't know the answer or it's not in the context, suggest contacting the Agentuity team
- Only answer questions related to Agentuity platform technical documentation

## What NOT to do
- Don't make up information not in the context
- Don't provide lengthy theoretical explanations
- Don't include multiple approaches unless specifically asked
- Don't use marketing language or superlatives"""

    user_prompt = f"""Based on the following Agentuity documentation, please answer this question: {query}

Available Documentation:
{context_text}

Provide your response following the structure specified in your instructions."""
    
    try:
        llm_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Lower temperature for more consistent, factual responses
            max_tokens=500    # Limit response length to keep it concise
        )
        
        response_content = llm_response.choices[0].message.content
        
        # Ensure sources are always included even if LLM doesn't format them properly
        if "Source:" not in response_content and "[See:" not in response_content:
            response_content += "\n\n**Sources:**\n"
            for source in sources:
                if source['url']:
                    response_content += f"- [{source['title']}]({source['url']})\n"
                else:
                    response_content += f"- {source['title']}\n"
        
        return response_content
        
    except Exception as e:
        return f"I encountered an error while processing your question about the Agentuity platform. Please try rephrasing your question or contact the Agentuity team for assistance. Error: {str(e)}"