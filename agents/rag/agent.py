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
    chunks = generate_docs_chunks("./docs/python/examples")
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
    sources = set()
    
    for i, result in enumerate(results):
        content = result.metadata.get('content', '')
        source = result.metadata.get('source', f'Document {i+1}')
        sources.add(source)
        
        context_text += f"Document: {source}\n"
        context_text += f"Content: {content}\n"
        context_text += f"---\n"
    
    system_prompt = """You are a helpful assistant that answers questions based on provided documents. 
    Always cite which documents your answer is based on. 
    At the end of your response, include a "Sources:" section listing the documents you referenced."""
    
    user_prompt = f"""Based on the following documents, please answer this question: {query}

Documents:
{context_text}

Please provide a comprehensive answer and cite your sources at the end."""
    
    llm_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    return llm_response.choices[0].message.content