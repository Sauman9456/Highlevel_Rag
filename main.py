import os
from datetime import datetime
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma as load_chroma
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever, BM25Retriever, ContextualCompressionRetriever
from langchain_cohere import CohereRerank
import instructor
from pydantic import BaseModel, Field
from openai import OpenAI
from fastapi import FastAPI
from typing import List, Dict


# Set environment variables for API keys
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
os.environ["COHERE_API_KEY"] = os.environ.get("COHERE_API_KEY", "")

def load_vector_store_docs(persist_directory, embedding_model):
  vector_store = load_chroma(
      embedding_function=embedding_model,
      persist_directory = persist_directory

  )
  """
    Load vector store and return vector store and documents.
    
    Args:
        persist_directory (str): Directory where the vector store is persisted.
        embedding_model: Embedding model used for vectorization.
    
    Returns:
        tuple: Vector store and list of documents.
    """
  
  vectorstore_data = vector_store.get()
  docs = []
  for content, metadata in zip(vectorstore_data['documents'], vectorstore_data['metadatas']):
    docs.append(Document(page_content=content, metadata=metadata))
  return vector_store, docs


def get_retriver(persist_directory, embedding_model, top = 20):
  """
    Initialize and return retrievers, including a compression retriever.
    
    Args:
        persist_directory (str): Directory where vector store is stored.
        embedding_model: Embedding model for vectorization.
        top (int, optional): Number of top documents to retrieve. Default is 20.
    
    Returns:
        tuple: Compression retriever and vector store retriever.
    """
  vectorstore, docs = load_vector_store_docs(persist_directory, embedding_model)

  vectorstore_retreiver = vectorstore.as_retriever(search_kwargs={"k": int(top)})

  keyword_retriever = BM25Retriever.from_documents(
      docs, k=int(top), top_n = int(top)
  )
  ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retreiver,
                                                    keyword_retriever],
                                        weights=[0.5, 0.5], k=int(top), top_n = int(top))


  compressor = CohereRerank(model="rerank-english-v3.0", top_n = 20)
  compression_retriever = ContextualCompressionRetriever(
      base_compressor=compressor, base_retriever=ensemble_retriever
  )

  return compression_retriever, vectorstore_retreiver



class Alternate_Questions(BaseModel):
    questions: List = Field(description="List of alternate questions")

def get_alternate_questions(question, titles):
  """
    Generate alternate questions based on the given question and document titles.

    Args:
        question (str): Original question.
        titles (str): Titles of documents for context.

    Returns:
        list: List of original and alternate questions.
    """
  

  client = instructor.from_openai(OpenAI())
  system_prompt = f"""You are an AI language model assistant. Your task is to generate three different versions of the given user question to retrieve relevant documents from a vector database.
By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search and keyword search.
Provide these alternative questions separated by newlines and number.

Note: Refer to the provided document titles from the vector database to generate alternate questions. These titles contain domain-specific jargon, terminology, acronyms, and synonyms that will assist you in creating contextually accurate questions.
My very few findings:
1. 'LC' is an abbreviation for Lead Connects. Here Sub account means LC account.
2. Terms like Transfer, Porting, and Moving are synonymous in case of porting or transfering phone number to new.
3. Scheduled or booked Meetings and Availability can be associated with the Calendar view.
4. Issues related to 1. Domain Errors 2. sender domain error 3. Dedicated email isn't sending or working,  can be related Domain and Verify DNS records".
5. Specific If/Else in a question pertains to Workflow Actions - If/Else conditions.
6. Issues related to 1. Meeting or zoom meeting does not appear 2. calendars sync, can be found in Linked Calendars & Conflict Calendars.
7. Conversation AI can be related AI(artificial intelligence) Action
8. Email issues can also be related to SMTPs, Domain and DNS

Based on my findings, analyze the given questions, create your findings for the given question based on the following titles then based on your findings, generate alternate questions more aligned with the domain, incorporating specific language and nuances for better precision in search results.

# Document titles
----------------------------------------------------------------------------------------------------------------
{titles}
----------------------------------------------------------------------------------------------------------------


Original question: {question}

Alternate Questions:
1.
"""
  response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[{'role': 'system', 'content': system_prompt}],
      response_model=Alternate_Questions,
      temperature=0
  )
  return [question]+response.questions



def get_retrive_doc(query, index_retiver, compress_retriever):
  """
    Retrieve documents based on query, returning relevant documents and queries.

    Args:
        query (str): User query.
        index_retriever: Retriever for creating domain knwolege from indexes.
        compress_retriever: Retriever for Advance rag and re-ranking compression.

    Returns:
        tuple: Retrieved documents and all queries.
    """
  
  retrive_doc = []
  check_unique_url = []
  index_docs =  index_retiver.invoke(query)
  index_title = ""
  for i_doc in index_docs:
    if i_doc.metadata['section_index'] != "":
      index_title = index_title + i_doc.metadata['section_index'] + "\n\n\n"
    else:
      index_title = index_title + i_doc.metadata['page_index'] + "\n\n\n"

  all_query = get_alternate_questions(query, index_title)

  for q in all_query:
    compressed_docs = compress_retriever.invoke(q)
    for doc in compressed_docs:
      if doc.metadata['url'] not in check_unique_url:
        check_unique_url.append(doc.metadata['url'])
        retrive_doc.append(doc)
      else:
        for i in range(len(retrive_doc)):
          if retrive_doc[i].metadata['url'] == doc.metadata['url']:
            if retrive_doc[i].metadata['relevance_score'] < doc.metadata['relevance_score']:
              retrive_doc[i].metadata['relevance_score']  = doc.metadata['relevance_score']
            if retrive_doc[i].page_content != doc.page_content: #Same document but different chunk
              retrive_doc[i].page_content = retrive_doc[i].page_content + doc.page_content
              retrive_doc[i].metadata['section_summary'] = retrive_doc[i].metadata['section_summary'] + doc.metadata['section_summary']
              retrive_doc[i].metadata['relevance_score'] = retrive_doc[i].metadata['relevance_score'] + 0.1 #increasing score by 10%

  retrive_doc = sorted(retrive_doc, key=lambda doc: doc.metadata['relevance_score'], reverse=True)
  retrive_doc = retrive_doc[:14] # selecting top 15 based on score
  return retrive_doc, all_query

class AnswerCitation(BaseModel):
    """
    Citations and Answer
    """
    citation: List[int] = Field(description="Always include all the citations. in case of no answer citation=[] ")

    answer: str = Field(description="Only include Answer, do not include any citations in this, that is python list. In case of no answer, answer = 'There is no answer available'")

def qet_ans(queries, docs):
  """
    Generate answer from retrieved documents.

    Args:
        query (list): List of alternate queries.
        docs (list): Retrieved documents.

    Returns:
        tuple: Answer text and citations.
    """
  
  alternate_queries = "\n".join(f"{index + 1}. {item}" for index, item in enumerate(queries[1:]))
  
  context_str = "\n\n\n\n".join(
            f"**documents: {i+1}**\n{context.page_content}" for i, context in enumerate(docs)
        )
  user_query = f"""
Documents
------------------------------------------
{context_str}
------------------------------------------


**Query:**
{queries[0]}
"""

  client = instructor.from_openai(OpenAI())
  
  system_prompt = f"""
INSTRUCTIONS:
1. You are an assistant who helps users answer their queries.
2. Always Answer the user's query from the given documents. The user will provide documents, each identified by a document number.
3. Give answer in step by step format.
4. Keep your answer concise with all required and requested details and solely on the information given in the document.
5. Always provide the answer with all relevant citations at the end of the answer, ensuring that each citation includes the corresponding document number used to create the answer. Provide the citation in the form of python list at the end of the whole answer not in between the answer.
7. Do not create or derive your own answer. If the answer is not directly available in the documents, just reply stating, 'There is no answer available', in case of no answer, citation will be empty list '[]'
8. Note: When providing an answer, reference only the minimum number of documents necessary. Treat each document as complete and independent, prioritizing the most relevant one that directly addresses the query. If multiple documents contain similar or duplicate content, cite only the most appropriate document for the answer.
"""
  response = client.chat.completions.create(
      model="gpt-4o",
      messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_query}],
      response_model=AnswerCitation,
      temperature=0.075
  )
  # print(system_prompt)
  return response.answer, response.citation



def rag_execution(query, index_retiver, compress_retriever):
  """
    Execute the retrieval and answer generation process.
    
    Args:
        query (str): User query.
        index_retriever: Index retriever instance.
        compress_retriever: Compression retriever instance.
    
    Returns:
        tuple: Answer text and citations.
    """
  
  retrive_docs, all_query = get_retrive_doc(query, index_retiver, compress_retriever)
  ans, citation = qet_ans(all_query, retrive_docs)
  actual_citation = []
  for i in range(len(retrive_docs)):
    if i + 1 in citation:
      actual_citation.append({'title': retrive_docs[i].metadata['title'],
                              'url': retrive_docs[i].metadata['url']})

  return ans, actual_citation





embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
persist_directory = "content/crawl/high_level_support_solution_chroma_langchain_db"
compress_retriever, _ = get_retriver(persist_directory, embedding_model)
_, index_retiver = get_retriver(persist_directory, embedding_model, 150)


app = FastAPI()

class Citation(BaseModel):
    title: str
    url: str

class InputData(BaseModel):
    input_text: str

class ResponseData(BaseModel):
    answer: str
    citation: List[Citation]

@app.post("/get_answer")
async def get_answer(data: InputData):
    """
    Endpoint to get answer and citations for a given query.

    Args:
        data (InputData): User query data.

    Returns:
        ResponseData: Answer and citations.
    """
    
    query = data.input_text

    answer, citation =  rag_execution(query, index_retiver, compress_retriever)


    return ResponseData(answer=answer, citation=citation)

