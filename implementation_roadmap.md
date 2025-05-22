# Implementation Roadmap

## Task 1: AI-Driven Task Assignment System

### Subtasks:

#### Data Preprocessing

Install required packages: scikit-learn, numpy

Create skill vectorizer using TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def create_skill_vectors(employees):
    skills_text = [' '.join(emp['skills']) for emp in employees]
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(skills_text)
```

#### KNN Implementation

```python
from sklearn.neighbors import NearestNeighbors

def train_knn(employee_vectors):
    knn = NearestNeighbors(n_neighbors=3, metric='cosine')
    knn.fit(employee_vectors)
    return knn
```

#### Assignment Algorithm

```python
def assign_tasks(knn_model, tasks, employees):
    assignments = {}
    for task in tasks:
        # Note: 'vectorizer' needs to be accessible in this function's scope.
        # It's used here based on the original text but not explicitly passed as an argument.
        task_vector = vectorizer.transform([' '.join(task['required_skills'])])
        _, indices = knn_model.kneighbors(task_vector)

        for idx in indices[0]:
            emp = employees[idx]
            if emp['current_load'] + task['estimated_hours'] <= emp['capacity']:
                assignments[task['task_id']] = emp['employee_id']
                emp['current_load'] += task['estimated_hours']
                break
    return assignments
```

#### Capacity-aware Scheduling

- Implement load balancing using greedy algorithm
- Add fallback mechanism for overload scenarios

## Task 2: RAG-Powered Task Decomposition

### Subtasks:

#### Document Processing

```python
from langchain.document_loaders import TextLoader
from llama_index import SimpleDirectoryReader # Assuming this is the intended import, though llama_index v0.10+ uses llama_index.core

def process_company_docs(docs_path):
    return SimpleDirectoryReader(docs_path).load_data()
```

#### Vector Store Setup

```python
from langchain.embeddings import OpenAIEmbeddings # Assuming older langchain or specific OpenAIEmbeddings class
# For newer langchain, it might be: from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def create_vector_store(docs):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)
```

#### Context-Aware Decomposition

```python
def decompose_with_rag(prd_text, vector_store):
    # Retrieve relevant context
    docs = vector_store.similarity_search(prd_text, k=3)
    context = "\n".join([d.page_content for d in docs]) # Note: Escaped newline for f-string

    # LLM Prompt (Assuming 'llm' is a pre-configured language model instance)
    prompt = f"""
    Given this project requirement:
    {prd_text}

    And company context:
    {context}

    Break down into tasks with:
    - Required skills
    - Estimated hours
    - Dependencies
    """
    # return llm.generate(prompt) # Original, .generate might be for older/specific LLM interface
    # For OpenAI models via Langchain, it might be llm.invoke(prompt) or similar
    # Keeping as is for fidelity to original text.
    return llm.generate(prompt)
```

## Task 3: Validation Framework

### Metrics Implementation:

```python
import time # Added import for time.time()

def calculate_accuracy(auto_tasks, manual_tasks):
    matches = 0
    for auto, manual in zip(auto_tasks, manual_tasks):
        # Assuming tasks are dicts and skills are comparable lists/sets
        if auto['required_skills'] == manual['required_skills']: # Or set(auto['skills']) == set(manual['skills'])
            matches +=1
    return matches/len(auto_tasks) if len(auto_tasks) > 0 else 0 # Avoid division by zero

def measure_planning_time():
    start = time.time()
    # Run planning process (placeholder for actual planning logic)
    # Example: generated_tasks = decompose_with_rag(prd_sample, vector_store_sample)
    #          assigned_tasks = assign_tasks(knn_model_sample, generated_tasks, employees_sample)
    end = time.time()
    return end - start
```

### 3. Step-by-Step Implementation Plan

| Phase              | Tasks                                    | Subtasks                                                                                                        | Duration |
| ------------------ | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------- | -------- |
| 1. Setup           | - Environment Configuration              | 1. Install Python 3.10+ <br> 2. Set up virtual environment <br> 3. Install required packages (requirements.txt) | 1 day    |
| 2. Data Generation | - Synthetic Data Creation                | 1. Implement employee generator <br> 2. Create task templates <br> 3. Generate sample PRDs                      | 2 days   |
| 3. Core Algorithms | - KNN Implementation <br> - RAG Pipeline | 1. Skill vectorization <br> 2. Nearest neighbor search <br> 3. Document indexing <br> 4. Context retrieval      | 5 days   |
| 4. Integration     | - Trello Sync <br> - Capacity Management | 1. API authentication setup <br> 2. Webhook configuration <br> 3. Load balancing integration                    | 3 days   |
| 5. Testing         | - Unit Tests <br> - E2E Testing          | 1. Accuracy benchmarks <br> 2. Stress testing <br> 3. User acceptance testing                                   | 3 days   |
| 6. Deployment      | - Dockerization <br> - CI/CD Setup       | 1. Containerize components <br> 2. GitHub Actions workflow <br> 3. Monitoring setup                             | 2 days   |

### 4. Validation Checklist

#### Task Assignment Validation:

- 95% of tasks assigned have ≥1 matching skill
- No employee exceeds 90% capacity
- Assignment time <2s per task

#### RAG Decomposition Validation:

- 80% of generated
  (Note: The RAG Decomposition Validation checklist seems to be incomplete in the provided text.)
