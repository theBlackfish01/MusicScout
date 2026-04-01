# MusicScout 🎵

MusicScout is an intelligent multi-agent orchestration service that combines qualitative music research with quantitative data analysis. It leverages **LangGraph** to coordinate specialized agents and **FastAPI** to provide a robust RESTful interface.

## 🚀 Vision

To provide a seamless workflow for music researchers and analysts. Instead of manually searching for facts and then calculating statistics, MusicScout automates the entire process from identifying artist history to computing average track lengths or chart positions.

## 🤖 System Architecture

The system follows the **Supervisor Pattern**:

- **Supervisor**: Evaluates user queries and routes them to the appropriate worker.
- **Research Agent ("The Musicologist")**: Uses Wikipedia and DuckDuckGo to gather factual data.
- **Analysis Agent ("The Data Analyst")**: Utilizes a Python REPL for arithmetic, statistics, and data structuring.

## 🛠️ Tech Stack

- **Framework**: LangGraph, LangChain
- **API**: FastAPI, Uvicorn
- **LLMs**: OpenAI (GPT-4o), Google Gemini
- **Tools**: Wikipedia API, DuckDuckGo Search, Python REPL

## ⚙️ Setup & Installation

### 1. Prerequisites

- Python 3.11+
- API Keys for OpenAI or Gemini

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd MusicScout

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_key_here
# Optional:
LLM_PROVIDER=openai # or gemini
OPENAI_MODEL=gpt-4o
```

### 4. Running the Service

```bash
uvicorn src.main:app --reload
```

## 📡 API Usage

### Execute a Query

**Endpoint**: `POST /v1/execute`

**Request Body**:

```json
{
  "query": "Find the top 5 longest tracks by Pink Floyd and calculate their average length in minutes.",
  "thread_id": "session-123"
}
```

**Response**:

```json
{
  "answer": "The average length of Pink Floyd's top 5 longest tracks is...",
  "total_tokens": 1250,
  "prompt_tokens": 800,
  "completion_tokens": 450,
  "total_cost_usd": 0.012
}
```

## 🧪 Testing

Run the test suite using `pytest`:

```bash
pytest tests/
```

## 📄 License

This project is licensed under the MIT License.
