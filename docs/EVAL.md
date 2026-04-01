# MusicScout Evaluation Report (EVAL.md)

## 1. Project Overview & Agent Personas

MusicScout is a multi-agent system designed to solve queries that require a combination of up-to-date factual retrieval and deterministic mathematical computation. Standard LLMs typically struggle with this, either hallucinating math or relying on outdated training data.

To solve this, MusicScout employs a Supervisor Architecture using LangGraph to orchestrate two specialized workers:

* **ResearchAgent:** Equipped with `DuckDuckGoSearchRun` and `WikipediaQueryRun`. Its persona is a strict fact-finding data pipeline. It is explicitly instructed to avoid answering from memory and to retrieve raw data formats (e.g., MM:SS for runtimes, exact dates).
* **AnalysisAgent:** Equipped with a `PythonREPL` tool. Its persona is a deterministic calculator. By offloading arithmetic to a Python environment, it ensures 100% accuracy on averages, differences, and statistical queries based on the ResearchAgent's findings.

---

## 2. Feature Implementation & Evaluation

### State Persistence

The system utilizes LangGraph's `SqliteSaver` coupled with a physical `checkpoints.sqlite` database to ensure state persistence across API lifecycles.

**Testing Persistence:** This can be verified by simulating a server interruption:

1. Send a query via `POST /v1/execute` using a unique `thread_id` (e.g., `thread-123`).
2. Stop the FastAPI server completely.
3. Restart the server.
4. Send a follow-up query (e.g., "What was the first album you mentioned?") using the *same* `thread_id`. The Supervisor will successfully read the historical state from the database and answer without re-triggering the ResearchAgent.

### Trace Logging Integration

To ensure the multi-agent orchestration is entirely observable without requiring a LangSmith dashboard, the FastAPI endpoint extracts and surfaces the execution trace natively. The system iterates through the final state array, mapping `AIMessage` tool calls to their corresponding `ToolMessage` outputs, and returns this step-by-step trace in the JSON response payload.

---

## 3. Architectural Evolution & LLM Failure Analysis

During development, the system exhibited several classic "agentic failure modes." Diagnosing and engineering around these failures was critical to achieving a stable production state.

* **Failure Mode 1: The "Eager Router" (Premature Delegation)**
  * *The Issue:* For a query like "Calculate the exact difference in days between two albums," the Supervisor would read the word "Calculate" and immediately route to the AnalysisAgent, skipping the ResearchAgent. The AnalysisAgent would fail because it lacked the raw dates to compute.
  * *The Fix:* I updated the Supervisor's routing logic to be dependency-aware. The system prompt now explicitly forces the Supervisor to verify if the prerequisite raw data exists in the conversation history before dispatching the AnalysisAgent.

* **Failure Mode 2: The "Silent Handoff" Loop**
  * *The Issue:* The ResearchAgent would sometimes end its turn conversationally: "Here is the data. Would you like me to calculate the average?" When the Supervisor routed this to the AnalysisAgent, the AnalysisAgent's LLM assumed it was the human's turn to speak and yielded an empty string (`""`), causing an infinite routing loop between the Supervisor and the silent agent.
  * *The Fix:* I instructed the ResearchAgent to act purely as a data pipeline with no conversational follow-ups. Furthermore, I injected a programmatic "nudge" (`HumanMessage`) into the AnalysisAgent's context window, forcing it to recognize it was time to execute code.

* **Failure Mode 3: The "Perfectionist Rabbit Hole" (Data Starvation)**
  * *The Issue:* When researching ambiguous trivia (e.g., whether Pink Floyd's 26-minute *Shine On You Crazy Diamond* counts as one track or two), the ResearchAgent would get confused by conflicting search snippets. It would endlessly tweak its search queries looking for a definitive answer until the LangGraph recursion limit (step 30) crashed the API.
  * *The Fix:* I implemented a two-factor guardrail. First, an "Anti-Rabbit Hole" prompt directive giving the agent permission to use best-effort estimates if exact data is buried. Second, a Deterministic Circuit Breaker in the LangGraph conditional edge that counts consecutive tool calls and programmatically intercepts execution if an agent gets trapped in a search loop.

* **Failure Mode 4: The "Stubborn Supervisor"**
  * *The Issue:* If the AnalysisAgent threw an error asking for more data, the Supervisor—knowing the data was already in the chat—would argue with its worker and stubbornly re-route the task back to it, causing another infinite loop.
  * *The Fix:* I implemented deterministic overrides in the Supervisor node. If an agent outputs a hardcoded `SYSTEM ERROR` string, the Python routing logic intercepts it and forces a `FINISH` route, ensuring the system never relies solely on LLM compliance to break a death loop.

---

## 4. System Optimization: Handling User Ambiguity

While the system is robust against ambiguous data retrieval (via the "Anti-Rabbit Hole" directive), handling ambiguous *user prompts* requires a specific Supervisor-level optimization.

Currently, if a user submits a vague query like *"Pink Floyd and Led Zeppelin"*, the Supervisor is forced to guess the intent, often resulting in a hallucinated delegation. To handle this more effectively, the Supervisor's system prompt would be upgraded with a **"Reject and Clarify"** (Human-in-the-Loop) directive.

Instead of blindly dispatching workers, the Supervisor would be instructed to evaluate the request for actionable intent. If the query lacks a clear research goal or mathematical operation, the Supervisor bypasses the agents, routes directly to `FINISH`, and outputs a clarifying question (e.g., *"Are you looking to compare their total runtimes, average ages, or Grammy wins?"*). This prevents wasted API calls and enforces a deterministic intent-gathering phase.

---

## 5. Final Stress Test Results

After implementing the guardrails discussed above, the system was evaluated against 5 complex, multi-hop queries.

**Success Metrics:**

* **Accuracy:** Does the answer correctly reflect the facts and calculations?
* **Efficiency:** Did it complete within the recursion limit of 30 steps?
* **Reliability:** Did the state persist and route correctly?

| Query ID | Description | Status | Latency | Tokens | Cost (USD) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `test-beatles-001` | Beatles average age at Sgt. Pepper release | SUCCESS | 44s | 10,598 | 0.00 |
| `test-floyd-002` | Top 3 longest Pink Floyd tracks & total runtime | SUCCESS | 95s | 34,647 | 0.00 |
| `test-timeline-003`| Sum of release years for first 3 Led Zeppelin albums | SUCCESS | 40s | 4,163 | 0.00 |
| `test-grammys-004` | Beyoncé/Taylor/Adele Grammy weighted score | SUCCESS | 116s | 38,290 | 0.00 |
| `test-tempo-005` | BPM of Michael Jackson's 'Thriller' vs 'Bad' | SUCCESS | 30s | 4,633 | 0.00 |

### Detailed Execution Breakdown

* **[test-beatles-001] Beatles Average Age**
  * *Flow:* Supervisor -> ResearchAgent (finds 4 distinct birthdays and 1 release date) -> Supervisor -> AnalysisAgent (writes datetime script to calculate exact days lived, converts to years, and averages) -> Supervisor -> FINISH.
  * *Result:* Successfully calculated the average age as ~25 years old.

* **[test-floyd-002] Longest Pink Floyd Tracks**
  * *Flow:* Successfully navigated the "Perfectionist Rabbit Hole." The ResearchAgent identified Atom Heart Mother (23:41), Echoes (23:31), and Shine On You Crazy Diamond I-V (13:32). The AnalysisAgent cleanly parsed the MM:SS strings, converted them to seconds, and output the total runtime.

* **[test-timeline-003] Led Zeppelin Album Year Summation**
  * *Flow:* Correctly identified Led Zeppelin (1969), Led Zeppelin II (1969), and Led Zeppelin III (1970) as the first three albums and summed their release years (5908) successfully.

* **[test-grammys-004] Grammy Weighted Score**
  * *Flow:* Correctly fetched Grammy counts (e.g., 35 for Beyoncé) and debut years (2003 for Beyoncé) and performed the custom weighted calculation (10 points per Grammy minus 1 point per year since debut). Declared Beyoncé the winner.

* **[test-tempo-005] BPM Comparison (MJ)**
  * *Flow:* Bypassed the "Eager Router" flaw. Supervisor waited for the ResearchAgent to pull the raw BPM data for Thriller (118 BPM) and Bad (114 BPM) before passing control to the AnalysisAgent to perform the quantitative comparison.
