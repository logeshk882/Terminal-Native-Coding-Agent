# TMCA --- Terminal-Native Coding Assistant

## Detailed Architecture, Implementation Plan, and Learning Roadmap

> **Goal:** Build a terminal-native coding assistant from first
> principles in Python. The project should teach the underlying
> mechanics of LLM APIs, tool calling, agent loops, repository
> understanding, safe execution, terminal UI, MCP, observability, and
> evaluation rather than hiding them behind an agent framework.

------------------------------------------------------------------------

# 1. Project Objective

TMCA is a terminal-native AI coding assistant that can:

-   understand natural-language coding tasks
-   inspect a local repository
-   search and retrieve relevant code
-   read and edit files
-   generate and apply patches
-   execute tests and shell commands
-   inspect command/test results
-   iterate when an implementation fails
-   use external documentation/web resources
-   operate inside a sandbox
-   maintain session and task state
-   expose an interactive terminal UI
-   record traces, tokens, latency, tool calls, and errors
-   optionally create Git commits and GitHub pull requests

The most important learning objective is to understand the complete
loop:

``` text
User
  ↓
Terminal UI
  ↓
Agent Runtime
  ↓
Context Retrieval
  ↓
LLM
  ↓
Tool Call
  ↓
Schema Validation
  ↓
Permission Check
  ↓
Tool Execution
  ↓
Observation
  ↓
Agent State Update
  ↓
LLM
  ↓
Repeat until verified
  ↓
Final Response
```

------------------------------------------------------------------------

# 2. Recommended Python Technology Stack

## Core

  -------------------------------------------------------------------------
  Layer                   Technology              Purpose
  ----------------------- ----------------------- -------------------------
  Language                Python 3.12+            Main implementation

  CLI                     Typer                   Command-line commands and
                                                  options

  TUI                     Textual                 Interactive terminal
                                                  application

  Rendering               Rich                    Markdown, syntax
                                                  highlighting, tables,
                                                  diffs

  HTTP                    HTTPX                   Raw asynchronous API
                                                  communication

  Data models             Pydantic                Tool schemas,
                                                  configuration, validation

  Async                   asyncio                 Concurrent/long-running
                                                  work

  Configuration           python-dotenv +         Local
                          environment variables   configuration/secrets

  Tests                   pytest                  Unit and integration
                                                  testing

  Storage                 SQLite                  Sessions, messages, task
                                                  state, traces

  Search                  ripgrep                 Fast lexical code search

  Parsing                 Tree-sitter             AST/symbol-aware code
                                                  intelligence

  Retrieval               BM25 + embeddings       Repository-aware
                                                  retrieval

  Browser                 Playwright              Documentation/web
                                                  interaction

  Sandbox                 Docker initially;       Isolated command
                          E2B/Daytona later       execution

  Protocol                MCP Python SDK          External tool
                                                  interoperability

  Git                     Git CLI initially       Repository operations

  Observability           OpenTelemetry +         Agent traces and metrics
                          Langfuse                
  -------------------------------------------------------------------------

Do not install every dependency at the beginning. Add technologies when
their corresponding milestone is reached.

------------------------------------------------------------------------

# 3. Design Principles

## 3.1 Build primitives before frameworks

Initially avoid LangChain, LangGraph, CrewAI, or other agent
orchestration frameworks.

Implement these yourself:

``` text
LLMProvider
Message
LLMResponse
Tool
ToolSchema
ToolRegistry
ToolExecutor
AgentState
AgentLoop
ContextManager
PermissionManager
```

After understanding them, study frameworks and compare their
abstractions to your implementation.

## 3.2 Separate interfaces from implementations

The agent should depend on:

``` text
LLMProvider
```

rather than:

``` text
OpenRouterProvider
```

Likewise:

``` text
Tool
```

rather than:

``` text
specific filesystem implementation
```

This allows providers and tools to be swapped without rewriting the
agent.

## 3.3 Treat the model as an unreliable planner

The LLM can:

-   misunderstand a task
-   select the wrong tool
-   produce invalid arguments
-   hallucinate files
-   write incorrect code
-   stop before verification

Therefore every important action should be grounded by tools and real
observations.

## 3.4 Make execution observable

Every meaningful agent step should be traceable:

``` text
run_id
step_id
timestamp
model
input tokens
output tokens
tool name
arguments
result
latency
exit code
error
```

## 3.5 Make destructive actions explicit

The agent should not have unrestricted host access.

Use:

``` text
Policy
  ↓
Risk classification
  ↓
Approval / sandbox
  ↓
Execution
```

------------------------------------------------------------------------

# 4. Target Architecture

``` text
                         ┌──────────────────────┐
                         │        USER          │
                         └──────────┬───────────┘
                                    ↓
                         ┌──────────────────────┐
                         │   Python CLI / TUI    │
                         │    Typer + Textual    │
                         │        + Rich         │
                         └──────────┬───────────┘
                                    ↓
                         ┌──────────────────────┐
                         │    Agent Runtime     │
                         │                      │
                         │ State / Loop / Plan  │
                         │ Act / Observe / Stop │
                         └──────────┬───────────┘
                                    ↓
                         ┌──────────────────────┐
                         │   Context Engine     │
                         │                      │
                         │ ripgrep / BM25       │
                         │ Tree-sitter          │
                         │ embeddings / rerank  │
                         └──────────┬───────────┘
                                    ↓
                         ┌──────────────────────┐
                         │     LLM Gateway      │
                         │                      │
                         │ OpenAI               │
                         │ Anthropic            │
                         │ OpenRouter            │
                         │ Ollama/local         │
                         └──────────┬───────────┘
                                    ↕
                         ┌──────────────────────┐
                         │   Tool Dispatcher    │
                         │ Registry / Validation│
                         │ Permissions / Hooks  │
                         └──────────┬───────────┘
                                    ↓
             ┌──────────────────────┼──────────────────────┐
             ↓                      ↓                      ↓
       Filesystem               Search                   Shell
       read/write/patch         ripgrep                  git/run
             │                      │                      │
             └──────────────────────┼──────────────────────┘
                                    ↓
                         ┌──────────────────────┐
                         │   Sandbox Runtime    │
                         │ Docker / E2B /        │
                         │ Daytona              │
                         └──────────┬───────────┘
                                    ↓
                              Tests / Runtime
                                    ↓
                               Observation
                                    ↓
                              Agent Runtime

        ┌────────────────────────────────────────────────────┐
        │ Cross-cutting systems                              │
        │                                                    │
        │ Security │ Hooks │ SQLite │ OpenTelemetry │ GitHub │
        └────────────────────────────────────────────────────┘
```

------------------------------------------------------------------------

# 5. Recommended Repository Structure

Start with a small structure and grow into this:

``` text
TMCA/
├── .venv/
├── src/
│   └── tmca/
│       ├── __init__.py
│       ├── main.py
│       │
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── agent.py
│       │   ├── loop.py
│       │   ├── state.py
│       │   ├── planner.py
│       │   └── prompts.py
│       │
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── openrouter.py
│       │   ├── openai.py
│       │   ├── anthropic.py
│       │   └── ollama.py
│       │
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── registry.py
│       │   ├── executor.py
│       │   ├── schemas.py
│       │   └── builtin/
│       │       ├── __init__.py
│       │       ├── filesystem.py
│       │       ├── search.py
│       │       ├── shell.py
│       │       └── git.py
│       │
│       ├── context/
│       │   ├── __init__.py
│       │   ├── manager.py
│       │   ├── scanner.py
│       │   ├── chunker.py
│       │   ├── retriever.py
│       │   ├── ranker.py
│       │   └── tokenizer.py
│       │
│       ├── security/
│       │   ├── __init__.py
│       │   ├── permissions.py
│       │   ├── policies.py
│       │   ├── paths.py
│       │   └── secrets.py
│       │
│       ├── tui/
│       │   ├── __init__.py
│       │   ├── app.py
│       │   ├── chat.py
│       │   ├── file_browser.py
│       │   ├── editor.py
│       │   ├── diff_viewer.py
│       │   └── shell_panel.py
│       │
│       ├── browser/
│       │   ├── __init__.py
│       │   ├── search.py
│       │   └── browser.py
│       │
│       ├── mcp/
│       │   ├── __init__.py
│       │   ├── client.py
│       │   └── adapters.py
│       │
│       ├── sandbox/
│       │   ├── __init__.py
│       │   └── runtime.py
│       │
│       ├── hooks/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── registry.py
│       │   └── builtin.py
│       │
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── database.py
│       │   └── models.py
│       │
│       ├── observability/
│       │   ├── __init__.py
│       │   ├── logger.py
│       │   ├── tracing.py
│       │   └── metrics.py
│       │
│       └── github/
│           ├── __init__.py
│           └── pull_request.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── .env
├── .gitignore
├── pyproject.toml
├── README.md
└── docs/
```

Do not create all of these files on day one.

------------------------------------------------------------------------

# 6. Phase 0 --- Development Environment

## Objective

Create a reproducible Python project.

Recommended commands:

``` powershell
mkdir TMCA
cd TMCA

python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install httpx pydantic typer rich python-dotenv
```

Later add dependencies by phase.

## `.gitignore`

``` gitignore
.venv/
.env
__pycache__/
.pytest_cache/
*.pyc
.tmca/
```

## Environment variables

``` env
OPENROUTER_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

Never commit secrets.

------------------------------------------------------------------------

# 7. Phase 1 --- LLM Provider Foundation

## Objective

Understand an LLM API before building an agent.

Start with:

``` text
User input
  ↓
Message
  ↓
Provider
  ↓
HTTP request
  ↓
Model
  ↓
HTTP response
  ↓
Normalized response
```

## `llm/base.py`

Use a small internal representation:

``` python
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Message:
    role: str
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str | None = None


class LLMProvider(ABC):

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
    ) -> LLMResponse:
        raise NotImplementedError
```

### Why `dataclass`?

`Message` and `LLMResponse` are structured data containers.

Instead of repeatedly passing unstructured dictionaries:

``` python
{"role": "user", "content": "hello"}
```

you can use:

``` python
Message(role="user", content="hello")
```

The model objects can later grow to contain:

``` text
Message
├── role
├── content
├── tool_calls
├── tool_call_id
└── metadata
```

### Why `ABC`?

`LLMProvider` defines a contract.

Every provider must implement:

``` python
chat(messages)
```

This allows:

``` text
LLMProvider
├── OpenRouterProvider
├── OpenAIProvider
├── AnthropicProvider
└── OllamaProvider
```

without changing the agent.

`ABC` is not mandatory. `typing.Protocol` is another valid design. Use
`ABC` initially because it makes the interface/implementation
relationship explicit while learning.

------------------------------------------------------------------------

# 8. Phase 1.1 --- Raw API Communication

Initially use `httpx` rather than a provider SDK.

Learn:

-   HTTP methods
-   URL
-   headers
-   authentication
-   JSON payloads
-   status codes
-   timeouts
-   response parsing

Conceptual request:

``` text
POST /provider-endpoint

Headers:
Authorization: Bearer <key>
Content-Type: application/json

Body:
{
  "model": "...",
  "messages": [...]
}
```

Provider-specific request formats can change. Treat the provider's
current API documentation as authoritative when implementing the
concrete adapter.

## Error categories

Implement distinct handling for:

``` text
4xx
├── authentication
├── invalid request
└── rate limit

5xx
├── provider failure
└── temporary outage

network
├── timeout
├── connection failure
└── DNS/TLS errors
```

Do not blindly retry every error.

------------------------------------------------------------------------

# 9. Phase 1.2 --- Streaming

After ordinary requests work, implement streaming.

Desired flow:

``` text
LLM
 ↓
chunk
 ↓
renderer
 ↓
terminal
```

Your provider interface can eventually expose an async stream:

``` python
async def stream(
    self,
    messages: list[Message],
):
    ...
```

The TUI should render chunks as they arrive.

Learn:

-   async generators
-   streaming HTTP responses
-   buffering
-   cancellation
-   partial responses
-   connection failures during streams

------------------------------------------------------------------------

# 10. Phase 1.3 --- Conversation State

Create an `AgentState`.

At minimum:

``` text
session_id
messages
current_task
tool_calls
observations
status
iteration
```

A session might look like:

``` text
SYSTEM
USER
ASSISTANT
TOOL CALL
TOOL RESULT
ASSISTANT
TOOL CALL
TOOL RESULT
ASSISTANT
```

The state should be independent from the UI.

------------------------------------------------------------------------

# 11. Phase 2 --- Tool Calling

This is the first major agent milestone.

## Fundamental concept

A tool is an operation that the model can request but your application
executes.

Example:

``` text
LLM:
I need to inspect main.py.

Tool call:
read_file
arguments:
{
  "path": "main.py"
}
```

Your runtime:

``` text
Tool call
   ↓
parse
   ↓
identify tool
   ↓
validate arguments
   ↓
permission check
   ↓
execute
   ↓
structured result
   ↓
append result to conversation
   ↓
LLM
```

The model does not directly execute Python functions. Your runtime
interprets the model's request.

------------------------------------------------------------------------

# 12. Tool Schema

Define each tool with:

``` text
name
description
input schema
executor
permission requirements
```

Example conceptual schema:

``` json
{
  "name": "read_file",
  "description": "Read a UTF-8 text file from the workspace.",
  "parameters": {
    "type": "object",
    "properties": {
      "path": {
        "type": "string"
      }
    },
    "required": ["path"]
  }
}
```

Use Pydantic to validate arguments internally.

For example:

``` python
from pydantic import BaseModel


class ReadFileArgs(BaseModel):
    path: str
```

If the model sends:

``` json
{
  "path": 123
}
```

the validation layer rejects or coerces it according to your configured
validation policy rather than letting malformed input reach the
filesystem.

------------------------------------------------------------------------

# 13. Tool Registry

Do not write:

``` python
if tool_name == "read_file":
    ...
elif tool_name == "write_file":
    ...
```

Instead:

``` text
ToolRegistry
├── read_file
├── write_file
├── search_files
├── apply_patch
├── run_command
└── git
```

Each registered tool provides:

``` text
metadata
schema
validator
executor
policy
```

This allows tools to be dynamically added and removed.

------------------------------------------------------------------------

# 14. Tool Executor

Separate registry from execution.

``` text
ToolRegistry
     ↓
find tool
     ↓
ToolExecutor
     ↓
validate
     ↓
authorize
     ↓
execute
     ↓
normalize result
```

The executor should return structured observations.

Example shell result:

``` json
{
  "exit_code": 1,
  "stdout": "...",
  "stderr": "AssertionError...",
  "duration_ms": 2310,
  "timed_out": false
}
```

This is much better than returning a single text string because the
agent can reason over structured execution state.

------------------------------------------------------------------------

# 15. Phase 3 --- Filesystem Tools

Implement in this order.

## `list_directory`

Purpose:

``` text
Understand project structure
```

## `read_file`

Purpose:

``` text
Inspect source code
```

## `search_files`

Initially delegate to `ripgrep`.

Example:

``` text
search_files("authenticate", "src/")
```

## `write_file`

Use cautiously.

## `apply_patch`

Prefer patch-based modifications over complete-file replacement.

------------------------------------------------------------------------

# 16. Safe Path Handling

Every filesystem tool must operate relative to a defined workspace.

Example:

``` text
workspace/
├── src/
├── tests/
└── README.md
```

If the model asks for:

``` text
../../.ssh/id_rsa
```

the path policy should reject it.

Normalize paths and ensure the resolved path remains inside the
workspace.

------------------------------------------------------------------------

# 17. Phase 4 --- Shell Execution

Implement:

``` text
run_command(command)
```

The execution pipeline:

``` text
LLM request
   ↓
command parser
   ↓
risk policy
   ↓
approval?
   ↓
sandbox
   ↓
subprocess
   ↓
stdout/stderr
   ↓
exit code
   ↓
observation
```

Capture:

``` text
command
stdout
stderr
exit code
duration
timeout
```

Support cancellation and timeouts.

------------------------------------------------------------------------

# 18. Command Safety

Classify commands.

Example policy:

``` text
LOW RISK
├── pytest
├── ruff
├── mypy
├── git status
└── git diff

MEDIUM RISK
├── pip install
├── npm install
├── git checkout
└── network commands

HIGH RISK
├── rm
├── git reset --hard
├── destructive database commands
└── arbitrary system modification
```

Never assume this classification is perfect. Make the policy
configurable.

------------------------------------------------------------------------

# 19. Phase 5 --- The Agent Loop

Now combine the LLM and tools.

Basic loop:

``` text
while task_not_complete:

    send current state to LLM

    if final response:
        stop

    if tool calls:
        for each tool call:
            validate
            authorize
            execute
            append observation

    increment iteration
```

Conceptually:

``` python
while not state.finished:

    response = await llm.chat(state.messages)

    if response.is_final:
        state.finished = True
        break

    for tool_call in response.tool_calls:
        result = await executor.execute(tool_call)
        state.add_observation(result)
```

This is the core of your agent.

------------------------------------------------------------------------

# 20. Plan / Act / Observe

Once basic tool calling works, formalize the loop.

``` text
PLAN
  ↓
ACT
  ↓
OBSERVE
  ↓
UPDATE STATE
  ↓
PLAN
```

Example:

``` text
Task:
Fix authentication tests.

PLAN:
1. Find authentication code.
2. Inspect failing test.
3. Determine cause.
4. Patch code.
5. Run tests.

ACT:
search_files("authenticate")

OBSERVE:
src/auth.py
tests/test_auth.py

ACT:
read_file("src/auth.py")

OBSERVE:
...

ACT:
apply_patch(...)

OBSERVE:
Patch applied.

ACT:
run_command("pytest tests/test_auth.py")

OBSERVE:
FAILED

PLAN UPDATE:
Investigate token expiry behavior.

...
```

Do not implement hidden chain-of-thought storage. Store concise,
operational state such as plans, actions, observations, errors, and
outcomes.

------------------------------------------------------------------------

# 21. Stop Conditions

The agent must know when to stop.

Possible successful stop:

``` text
requested task complete
tests pass
required artifact produced
```

Possible failure stop:

``` text
maximum iterations
repeated identical failures
permission denied
sandbox failure
provider failure
budget exceeded
```

Example:

``` text
MAX_ITERATIONS = 20
```

Never allow an unbounded loop.

------------------------------------------------------------------------

# 22. Phase 6 --- Repository Context

Now solve the large-codebase problem.

Do not immediately use vector embeddings.

Start:

``` text
ripgrep
   ↓
relevant files
   ↓
read files
   ↓
LLM
```

Then add:

``` text
BM25
   ↓
ranked lexical retrieval
```

Then:

``` text
Tree-sitter
   ↓
symbols / functions / classes
```

Finally:

``` text
embeddings
   ↓
semantic retrieval
```

------------------------------------------------------------------------

# 23. Repository-Aware Retrieval Architecture

``` text
Repository
    ↓
Scanner
    ↓
Ignore rules
    ↓
File metadata
    ↓
Chunking
    ↓
┌───────────────┬─────────────────┬─────────────────┐
│ lexical index │ symbol index    │ vector index    │
│ ripgrep/BM25  │ Tree-sitter     │ embeddings      │
└───────────────┴─────────────────┴─────────────────┘
                       ↓
                 Query analysis
                       ↓
                Hybrid retrieval
                       ↓
                   Reranking
                       ↓
                 Context budget
                       ↓
                      LLM
```

------------------------------------------------------------------------

# 24. Why Hybrid Retrieval?

Code search benefits from exact matches and semantic similarity.

Example:

User asks:

``` text
Where is user authentication handled?
```

Exact search can find:

``` text
authenticate
login
JWT
middleware
```

Semantic retrieval can find code that performs authentication without
using the exact word.

Tree-sitter can identify:

``` text
class AuthMiddleware
function validate_token
function login
```

Combining these signals is usually more useful than relying on only one
retrieval method.

------------------------------------------------------------------------

# 25. Context Manager

The Context Manager should answer:

> What information should be included in the next LLM request?

Inputs:

``` text
user task
current plan
previous observations
repository search results
relevant files
tool results
```

Outputs:

``` text
bounded model context
```

It should enforce:

``` text
token budget
file budget
tool-output budget
recency
relevance
deduplication
```

------------------------------------------------------------------------

# 26. Context Compaction

Long agent runs eventually exceed the context window.

Use:

``` text
full history
   ↓
identify old low-value messages
   ↓
summarize
   ↓
retain important state
   ↓
continue
```

Preserve:

``` text
task
current plan
important files
completed changes
known errors
test results
pending actions
```

Do not blindly summarize away important tool results.

------------------------------------------------------------------------

# 27. Phase 7 --- Tree-sitter Code Intelligence

Tree-sitter should eventually provide:

``` text
functions
classes
methods
imports
definitions
references
syntax structure
```

Instead of retrieving arbitrary line ranges, retrieve meaningful units:

``` text
function authenticate(...)
class AuthService
method validate_token(...)
```

This improves context quality.

Later, add language-server integration if deeper semantic navigation is
needed.

------------------------------------------------------------------------

# 28. Phase 8 --- Planning and Self-Correction

Introduce explicit task state:

``` text
Task
├── goal
├── plan
├── current_step
├── completed_steps
├── failed_steps
├── observations
└── verification_status
```

Self-correction should be grounded in external feedback.

Strong feedback:

``` text
compiler error
test failure
lint error
runtime exception
HTTP response
```

Weak feedback:

``` text
model says "this looks correct"
```

Prefer executable verification whenever possible.

------------------------------------------------------------------------

# 29. Verification Loop

For coding tasks:

``` text
Generate change
    ↓
Apply patch
    ↓
Run formatter/linter
    ↓
Run targeted tests
    ↓
Run broader tests
    ↓
Inspect failures
    ↓
Patch again if needed
    ↓
Repeat
```

A task should not be marked complete merely because the model claims it
is complete.

------------------------------------------------------------------------

# 30. Phase 9 --- Security Architecture

Security is a first-class subsystem.

``` text
                  Tool Request
                       ↓
                Schema Validation
                       ↓
                 Path Validation
                       ↓
                Permission Policy
                       ↓
                 Risk Assessment
                       ↓
             ┌─────────┴─────────┐
             ↓                   ↓
          Allowed             Approval
             ↓                   ↓
          Sandbox              User
             └─────────┬─────────┘
                       ↓
                    Execute
```

Protect:

-   filesystem
-   environment variables
-   API keys
-   SSH keys
-   cloud credentials
-   network access
-   destructive commands
-   arbitrary code execution

Treat repository instructions and external web content as untrusted
input. A malicious file or webpage can contain instructions designed to
manipulate an agent.

------------------------------------------------------------------------

# 31. Sandbox

Initial development:

``` text
local workspace
+
strict permission layer
```

Production-style execution:

``` text
Host
  ↓
Agent
  ↓
Sandbox
  ├── isolated filesystem
  ├── non-root user
  ├── CPU limit
  ├── memory limit
  ├── timeout
  └── controlled network
```

Start with Docker to learn the concept. Later evaluate E2B or Daytona if
you need managed execution environments.

------------------------------------------------------------------------

# 32. Phase 10 --- MCP

Only add MCP after your own tool system works.

First:

``` text
Agent
 ↓
Tool Registry
 ↓
Tool Executor
```

Then:

``` text
Agent
 ↓
MCP Client
 ↓
MCP Server
 ↓
External Tool
```

MCP should be an interoperability layer, not a replacement for learning
tool calling.

Your internal architecture can support:

``` text
BuiltinTool
MCPTool
RemoteTool
```

through one normalized interface.

------------------------------------------------------------------------

# 33. Phase 11 --- Terminal UI

Use:

``` text
Typer
+
Textual
+
Rich
```

Typer handles commands such as:

``` bash
tmca
tmca "fix the tests"
tmca --model ...
tmca --resume ...
```

Textual handles the interactive application.

Rich handles:

-   Markdown
-   syntax highlighting
-   tables
-   panels
-   progress
-   diffs
-   formatted errors

------------------------------------------------------------------------

# 34. Suggested TUI

``` text
┌──────────────────────────────────────────────────────────────┐
│ TMCA        project: myapp        model: selected-model      │
├──────────────────────┬───────────────────────────────────────┤
│ FILES                │ CODE / DIFF                          │
│                      │                                       │
│ ▼ src                │ auth.py                              │
│   ▼ auth             │                                       │
│     auth.py          │ def authenticate(...):               │
│     middleware.py    │     ...                              │
│   ▼ api              │                                       │
│                      │                                       │
├──────────────────────┴───────────────────────────────────────┤
│ AGENT                                                      │
│ Searching for authentication implementation...              │
│ $ pytest tests/auth -q                                      │
│ FAILED 1                                                   │
│ Applying patch...                                           │
├──────────────────────────────────────────────────────────────┤
│ > Fix the failing authentication test                       │
└──────────────────────────────────────────────────────────────┘
```

Important interactions:

-   file tree navigation
-   open file
-   diff preview
-   accept/reject patch
-   command output
-   tool approval
-   cancel agent
-   retry
-   inspect trace
-   switch model

------------------------------------------------------------------------

# 35. Phase 12 --- Browser Integration

Build this progressively.

## Level 1

``` text
web_search(query)
```

## Level 2

``` text
fetch_documentation(url)
```

## Level 3

``` text
browser.open()
browser.click()
browser.type()
browser.extract()
```

Use browser automation primarily when the agent needs information or
interaction unavailable through normal HTTP retrieval.

Treat web content as untrusted.

------------------------------------------------------------------------

# 36. Phase 13 --- Hooks

Hooks provide lifecycle interception.

Useful events:

``` text
before_session
after_session

before_prompt
after_prompt

before_model_call
after_model_call

before_tool
after_tool

before_compaction
after_compaction
```

Example:

``` text
Tool request
   ↓
before_tool hook
   ↓
permission check / logging
   ↓
execute
   ↓
after_tool hook
   ↓
trace result
```

Hooks should not become a hidden second agent. Keep them deterministic
and easy to inspect.

------------------------------------------------------------------------

# 37. Phase 14 --- Storage

Use SQLite for:

``` text
sessions
messages
tasks
tool_calls
tool_results
files
runs
metrics
```

Possible conceptual schema:

``` text
sessions
---------
id
created_at
updated_at
working_directory

messages
--------
id
session_id
role
content
created_at

tool_calls
----------
id
session_id
name
arguments
status
created_at

tool_results
------------
id
tool_call_id
result
exit_code
duration_ms
created_at
```

Keep raw provider payloads optional and redact secrets before
persistence.

------------------------------------------------------------------------

# 38. Phase 15 --- Observability

Start with JSONL:

``` text
.tmca/runs/<run-id>.jsonl
```

Example events:

``` json
{
  "event": "tool_call",
  "tool": "search_files",
  "duration_ms": 42
}
```

Then add OpenTelemetry.

Trace structure:

``` text
Agent Run
 ├── model generation
 ├── tool call: search
 ├── tool call: read_file
 ├── model generation
 ├── tool call: apply_patch
 ├── tool call: shell
 └── final generation
```

Track:

``` text
latency
tokens
estimated cost
tool count
errors
iterations
success
```

Langfuse can then provide a UI for inspecting runs.

------------------------------------------------------------------------

# 39. Phase 16 --- Git Integration

Start with read-only operations:

``` text
git status
git diff
git log
```

Then:

``` text
git add
git commit
```

Finally:

``` text
git push
GitHub pull request
```

Never automatically push or create a PR without a configurable approval
policy.

A useful workflow is:

``` text
Agent changes
    ↓
Tests
    ↓
git diff
    ↓
user approval
    ↓
commit
    ↓
push
    ↓
PR
```

------------------------------------------------------------------------

# 40. Phase 17 --- Evaluation

You need objective evaluation.

## Unit tests

Test individual components:

``` text
LLM parser
tool validator
path security
command policy
patch parser
context ranking
state transitions
```

## Integration tests

Test:

``` text
LLM
 ↓
tool
 ↓
filesystem
 ↓
shell
 ↓
result
```

Use mocked LLM responses for deterministic tests.

## Agent benchmarks

Eventually evaluate on coding benchmarks such as:

-   HumanEval
-   HumanEvalFix
-   RepoEval
-   SWE-bench
-   SWE-bench Verified

For repository-level coding agents, SWE-bench is particularly relevant
because tasks involve real GitHub issues and repository modifications.

------------------------------------------------------------------------

# 41. Metrics

Do not only measure "did it work?"

Track:

``` text
Task success rate
Patch success rate
Test pass rate
Tool-call validity
Tool-call success rate
Average iterations
Average tool calls
Average tokens
Cost per task
Latency
Human intervention rate
Recovery rate after failure
```

Example evaluation record:

``` text
Task: Fix issue #123

Success: YES
Iterations: 7
Tool calls: 13
Input tokens: ...
Output tokens: ...
Cost: ...
Time: ...
Human approvals: 2
Tests passed: YES
```

------------------------------------------------------------------------

# 42. Development Milestones

## Milestone 1 --- LLM CLI

Deliver:

``` bash
tmca "Explain recursion"
```

Features:

-   API key
-   HTTP request
-   response parsing
-   basic errors

## Milestone 2 --- Streaming

Deliver:

``` bash
tmca "Explain this concept"
```

with live token rendering.

## Milestone 3 --- Conversation

Deliver:

``` bash
tmca
> Explain JWT
> What are its weaknesses?
> Show an implementation
```

## Milestone 4 --- Calculator Tool

Prove:

``` text
LLM → tool call → validation → execution → result → LLM
```

## Milestone 5 --- Filesystem

Deliver:

``` bash
tmca "Explain this project"
```

using:

``` text
list
search
read
```

## Milestone 6 --- Code Editing

Deliver:

``` bash
tmca "Fix this bug"
```

using:

``` text
search
read
patch
```

## Milestone 7 --- Shell

Deliver:

``` bash
tmca "Run the tests and fix failures"
```

## Milestone 8 --- Agent Loop

Implement:

``` text
Plan → Act → Observe → Repeat
```

## Milestone 9 --- Repository Intelligence

Add:

``` text
ripgrep
BM25
Tree-sitter
hybrid retrieval
```

## Milestone 10 --- Security

Add:

``` text
permissions
sandbox
timeouts
path restrictions
secret protection
```

## Milestone 11 --- TUI

Add:

``` text
Textual
file tree
diff viewer
shell panel
approval modals
```

## Milestone 12 --- MCP

Expose/connect external tools.

## Milestone 13 --- Observability

Add:

``` text
OpenTelemetry
Langfuse
```

## Milestone 14 --- GitHub

Add:

``` text
commit
push
PR
```

## Milestone 15 --- Evaluation

Run benchmark tasks and publish metrics.

------------------------------------------------------------------------

# 43. Recommended Development Order

Do not build the architecture in the order it appears visually.

Build it in dependency order:

``` text
1. Python project
       ↓
2. LLM HTTP client
       ↓
3. Provider abstraction
       ↓
4. Streaming
       ↓
5. Conversation state
       ↓
6. Tool schema
       ↓
7. Tool registry
       ↓
8. Tool executor
       ↓
9. Filesystem tools
       ↓
10. Shell tool
       ↓
11. Agent loop
       ↓
12. Verification
       ↓
13. Context management
       ↓
14. Repository retrieval
       ↓
15. Tree-sitter
       ↓
16. Planning
       ↓
17. Self-correction
       ↓
18. Security
       ↓
19. Sandbox
       ↓
20. TUI
       ↓
21. MCP
       ↓
22. Browser
       ↓
23. Observability
       ↓
24. GitHub
       ↓
25. Benchmarking
```

------------------------------------------------------------------------

# 44. What NOT to Build First

Avoid these in Version 0.1:

``` text
❌ RAG
❌ vector database
❌ browser automation
❌ MCP
❌ multi-agent system
❌ Docker orchestration
❌ GitHub PR automation
❌ complex TUI
❌ autonomous background agents
❌ fine-tuning
```

First prove:

``` text
LLM
 ↓
Tool
 ↓
Observation
 ↓
LLM
```

Then expand.

------------------------------------------------------------------------

# 45. Learning Method for Every Component

For every subsystem, follow this cycle:

``` text
1. Understand the problem
2. Read the protocol/API specification
3. Implement the smallest version
4. Write unit tests
5. Add failure handling
6. Integrate with agent
7. Measure behavior
8. Compare with mature open-source projects
9. Refactor
10. Document what you learned
```

For example, for tool calling:

``` text
Study provider tool-call format
        ↓
Implement one calculator tool
        ↓
Validate arguments
        ↓
Execute
        ↓
Return result
        ↓
Test malformed arguments
        ↓
Add filesystem tool
        ↓
Add shell tool
```

This is much more educational than starting with an agent framework.

------------------------------------------------------------------------

# 46. Research and Open-Source Study Order

Study these alongside implementation:

## ReAct

Use it when implementing:

``` text
Plan/Act/Observe
```

## Toolformer

Use it when learning:

``` text
tool selection
tool arguments
tool results
```

## RepoCoder

Use it when implementing:

``` text
repository retrieval
iterative retrieval + generation
```

## SWE-bench

Use it when implementing:

``` text
repository-level evaluation
```

## SWE-agent

Use it when designing:

``` text
agent-computer interface
coding tools
execution feedback
```

## AutoCodeRover

Use it when implementing:

``` text
AST-aware repository search
fault localization
```

## Self-Refine

Use it when designing:

``` text
feedback → refinement
```

Also study practical implementations such as:

``` text
mini-SWE-agent
SWE-agent
Aider
OpenCode
OpenHands
Goose
MCP SDK
```

Study them for design ideas, not as code to copy.

------------------------------------------------------------------------

# 47. First Production-Quality Task

Your first serious target should be:

``` text
tmca
```

Then:

``` text
> Explain this project.
```

Expected behavior:

``` text
1. Identify working directory.
2. List top-level files.
3. Inspect README/config.
4. Search relevant source files.
5. Read selected files.
6. Summarize architecture.
```

Next:

``` text
> Find the failing tests and fix them.
```

Expected behavior:

``` text
1. Inspect project.
2. Run tests.
3. Capture failure.
4. Locate relevant code.
5. Read source.
6. Generate patch.
7. Show diff.
8. Request approval if required.
9. Apply patch.
10. Run targeted tests.
11. If failure, investigate.
12. Repeat within iteration limit.
13. Run final verification.
14. Report changes.
```

This is the first point where your project genuinely qualifies as a
coding agent.

------------------------------------------------------------------------

# 48. Final Architecture

The final TMCA system should conceptually look like:

``` text
┌───────────────────────────────────────────────────────────────┐
│                         TERMINAL TUI                          │
│                 Typer + Textual + Rich                       │
│                                                               │
│ Chat │ Files │ Editor │ Diff │ Shell │ Browser │ Approvals   │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                       AGENT RUNTIME                           │
│                                                               │
│ Session │ State │ Planner │ Loop │ Verification │ Stop Rules  │
└───────────────────────────────┬───────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
          ┌──────────────────┐     ┌──────────────────┐
          │  CONTEXT ENGINE  │     │    LLM GATEWAY   │
          │                  │     │                  │
          │ ripgrep          │     │ OpenRouter       │
          │ BM25             │     │ OpenAI           │
          │ Tree-sitter      │     │ Anthropic        │
          │ embeddings       │     │ Ollama           │
          │ reranking        │     └────────┬─────────┘
          └────────┬─────────┘              │
                   └──────────┬─────────────┘
                              ▼
                    ┌──────────────────┐
                    │  TOOL RUNTIME    │
                    │                  │
                    │ Schema           │
                    │ Registry         │
                    │ Dispatcher       │
                    │ Validation       │
                    │ Permissions      │
                    │ Hooks            │
                    └────────┬─────────┘
                             │
             ┌───────────────┼────────────────┐
             ▼               ▼                ▼
        Filesystem         Shell             Git
        read/edit          commands          operations
             │               │                │
             └───────────────┼────────────────┘
                             ▼
                    ┌──────────────────┐
                    │     SANDBOX      │
                    │ Docker/E2B/etc.  │
                    └────────┬─────────┘
                             ▼
                       Tests / Runtime
                             │
                             ▼
                         Observation
                             │
                             └──────→ Agent Loop

Cross-cutting:
Security → path policy, command policy, secret protection
Storage → SQLite
Observability → OpenTelemetry → Langfuse
External tools → MCP
Delivery → Git → GitHub PR
Evaluation → SWE-bench / custom benchmark
```

------------------------------------------------------------------------

# 49. Resume-Ready Project Description

After the core features are genuinely implemented:

> **Terminal-Native AI Coding Assistant** --- Built a model-agnostic
> terminal coding agent from scratch in Python, implementing LLM API
> integration, streaming, structured tool calling, repository-aware code
> retrieval, Tree-sitter-based code intelligence, filesystem/shell/Git
> tools, Plan--Act--Observe execution, test-driven self-correction,
> sandboxed execution, interactive TUI, and OpenTelemetry-based agent
> observability.

Possible stack:

``` text
Python
Typer
Textual
Rich
HTTPX
Pydantic
OpenAI/Anthropic/OpenRouter
SQLite
ripgrep
Tree-sitter
BM25
Embeddings/RAG
Playwright
Docker/E2B/Daytona
MCP
OpenTelemetry
Langfuse
Git
pytest
```

Only list components that you have actually implemented.

------------------------------------------------------------------------

# 50. Immediate Next Step

Do not start with the complete architecture.

Start with exactly these files:

``` text
src/
└── tmca/
    ├── __init__.py
    ├── main.py
    └── llm/
        ├── __init__.py
        ├── base.py
        └── openrouter.py
```

Your first milestone is:

``` text
tmca "Explain what an API is"
        ↓
Message
        ↓
OpenRouterProvider
        ↓
HTTPX
        ↓
LLM
        ↓
LLMResponse
        ↓
Rich terminal output
```

Then implement streaming.

Then tool calling.

Then filesystem tools.

Then the agent loop.

Everything else should be built on top of those foundations.
