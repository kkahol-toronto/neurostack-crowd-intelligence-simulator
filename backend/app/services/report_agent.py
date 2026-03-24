"""
Report Agent
ReACT-style simulation report generation with Zep retrieval

Features:
1. Builds reports from the simulation brief and Zep graph
2. Plans an outline, then generates each section
3. Each section uses multi-step ReACT (tool use + reflection)
4. Chat mode can call retrieval tools on demand
"""

import os
import json
import time
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from .zep_tools import (
    ZepToolsService, 
    SearchResult, 
    InsightForgeResult, 
    PanoramaResult,
    InterviewResult
)

logger = get_logger('neurostack_cis.report_agent')


class ReportLogger:
    """
    Structured report agent logger
    
    Writes agent_log.jsonl under the report folder with one JSON object per step.
    Each line is JSON: timestamp, action, details, etc.
    """
    
    def __init__(self, report_id: str):
        """
        Initialize logger
        
        Args:
            report_id: report id (determines log path)
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'agent_log.jsonl'
        )
        self.start_time = datetime.now()
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        """Ensure log directory exists"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _get_elapsed_time(self) -> float:
        """Elapsed seconds since start"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def log(
        self, 
        action: str, 
        stage: str,
        details: Dict[str, Any],
        section_title: str = None,
        section_index: int = None
    ):
        """
        Append one log entry
        
        Args:
            action: action name, e.g. start, tool_call, llm_response, section_complete
            stage: stage: planning, generating, completed
            details: details dict (not truncated)
            section_title: current section title (optional)
            section_index: current section index (optional)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(self._get_elapsed_time(), 2),
            "report_id": self.report_id,
            "action": action,
            "stage": stage,
            "section_title": section_title,
            "section_index": section_index,
            "details": details
        }
        
        # Append JSONL
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def log_start(self, simulation_id: str, graph_id: str, simulation_requirement: str):
        """Log report generation start"""
        self.log(
            action="report_start",
            stage="pending",
            details={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "simulation_requirement": simulation_requirement,
                "message": "Report generation started"
            }
        )
    
    def log_planning_start(self):
        """Log outline planning start"""
        self.log(
            action="planning_start",
            stage="planning",
            details={"message": "Starting outline planning"}
        )
    
    def log_planning_context(self, context: Dict[str, Any]):
        """Log planning context"""
        self.log(
            action="planning_context",
            stage="planning",
            details={
                "message": "Fetched simulation context",
                "context": context
            }
        )
    
    def log_planning_complete(self, outline_dict: Dict[str, Any]):
        """Log outline complete"""
        self.log(
            action="planning_complete",
            stage="planning",
            details={
                "message": "Outline planning complete",
                "outline": outline_dict
            }
        )
    
    def log_section_start(self, section_title: str, section_index: int):
        """Log section start"""
        self.log(
            action="section_start",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={"message": f"Starting section: {section_title}"}
        )
    
    def log_react_thought(self, section_title: str, section_index: int, iteration: int, thought: str):
        """Log ReACT thought"""
        self.log(
            action="react_thought",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "thought": thought,
                "message": f"ReACT iteration {iteration} thought"
            }
        )
    
    def log_tool_call(
        self, 
        section_title: str, 
        section_index: int,
        tool_name: str, 
        parameters: Dict[str, Any],
        iteration: int
    ):
        """Log tool call"""
        self.log(
            action="tool_call",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "parameters": parameters,
                "message": f"Tool call: {tool_name}"
            }
        )
    
    def log_tool_result(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        result: str,
        iteration: int
    ):
        """Log full tool result (untruncated)"""
        self.log(
            action="tool_result",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "result": result,  # full result
                "result_length": len(result),
                "message": f"Tool {tool_name} returned"
            }
        )
    
    def log_llm_response(
        self,
        section_title: str,
        section_index: int,
        response: str,
        iteration: int,
        has_tool_calls: bool,
        has_final_answer: bool
    ):
        """Log full LLM response"""
        self.log(
            action="llm_response",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "response": response,  # full response
                "response_length": len(response),
                "has_tool_calls": has_tool_calls,
                "has_final_answer": has_final_answer,
                "message": f"LLM response (tool_calls={has_tool_calls}, final_answer={has_final_answer})"
            }
        )
    
    def log_section_content(
        self,
        section_title: str,
        section_index: int,
        content: str,
        tool_calls_count: int
    ):
        """Log section body written (not necessarily final)"""
        self.log(
            action="section_content",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": content,  # full content
                "content_length": len(content),
                "tool_calls_count": tool_calls_count,
                "message": f"Section {section_title} body written"
            }
        )
    
    def log_section_full_complete(
        self,
        section_title: str,
        section_index: int,
        full_content: str
    ):
        """
        Log section complete

        Frontends can watch this entry for true section completion and full text.
        """
        self.log(
            action="section_complete",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": full_content,
                "content_length": len(full_content),
                "message": f"Section {section_title} complete"
            }
        )
    
    def log_report_complete(self, total_sections: int, total_time_seconds: float):
        """Log report complete"""
        self.log(
            action="report_complete",
            stage="completed",
            details={
                "total_sections": total_sections,
                "total_time_seconds": round(total_time_seconds, 2),
                "message": "Report generation complete"
            }
        )
    
    def log_error(self, error_message: str, stage: str, section_title: str = None):
        """Log error"""
        self.log(
            action="error",
            stage=stage,
            section_title=section_title,
            section_index=None,
            details={
                "error": error_message,
                "message": f"Error: {error_message}"
            }
        )


class ReportConsoleLogger:
    """
    Console log mirror for Report Agent
    
    Mirrors INFO/WARNING logs to console_log.txt under the report folder.
    Plain text, unlike structured agent_log.jsonl.
    """
    
    def __init__(self, report_id: str):
        """
        Init console file handler
        
        Args:
            report_id: report id (determines log path)
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'console_log.txt'
        )
        self._ensure_log_file()
        self._file_handler = None
        self._setup_file_handler()
    
    def _ensure_log_file(self):
        """Ensure log directory exists"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _setup_file_handler(self):
        """Attach file handler"""
        import logging
        
        # File handler
        self._file_handler = logging.FileHandler(
            self.log_file_path,
            mode='a',
            encoding='utf-8'
        )
        self._file_handler.setLevel(logging.INFO)
        
        # Same concise format as console
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        self._file_handler.setFormatter(formatter)
        
        # Attach to report_agent loggers
        loggers_to_attach = [
            'neurostack_cis.report_agent',
            'neurostack_cis.zep_tools',
        ]
        
        for logger_name in loggers_to_attach:
            target_logger = logging.getLogger(logger_name)
            # Avoid duplicate handlers
            if self._file_handler not in target_logger.handlers:
                target_logger.addHandler(self._file_handler)
    
    def close(self):
        """Detach and close file handler"""
        import logging
        
        if self._file_handler:
            loggers_to_detach = [
                'neurostack_cis.report_agent',
                'neurostack_cis.zep_tools',
            ]
            
            for logger_name in loggers_to_detach:
                target_logger = logging.getLogger(logger_name)
                if self._file_handler in target_logger.handlers:
                    target_logger.removeHandler(self._file_handler)
            
            self._file_handler.close()
            self._file_handler = None
    
    def __del__(self):
        """Close handler on teardown"""
        self.close()


class ReportStatus(str, Enum):
    """Report lifecycle status"""
    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportSection:
    """One section"""
    title: str
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content
        }

    def to_markdown(self, level: int = 2) -> str:
        """Render as Markdown"""
        md = f"{'#' * level} {self.title}\n\n"
        if self.content:
            md += f"{self.content}\n\n"
        return md


@dataclass
class ReportOutline:
    """Outline"""
    title: str
    summary: str
    sections: List[ReportSection]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections]
        }
    
    def to_markdown(self) -> str:
        """Render as Markdown"""
        md = f"# {self.title}\n\n"
        md += f"> {self.summary}\n\n"
        for section in self.sections:
            md += section.to_markdown()
        return md


@dataclass
class Report:
    """Full report"""
    report_id: str
    simulation_id: str
    graph_id: str
    simulation_requirement: str
    status: ReportStatus
    outline: Optional[ReportOutline] = None
    markdown_content: str = ""
    created_at: str = ""
    completed_at: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "simulation_id": self.simulation_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "status": self.status.value,
            "outline": self.outline.to_dict() if self.outline else None,
            "markdown_content": self.markdown_content,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error
        }


# ═══════════════════════════════════════════════════════════════
# Prompt templates
# ═══════════════════════════════════════════════════════════════

# ── Tool descriptions ──

TOOL_DESC_INSIGHT_FORGE = """\
【Deep insight retrieval — primary tool】
Our strongest retrieval routine for deep analysis. It will:
1. Decompose your question into sub-questions
2. Search the simulation graph from multiple angles
3. Combine semantic search, entity analysis, and relation chains
4. Return the richest possible evidence pack

【When to use】
- You need depth on a topic
- You need several angles on the same event
- You need material to support a section

【Returns】
- Verbatim facts you can quote
- Entity-level insights
- Relation-chain summaries"""

TOOL_DESC_PANORAMA_SEARCH = """\
【Panorama search — full view】
Returns the broadest picture of the simulation, including history. It will:
1. Load relevant nodes and relations
2. Separate currently valid facts from historical / expired ones
3. Help you see how the situation evolved

【When to use】
- You need the full timeline
- You need to compare phases of public reaction
- You need comprehensive entities and relations

【Returns】
- Currently valid facts (latest simulation state)
- Historical / expired facts (evolution trail)
- All involved entities"""

TOOL_DESC_QUICK_SEARCH = """\
【Quick search — lightweight】
Fast, lightweight retrieval for straightforward lookups.

【When to use】
- You need a specific fact quickly
- You need to verify a claim
- Simple lookup tasks

【Returns】
- The most relevant facts for the query"""

TOOL_DESC_INTERVIEW_AGENTS = """\
【Deep interviews — real agents (dual platform)】
Calls the live OASIS interview API against running simulated agents.
This is not an LLM role-play; it is the real interview endpoint with raw agent replies.
By default Twitter and Reddit are interviewed for richer coverage.

Flow:
1. Load persona files for all agents
2. Pick agents most relevant to the topic (e.g. students, media, officials)
3. Generate interview questions
4. POST /api/simulation/interview/batch for real dual-platform interviews
5. Merge results for multi-perspective analysis

【When to use】
- You need role-specific takes (student vs media vs official)
- You need multiple viewpoints
- You need authentic agent replies from OASIS
- You want “interview transcript” flavor in the report

【Returns】
- Agent identities
- Per-platform answers on Twitter and Reddit
- Pull quotes
- Summary and contrast

【Important】The OASIS simulation must be running."""

# ── Outline planning ──

PLAN_SYSTEM_PROMPT = """\
You are an expert author of “future scenario” reports. You have a god’s-eye view of the simulated world—every agent’s actions, statements, and interactions.

【Core idea】
We built a simulated world and injected a “simulation requirement” as a controlled variable. The trajectory of that world is a forecast of what might happen. You are not looking at “lab data”; you are looking at a rehearsal of the future.

【Your job】
Write a future-scenario report that answers:
1. Under our assumptions, what happens next?
2. How do different agent populations react and act?
3. What trends and risks does this run surface?

【Positioning】
- ✅ This is a forecast grounded in the simulation—what happens if…
- ✅ Focus on outcomes: narrative, crowd behavior, emergent patterns, risks
- ✅ Agent speech and action stand in for future population behavior
- ❌ Not an analysis of the real-world status quo
- ❌ Not a generic “public opinion overview” with no forecast angle

【Section count】
- Minimum 2 sections, maximum 5
- No nested outline items—each section is one full piece
- Be concise; center on the core forecast
- You choose the structure from what the run suggests

Output JSON only, shape:
{
    "title": "Report title",
    "summary": "One-line core forecast",
    "sections": [
        {
            "title": "Section title",
            "description": "What this section covers"
        }
    ]
}

sections must contain between 2 and 5 items."""

PLAN_USER_PROMPT_TEMPLATE = """\
【Scenario】
Simulation requirement (injected variable): {simulation_requirement}

【World scale】
- Entities: {total_nodes}
- Relations: {total_edges}
- Entity types: {entity_types}
- Active agents (typed entities): {total_entities}

【Sample forecast facts from the run】
{related_facts_json}

With a god’s-eye view of this rehearsal, ask:
1. What state does the future take under our assumptions?
2. How do different populations (agents) behave?
3. What trends deserve attention?

Design the best section plan from these signals.

Reminder: 2–5 sections, concise and forecast-focused."""

# ── Section generation ──

SECTION_SYSTEM_PROMPT_TEMPLATE = """\
You are writing one section of a future-scenario report.

Report title: {report_title}
Report summary: {report_summary}
Scenario (simulation requirement): {simulation_requirement}

Section to write: {section_title}

═══════════════════════════════════════════════════════════════
【Core idea】
═══════════════════════════════════════════════════════════════

The simulation is a rehearsal of the future. We injected conditions (the simulation requirement).
Agent behavior and interaction forecast how populations might behave.

Your job:
- Show what happens under those conditions
- Describe how groups (agents) react
- Surface trends, risks, and opportunities

❌ Do not write as a static analysis of “the real world today”
✅ Focus on “what happens next”—the graph holds the forecast

═══════════════════════════════════════════════════════════════
【Hard rules】
═══════════════════════════════════════════════════════════════

1. 【Use tools to observe the simulation】
   - You are watching a rehearsal of the future
   - Every claim must come from events and agent behavior in the graph
   - Do not fill gaps with outside knowledge
   - Call tools at least 3 times (at most 5) per section

2. 【Quote agent behavior】
   - Speech and actions are forecasts of population behavior
   - Use blockquotes, e.g.
     > "A group might say: …"
   - These quotes are your primary evidence

3. 【Language consistency】
   - Tool output may be English or mixed
   - Write the report in clear English
   - Translate any non-English snippets into fluent English before quoting, preserving meaning
   - Applies to body text and blockquotes

4. 【Stay faithful】
   - Reflect what the simulation actually shows
   - Do not invent facts absent from the run
   - If evidence is thin, say so

═══════════════════════════════════════════════════════════════
【⚠️ Format — critical】
═══════════════════════════════════════════════════════════════

【One section = one unit】
- Do not use Markdown headings (#, ##, ###, ####) inside the section body
- Do not repeat the section title as a heading—the system adds it
- Use **bold**, paragraphs, blockquotes, and lists only

【Good pattern】
```
This section traces how attention spread. From the simulation we see...

**Early burst phase**

Platform A acted as the first arena for the story:

> "Platform A carried most of the initial volume..."

**Amplification phase**

Platform B widened reach:

- Strong visuals
- High emotional resonance
```

【Bad pattern】
```
## Executive summary   ← do not add headings
### Phase one          ← no ### subheads

This section traces...
```

═══════════════════════════════════════════════════════════════
【Tools】(use 3–5 calls per section)
═══════════════════════════════════════════════════════════════

{tools_description}

【Mix tools】
- insight_forge: deep pass—sub-questions, facts, relations
- panorama_search: full picture, timeline, evolution
- quick_search: spot-check a fact
- interview_agents: first-person views from running agents

═══════════════════════════════════════════════════════════════
【Workflow】
═══════════════════════════════════════════════════════════════

Each reply must do exactly one of:

A) Call a tool — include reasoning, then:
<tool_call>
{{"name": "tool_name", "parameters": {{"param": "value"}}}}
</tool_call>
The system runs the tool; never fabricate observations.

B) Final text — after enough evidence, start with "Final Answer:" and write the section.

⚠️ Never mix tool calls and Final Answer in one reply.
⚠️ Never invent tool output.
⚠️ At most one tool call per reply.

═══════════════════════════════════════════════════════════════
【Section content】
═══════════════════════════════════════════════════════════════

1. Ground everything in retrieved simulation data
2. Quote generously
3. Markdown without headings: **bold**, lists, blank lines between paragraphs
4. Blockquotes must stand alone with blank lines around them

   ✅ Good:
   ```
   The official reply felt thin to observers.

   > "The playbook looked rigid for a fast-moving social feed."

   That sentiment was widely shared.
   ```

   ❌ Bad:
   ```
   The reply felt thin. > "The playbook..." Many agreed.
   ```
5. Stay coherent with earlier sections
6. Do not repeat points already covered below
7. Again: no headings—use **bold** for mini-leads"""

SECTION_USER_PROMPT_TEMPLATE = """\
Completed sections (read carefully—do not repeat):
{previous_content}

═══════════════════════════════════════════════════════════════
【Task】Write section: {section_title}
═══════════════════════════════════════════════════════════════

【Reminders】
1. Avoid repeating prior sections.
2. Start by calling tools—do not skip retrieval.
3. Mix tools; do not rely on only one.
4. Content must come from tools, not general knowledge.

【Format】
- No headings (# through ####)
- Do not open with "{section_title}" as a line—the UI adds the title
- Write body text; use **bold** instead of subheads

Steps:
1. Think what evidence you need
2. Call a tool
3. When ready, output Final Answer: (plain body, no headings)"""

# ── ReACT loop messages ──

REACT_OBSERVATION_TEMPLATE = """\
Observation (retrieval result):

═══ Tool {tool_name} returned ═══
{result}

═══════════════════════════════════════════════════════════════
Tools used: {tool_calls_count}/{max_tool_calls} (so far: {used_tools_str}){unused_hint}
- If enough: start with "Final Answer:" and write the section (quote the evidence above)
- If not: call one more tool
═══════════════════════════════════════════════════════════════"""

REACT_INSUFFICIENT_TOOLS_MSG = (
    "【Note】You only used tools {tool_calls_count} time(s); need at least {min_tool_calls}."
    " Call more tools for simulation evidence, then output Final Answer. {unused_hint}"
)

REACT_INSUFFICIENT_TOOLS_MSG_ALT = (
    "Only {tool_calls_count} tool call(s) so far; need at least {min_tool_calls}."
    " Please call tools for simulation data. {unused_hint}"
)

REACT_TOOL_LIMIT_MSG = (
    "Tool budget reached ({tool_calls_count}/{max_tool_calls}); no more tool calls."
    ' Write the section now starting with "Final Answer:" using what you have.'
)

REACT_UNUSED_TOOLS_HINT = "\n💡 Not used yet: {unused_list} — try another tool for a different angle"

REACT_FORCE_FINAL_MSG = "Tool limit reached. Output Final Answer: with the section text now."

# ── Chat prompt ──

CHAT_SYSTEM_PROMPT_TEMPLATE = """\
You are a concise simulation-forecast assistant.

【Context】
Scenario: {simulation_requirement}

【Existing report】
{report_content}

【Rules】
1. Prefer answering from the report
2. Answer directly—avoid long chain-of-thought
3. Call tools only if the report is insufficient
4. Be short, clear, structured

【Tools】(only if needed, at most 1–2 calls)
{tools_description}

【Tool format】
<tool_call>
{{"name": "tool_name", "parameters": {{"param": "value"}}}}
</tool_call>

【Style】
- Short and direct
- Use > quotes for key evidence
- Lead with the answer, then why"""

CHAT_OBSERVATION_SUFFIX = "\n\nAnswer briefly."

# ═══════════════════════════════════════════════════════════════
# ReportAgent
# ═══════════════════════════════════════════════════════════════


class ReportAgent:
    """
    Report Agent — simulation report generator

    Uses ReACT (reason + act):
    1. Plan: analyze brief, outline sections
    2. Generate: each section may call tools multiple times
    3. Reflect: check coverage and fidelity
    """
    
    # Max tool calls per section
    MAX_TOOL_CALLS_PER_SECTION = 5
    
    # Max reflection rounds (reserved)
    MAX_REFLECTION_ROUNDS = 3
    
    # Max tool calls per chat turn
    MAX_TOOL_CALLS_PER_CHAT = 2
    
    def __init__(
        self, 
        graph_id: str,
        simulation_id: str,
        simulation_requirement: str,
        llm_client: Optional[LLMClient] = None,
        zep_tools: Optional[ZepToolsService] = None
    ):
        """
        Construct Report Agent
        
        Args:
            graph_id: graph id
            simulation_id: simulation id
            simulation_requirement: simulation brief
            llm_client: optional LLM client
            zep_tools: optional ZepToolsService
        """
        self.graph_id = graph_id
        self.simulation_id = simulation_id
        self.simulation_requirement = simulation_requirement
        
        self.llm = llm_client or LLMClient()
        self.zep_tools = zep_tools or ZepToolsService()
        
        # Tool specs
        self.tools = self._define_tools()
        
        # Structured logger (set in generate_report）
        self.report_logger: Optional[ReportLogger] = None
        # Console mirror (set in generate_report）
        self.console_logger: Optional[ReportConsoleLogger] = None
        
        logger.info(f"ReportAgent ready: graph_id={graph_id}, simulation_id={simulation_id}")
    
    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        """Define tool specs for the LLM"""
        return {
            "insight_forge": {
                "name": "insight_forge",
                "description": TOOL_DESC_INSIGHT_FORGE,
                "parameters": {
                    "query": "Topic or question to analyze in depth",
                    "report_context": "Optional section context for finer sub-questions"
                }
            },
            "panorama_search": {
                "name": "panorama_search",
                "description": TOOL_DESC_PANORAMA_SEARCH,
                "parameters": {
                    "query": "Search query for ranking",
                    "include_expired": "Include expired/historical facts (default true)"
                }
            },
            "quick_search": {
                "name": "quick_search",
                "description": TOOL_DESC_QUICK_SEARCH,
                "parameters": {
                    "query": "Search query string",
                    "limit": "Result limit (optional, default 10)"
                }
            },
            "interview_agents": {
                "name": "interview_agents",
                "description": TOOL_DESC_INTERVIEW_AGENTS,
                "parameters": {
                    "interview_topic": "Interview topic or brief (free text)",
                    "max_agents": "Max agents to interview (optional, default 5, max 10)"
                }
            }
        }
    
    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any], report_context: str = "") -> str:
        """
        Execute one tool call
        
        Args:
            tool_name: tool name
            parameters: tool parameters
            report_context: extra context for insight_forge
            
        Returns:
            Tool output as text
        """
        logger.info(f"Tool: {tool_name}, params: {parameters}")
        
        try:
            if tool_name == "insight_forge":
                query = parameters.get("query", "")
                ctx = parameters.get("report_context", "") or report_context
                result = self.zep_tools.insight_forge(
                    graph_id=self.graph_id,
                    query=query,
                    simulation_requirement=self.simulation_requirement,
                    report_context=ctx
                )
                return result.to_text()
            
            elif tool_name == "panorama_search":
                # Panorama search
                query = parameters.get("query", "")
                include_expired = parameters.get("include_expired", True)
                if isinstance(include_expired, str):
                    include_expired = include_expired.lower() in ['true', '1', 'yes']
                result = self.zep_tools.panorama_search(
                    graph_id=self.graph_id,
                    query=query,
                    include_expired=include_expired
                )
                return result.to_text()
            
            elif tool_name == "quick_search":
                # Quick search
                query = parameters.get("query", "")
                limit = parameters.get("limit", 10)
                if isinstance(limit, str):
                    limit = int(limit)
                result = self.zep_tools.quick_search(
                    graph_id=self.graph_id,
                    query=query,
                    limit=limit
                )
                return result.to_text()
            
            elif tool_name == "interview_agents":
                # Interview via live OASIS API (dual platform)
                interview_topic = parameters.get("interview_topic", parameters.get("query", ""))
                max_agents = parameters.get("max_agents", 5)
                if isinstance(max_agents, str):
                    max_agents = int(max_agents)
                max_agents = min(max_agents, 10)
                result = self.zep_tools.interview_agents(
                    simulation_id=self.simulation_id,
                    interview_requirement=interview_topic,
                    simulation_requirement=self.simulation_requirement,
                    max_agents=max_agents
                )
                return result.to_text()
            
            # ========== Legacy tool names (redirected) ==========
            
            elif tool_name == "search_graph":
                # redirect -> quick_search
                logger.info("search_graph -> quick_search")
                return self._execute_tool("quick_search", parameters, report_context)
            
            elif tool_name == "get_graph_statistics":
                result = self.zep_tools.get_graph_statistics(self.graph_id)
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_entity_summary":
                entity_name = parameters.get("entity_name", "")
                result = self.zep_tools.get_entity_summary(
                    graph_id=self.graph_id,
                    entity_name=entity_name
                )
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_simulation_context":
                # redirect -> insight_forge
                logger.info("get_simulation_context -> insight_forge")
                query = parameters.get("query", self.simulation_requirement)
                return self._execute_tool("insight_forge", {"query": query}, report_context)
            
            elif tool_name == "get_entities_by_type":
                entity_type = parameters.get("entity_type", "")
                nodes = self.zep_tools.get_entities_by_type(
                    graph_id=self.graph_id,
                    entity_type=entity_type
                )
                result = [n.to_dict() for n in nodes]
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            else:
                return f"Unknown tool: {tool_name}. Use one of: insight_forge, panorama_search, quick_search, interview_agents"
                
        except Exception as e:
            logger.error(f"Tool failed: {tool_name}, error: {str(e)}")
            return f"Tool execution failed: {str(e)}"
    
    # Valid tool names for bare JSON fallback
    VALID_TOOL_NAMES = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from LLM output

        Supported formats (priority order):
        1. <tool_call>{"name": "tool_name", "parameters": {...}}</tool_call>
        2. Bare JSON object as the whole response
        """
        tool_calls = []

        # Format 1: XML-style
        xml_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        for match in re.finditer(xml_pattern, response, re.DOTALL):
            try:
                call_data = json.loads(match.group(1))
                tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        if tool_calls:
            return tool_calls

        # Format 2: bare JSON fallback
        # Only if format 1 missed
        stripped = response.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                call_data = json.loads(stripped)
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
                    return tool_calls
            except json.JSONDecodeError:
                pass

        # Trailing bare JSON extraction
        json_pattern = r'(\{"(?:name|tool)"\s*:.*?\})\s*$'
        match = re.search(json_pattern, stripped, re.DOTALL)
        if match:
            try:
                call_data = json.loads(match.group(1))
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        return tool_calls

    def _is_valid_tool_call(self, data: dict) -> bool:
        """Validate parsed JSON as a tool call"""
        # Accept name/parameters or tool/params
        tool_name = data.get("name") or data.get("tool")
        if tool_name and tool_name in self.VALID_TOOL_NAMES:
            # Normalize keys
            if "tool" in data:
                data["name"] = data.pop("tool")
            if "params" in data and "parameters" not in data:
                data["parameters"] = data.pop("params")
            return True
        return False
    
    def _get_tools_description(self) -> str:
        """Build tools description string"""
        desc_parts = ["Available tools:"]
        for name, tool in self.tools.items():
            params_desc = ", ".join([f"{k}: {v}" for k, v in tool["parameters"].items()])
            desc_parts.append(f"- {name}: {tool['description']}")
            if params_desc:
                desc_parts.append(f"  Parameters: {params_desc}")
        return "\n".join(desc_parts)
    
    def plan_outline(
        self, 
        progress_callback: Optional[Callable] = None
    ) -> ReportOutline:
        """
        Plan the report outline
        
        Uses the LLM on the simulation brief.
        
        Args:
            progress_callback: optional progress callback
            
        Returns:
            ReportOutline
        """
        logger.info("Starting outline planning...")
        
        if progress_callback:
            progress_callback("planning", 0, "Analyzing simulation brief...")
        
        # Fetch simulation context
        context = self.zep_tools.get_simulation_context(
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement
        )
        
        if progress_callback:
            progress_callback("planning", 30, "Generating outline...")
        
        system_prompt = PLAN_SYSTEM_PROMPT
        user_prompt = PLAN_USER_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            total_nodes=context.get('graph_statistics', {}).get('total_nodes', 0),
            total_edges=context.get('graph_statistics', {}).get('total_edges', 0),
            entity_types=list(context.get('graph_statistics', {}).get('entity_types', {}).keys()),
            total_entities=context.get('total_entities', 0),
            related_facts_json=json.dumps(context.get('related_facts', [])[:10], ensure_ascii=False, indent=2),
        )

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            if progress_callback:
                progress_callback("planning", 80, "Parsing outline...")
            
            # Parse sections
            sections = []
            for section_data in response.get("sections", []):
                sections.append(ReportSection(
                    title=section_data.get("title", ""),
                    content=""
                ))
            
            outline = ReportOutline(
                title=response.get("title", "Simulation analysis report"),
                summary=response.get("summary", ""),
                sections=sections
            )
            
            if progress_callback:
                progress_callback("planning", 100, "Outline planning complete")
            
            logger.info(f"Outline planning complete: {len(sections)} sections")
            return outline
            
        except Exception as e:
            logger.error(f"Outline planning failed: {str(e)}")
            # Default outline fallback
            return ReportOutline(
                title="Future scenario report",
                summary="Trends and risks from the simulated forecast",
                sections=[
                    ReportSection(title="Scenario and key findings"),
                    ReportSection(title="Forecasted population behavior"),
                    ReportSection(title="Outlook and risks")
                ]
            )
    
    def _generate_section_react(
        self, 
        section: ReportSection,
        outline: ReportOutline,
        previous_sections: List[str],
        progress_callback: Optional[Callable] = None,
        section_index: int = 0
    ) -> str:
        """
        Generate one section with ReACT.

        Loop: think → tool → observe → repeat until enough evidence, then Final Answer.

        Args:
            section: Target section
            outline: Full outline
            previous_sections: Text of prior sections (continuity)
            progress_callback: Optional progress hook
            section_index: Section index for logging

        Returns:
            Section body (Markdown body without top heading)
        """
        logger.info(f"ReACT section: {section.title}")
        
        if self.report_logger:
            self.report_logger.log_section_start(section.title, section_index)
        
        system_prompt = SECTION_SYSTEM_PROMPT_TEMPLATE.format(
            report_title=outline.title,
            report_summary=outline.summary,
            simulation_requirement=self.simulation_requirement,
            section_title=section.title,
            tools_description=self._get_tools_description(),
        )

        # User prompt: up to 4000 chars per prior section
        if previous_sections:
            previous_parts = []
            for sec in previous_sections:
                truncated = sec[:4000] + "..." if len(sec) > 4000 else sec
                previous_parts.append(truncated)
            previous_content = "\n\n---\n\n".join(previous_parts)
        else:
            previous_content = "(This is the first section.)"
        
        user_prompt = SECTION_USER_PROMPT_TEMPLATE.format(
            previous_content=previous_content,
            section_title=section.title,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        tool_calls_count = 0
        max_iterations = 5
        min_tool_calls = 3
        conflict_retries = 0
        used_tools = set()
        all_tools = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

        report_context = f"Section title: {section.title}\nSimulation brief: {self.simulation_requirement}"
        
        for iteration in range(max_iterations):
            if progress_callback:
                progress_callback(
                    "generating", 
                    int((iteration / max_iterations) * 100),
                    f"Retrieving & drafting ({tool_calls_count}/{self.MAX_TOOL_CALLS_PER_SECTION})"
                )
            
            response = self.llm.chat(
                messages=messages,
                temperature=0.5,
                max_tokens=4096
            )

            if response is None:
                logger.warning(f"Section {section.title} iter {iteration + 1}: LLM returned None")
                if iteration < max_iterations - 1:
                    messages.append({"role": "assistant", "content": "(empty response)"})
                    messages.append({"role": "user", "content": "Please continue."})
                    continue
                break

            logger.debug(f"LLM response: {response[:200]}...")

            tool_calls = self._parse_tool_calls(response)
            has_tool_calls = bool(tool_calls)
            has_final_answer = "Final Answer:" in response

            if has_tool_calls and has_final_answer:
                conflict_retries += 1
                logger.warning(
                    f"Section {section.title} iter {iteration+1}: "
                    f"both tool call and Final Answer (conflict #{conflict_retries})"
                )

                if conflict_retries <= 2:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": (
                            "[Format error] You cannot output a tool call and Final Answer in the same turn.\n"
                            "Do exactly one of:\n"
                            "- Call one tool (one <tool_call> block, no Final Answer)\n"
                            "- Or output final text starting with 'Final Answer:' (no <tool_call>)\n"
                            "Reply again with only one of these."
                        ),
                    })
                    continue
                else:
                    logger.warning(
                        f"Section {section.title}: {conflict_retries} conflicts; "
                        "truncating to first tool call"
                    )
                    first_tool_end = response.find('</tool_call>')
                    if first_tool_end != -1:
                        response = response[:first_tool_end + len('</tool_call>')]
                        tool_calls = self._parse_tool_calls(response)
                        has_tool_calls = bool(tool_calls)
                    has_final_answer = False
                    conflict_retries = 0

            if self.report_logger:
                self.report_logger.log_llm_response(
                    section_title=section.title,
                    section_index=section_index,
                    response=response,
                    iteration=iteration + 1,
                    has_tool_calls=has_tool_calls,
                    has_final_answer=has_final_answer
                )

            if has_final_answer:
                if tool_calls_count < min_tool_calls:
                    messages.append({"role": "assistant", "content": response})
                    unused_tools = all_tools - used_tools
                    unused_hint = f"(Not yet used: {', '.join(unused_tools)})" if unused_tools else ""
                    messages.append({
                        "role": "user",
                        "content": REACT_INSUFFICIENT_TOOLS_MSG.format(
                            tool_calls_count=tool_calls_count,
                            min_tool_calls=min_tool_calls,
                            unused_hint=unused_hint,
                        ),
                    })
                    continue

                final_answer = response.split("Final Answer:")[-1].strip()
                logger.info(f"Section {section.title} done (tool calls: {tool_calls_count})")

                if self.report_logger:
                    self.report_logger.log_section_content(
                        section_title=section.title,
                        section_index=section_index,
                        content=final_answer,
                        tool_calls_count=tool_calls_count
                    )
                return final_answer

            if has_tool_calls:
                if tool_calls_count >= self.MAX_TOOL_CALLS_PER_SECTION:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": REACT_TOOL_LIMIT_MSG.format(
                            tool_calls_count=tool_calls_count,
                            max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        ),
                    })
                    continue

                call = tool_calls[0]
                if len(tool_calls) > 1:
                    logger.info(f"LLM emitted {len(tool_calls)} tools; executing first only: {call['name']}")

                if self.report_logger:
                    self.report_logger.log_tool_call(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        parameters=call.get("parameters", {}),
                        iteration=iteration + 1
                    )

                result = self._execute_tool(
                    call["name"],
                    call.get("parameters", {}),
                    report_context=report_context
                )

                if self.report_logger:
                    self.report_logger.log_tool_result(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        result=result,
                        iteration=iteration + 1
                    )

                tool_calls_count += 1
                used_tools.add(call['name'])

                unused_tools = all_tools - used_tools
                unused_hint = ""
                if unused_tools and tool_calls_count < self.MAX_TOOL_CALLS_PER_SECTION:
                    unused_hint = REACT_UNUSED_TOOLS_HINT.format(unused_list=", ".join(unused_tools))

                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": REACT_OBSERVATION_TEMPLATE.format(
                        tool_name=call["name"],
                        result=result,
                        tool_calls_count=tool_calls_count,
                        max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        used_tools_str=", ".join(used_tools),
                        unused_hint=unused_hint,
                    ),
                })
                continue

            messages.append({"role": "assistant", "content": response})

            if tool_calls_count < min_tool_calls:
                unused_tools = all_tools - used_tools
                unused_hint = f"(Not yet used: {', '.join(unused_tools)})" if unused_tools else ""

                messages.append({
                    "role": "user",
                    "content": REACT_INSUFFICIENT_TOOLS_MSG_ALT.format(
                        tool_calls_count=tool_calls_count,
                        min_tool_calls=min_tool_calls,
                        unused_hint=unused_hint,
                    ),
                })
                continue

            logger.info(
                f"Section {section.title}: no 'Final Answer:' prefix; accepting raw output "
                f"(tool calls: {tool_calls_count})"
            )
            final_answer = response.strip()

            if self.report_logger:
                self.report_logger.log_section_content(
                    section_title=section.title,
                    section_index=section_index,
                    content=final_answer,
                    tool_calls_count=tool_calls_count
                )
            return final_answer
        
        logger.warning(f"Section {section.title}: max iterations reached; forcing final text")
        messages.append({"role": "user", "content": REACT_FORCE_FINAL_MSG})
        
        response = self.llm.chat(
            messages=messages,
            temperature=0.5,
            max_tokens=4096
        )

        if response is None:
            logger.error(f"Section {section.title}: force-final step returned None from LLM")
            final_answer = "(This section failed: empty LLM response. Please retry.)"
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
        else:
            final_answer = response
        
        if self.report_logger:
            self.report_logger.log_section_content(
                section_title=section.title,
                section_index=section_index,
                content=final_answer,
                tool_calls_count=tool_calls_count
            )
        
        return final_answer
    
    def generate_report(
        self, 
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        report_id: Optional[str] = None
    ) -> Report:
        """
        Generate the full report (streaming sections)
        
        Each section is saved as soon as it finishes.
        Layout:
        reports/{report_id}/
            meta.json       - metadata
            outline.json    - outline
            progress.json   - progress
            section_01.md   - section 1
            section_02.md   - section 2
            ...
            full_report.md  - full report
        
        Args:
            progress_callback: optional (stage, progress, message)
            report_id: optional; auto-generated if omitted
            
        Returns:
            Report object
        """
        import uuid
        
        # Auto report_id
        if not report_id:
            report_id = f"report_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()
        
        report = Report(
            report_id=report_id,
            simulation_id=self.simulation_id,
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement,
            status=ReportStatus.PENDING,
            created_at=datetime.now().isoformat()
        )
        
        # Titles of finished sections
        completed_section_titles = []
        
        try:
            # Init folder and state
            ReportManager._ensure_report_folder(report_id)
            
            # Structured logger (agent_log.jsonl)
            self.report_logger = ReportLogger(report_id)
            self.report_logger.log_start(
                simulation_id=self.simulation_id,
                graph_id=self.graph_id,
                simulation_requirement=self.simulation_requirement
            )
            
            # Init console file handler（console_log.txt）
            self.console_logger = ReportConsoleLogger(report_id)
            
            ReportManager.update_progress(
                report_id, "pending", 0, "Initializing report...",
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            # Phase 1: outline
            report.status = ReportStatus.PLANNING
            ReportManager.update_progress(
                report_id, "planning", 5, "Starting outline planning...",
                completed_sections=[]
            )
            
            # Log planning start
            self.report_logger.log_planning_start()
            
            if progress_callback:
                progress_callback("planning", 0, "Starting outline planning...")
            
            outline = self.plan_outline(
                progress_callback=lambda stage, prog, msg: 
                    progress_callback(stage, prog // 5, msg) if progress_callback else None
            )
            report.outline = outline
            
            # Log planning done
            self.report_logger.log_planning_complete(outline.to_dict())
            
            # Persist outline
            ReportManager.save_outline(report_id, outline)
            ReportManager.update_progress(
                report_id, "planning", 15, f"Outline complete,{len(outline.sections)} sections",
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            logger.info(f"Outline saved: {report_id}/outline.json")
            
            # Phase 2: sections
            report.status = ReportStatus.GENERATING
            
            total_sections = len(outline.sections)
            generated_sections = []  # prior sections for context
            
            for i, section in enumerate(outline.sections):
                section_num = i + 1
                base_progress = 20 + int((i / total_sections) * 70)
                
                # Progress
                ReportManager.update_progress(
                    report_id, "generating", base_progress,
                    f"Generating section: {section.title} ({section_num}/{total_sections})",
                    current_section=section.title,
                    completed_sections=completed_section_titles
                )
                
                if progress_callback:
                    progress_callback(
                        "generating", 
                        base_progress, 
                        f"Generating section: {section.title} ({section_num}/{total_sections})"
                    )
                
                # Generate section body
                section_content = self._generate_section_react(
                    section=section,
                    outline=outline,
                    previous_sections=generated_sections,
                    progress_callback=lambda stage, prog, msg:
                        progress_callback(
                            stage, 
                            base_progress + int(prog * 0.7 / total_sections),
                            msg
                        ) if progress_callback else None,
                    section_index=section_num
                )
                
                section.content = section_content
                generated_sections.append(f"## {section.title}\n\n{section_content}")

                # Save section
                ReportManager.save_section(report_id, section_num, section)
                completed_section_titles.append(section.title)

                # Log section done
                full_section_content = f"## {section.title}\n\n{section_content}"

                if self.report_logger:
                    self.report_logger.log_section_full_complete(
                        section_title=section.title,
                        section_index=section_num,
                        full_content=full_section_content.strip()
                    )

                logger.info(f"Section saved: {report_id}/section_{section_num:02d}.md")
                
                # Progress
                ReportManager.update_progress(
                    report_id, "generating", 
                    base_progress + int(70 / total_sections),
                    f"Section {section.title} done",
                    current_section=None,
                    completed_sections=completed_section_titles
                )
            
            # Phase 3: assemble
            if progress_callback:
                progress_callback("generating", 95, "Assembling full report...")
            
            ReportManager.update_progress(
                report_id, "generating", 95, "Assembling full report...",
                completed_sections=completed_section_titles
            )
            
            # Assemble via ReportManager
            report.markdown_content = ReportManager.assemble_full_report(report_id, outline)
            report.status = ReportStatus.COMPLETED
            report.completed_at = datetime.now().isoformat()
            
            # Total duration
            total_time_seconds = (datetime.now() - start_time).total_seconds()
            
            # Log completion
            if self.report_logger:
                self.report_logger.log_report_complete(
                    total_sections=total_sections,
                    total_time_seconds=total_time_seconds
                )
            
            # Save final
            ReportManager.save_report(report)
            ReportManager.update_progress(
                report_id, "completed", 100, "Report generation complete",
                completed_sections=completed_section_titles
            )
            
            if progress_callback:
                progress_callback("completed", 100, "Report generation complete")
            
            logger.info(f"Report generation complete: {report_id}")
            
            # Close console logger
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
            
        except Exception as e:
            logger.error(f"Report failed: {str(e)}")
            report.status = ReportStatus.FAILED
            report.error = str(e)
            
            # Log error
            if self.report_logger:
                self.report_logger.log_error(str(e), "failed")
            
            # Persist failure
            try:
                ReportManager.save_report(report)
                ReportManager.update_progress(
                    report_id, "failed", -1, f"Report failed: {str(e)}",
                    completed_sections=completed_section_titles
                )
            except Exception:
                pass  # ignore secondary save errors
            
            # Close console logger
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
    
    def chat(
        self, 
        message: str,
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Chat with the report agent
        
        The agent may call retrieval tools when needed
        
        Args:
            message: User message
            chat_history: Chat history
            
        Returns:
            {
                "response": "Assistant reply",
                "tool_calls": [Tools invoked],
                "sources": [Sources]
            }
        """
        logger.info(f"Report chat: {message[:50]}...")
        
        chat_history = chat_history or []
        
        # Load existing report text
        report_content = ""
        try:
            report = ReportManager.get_report_by_simulation(self.simulation_id)
            if report and report.markdown_content:
                # Truncate for context window
                report_content = report.markdown_content[:15000]
                if len(report.markdown_content) > 15000:
                    report_content += "\n\n... [report truncated] ..."
        except Exception as e:
            logger.warning(f"Failed to load report: {e}")
        
        system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            report_content=report_content if report_content else "(no report yet)",
            tools_description=self._get_tools_description(),
        )

        # messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # history
        for h in chat_history[-10:]:  # cap history
            messages.append(h)
        
        # User message
        messages.append({
            "role": "user", 
            "content": message
        })
        
        # light ReACT
        tool_calls_made = []
        max_iterations = 2  # few iterations
        
        for iteration in range(max_iterations):
            response = self.llm.chat(
                messages=messages,
                temperature=0.5
            )
            
            # parse tools
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # no tools: return text
                clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', response, flags=re.DOTALL)
                clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
                
                return {
                    "response": clean_response.strip(),
                    "tool_calls": tool_calls_made,
                    "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
                }
            
            # run tools
            tool_results = []
            for call in tool_calls[:1]:  # one tool per iteration max
                if len(tool_calls_made) >= self.MAX_TOOL_CALLS_PER_CHAT:
                    break
                result = self._execute_tool(call["name"], call.get("parameters", {}))
                tool_results.append({
                    "tool": call["name"],
                    "result": result[:1500]  # trim result
                })
                tool_calls_made.append(call)
            
            # append observation
            messages.append({"role": "assistant", "content": response})
            observation = "\n".join([f"[{r['tool']} result]\n{r['result']}" for r in tool_results])
            messages.append({
                "role": "user",
                "content": observation + CHAT_OBSERVATION_SUFFIX
            })
        
        # final pass
        final_response = self.llm.chat(
            messages=messages,
            temperature=0.5
        )
        
        # strip tool XML
        clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', final_response, flags=re.DOTALL)
        clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
        
        return {
            "response": clean_response.strip(),
            "tool_calls": tool_calls_made,
            "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
        }


class ReportManager:
    """
    Report persistence
    
    Save/load reports on disk
    
    On-disk layout (per report folder):
    reports/
      {report_id}/
        meta.json          - metadata and status
        outline.json       - outline
        progress.json      - progress
        section_01.md      - section 1
        section_02.md      - section 2
        ...
        full_report.md     - assembled report
    """
    
    # Root
    REPORTS_DIR = os.path.join(Config.UPLOAD_FOLDER, 'reports')
    
    @classmethod
    def _ensure_reports_dir(cls):
        """Ensure reports root"""
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)
    
    @classmethod
    def _get_report_folder(cls, report_id: str) -> str:
        """Report folder path"""
        return os.path.join(cls.REPORTS_DIR, report_id)
    
    @classmethod
    def _ensure_report_folder(cls, report_id: str) -> str:
        """mkdir report folder"""
        folder = cls._get_report_folder(report_id)
        os.makedirs(folder, exist_ok=True)
        return folder
    
    @classmethod
    def _get_report_path(cls, report_id: str) -> str:
        """meta.json path"""
        return os.path.join(cls._get_report_folder(report_id), "meta.json")
    
    @classmethod
    def _get_report_markdown_path(cls, report_id: str) -> str:
        """full_report.md path"""
        return os.path.join(cls._get_report_folder(report_id), "full_report.md")
    
    @classmethod
    def _get_outline_path(cls, report_id: str) -> str:
        """outline.json path"""
        return os.path.join(cls._get_report_folder(report_id), "outline.json")
    
    @classmethod
    def _get_progress_path(cls, report_id: str) -> str:
        """progress.json path"""
        return os.path.join(cls._get_report_folder(report_id), "progress.json")
    
    @classmethod
    def _get_section_path(cls, report_id: str, section_index: int) -> str:
        """section md path"""
        return os.path.join(cls._get_report_folder(report_id), f"section_{section_index:02d}.md")
    
    @classmethod
    def _get_agent_log_path(cls, report_id: str) -> str:
        """agent_log.jsonl path"""
        return os.path.join(cls._get_report_folder(report_id), "agent_log.jsonl")
    
    @classmethod
    def _get_console_log_path(cls, report_id: str) -> str:
        """console_log.txt path"""
        return os.path.join(cls._get_report_folder(report_id), "console_log.txt")
    
    @classmethod
    def get_console_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Read console log
        
        Console-style lines during generation,
        unlike structured agent_log.jsonl.
        
        Args:
            report_id: report id
            from_line: Start line for incremental reads (0 = start)
            
        Returns:
            {
                "logs": [Line list],
                "total_lines": total_lines,
                "from_line": from_line,
                "has_more": has_more
            }
        """
        log_path = cls._get_console_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    # Keep raw lines, strip trailing newline
                    logs.append(line.rstrip('\n\r'))
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # end of file
        }
    
    @classmethod
    def get_console_log_stream(cls, report_id: str) -> List[str]:
        """
        Read entire console log
        
        Args:
            report_id: report id
            
        Returns:
            Line list
        """
        result = cls.get_console_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def get_agent_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Read structured agent log
        
        Args:
            report_id: report id
            from_line: Start line for incremental reads (0 = start)
            
        Returns:
            {
                "logs": [Parsed log entries],
                "total_lines": total_lines,
                "from_line": from_line,
                "has_more": has_more
            }
        """
        log_path = cls._get_agent_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    try:
                        log_entry = json.loads(line.strip())
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        # skip bad json lines
                        continue
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # end of file
        }
    
    @classmethod
    def get_agent_log_stream(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Read full agent log
        
        Args:
            report_id: report id
            
        Returns:
            Parsed log entries
        """
        result = cls.get_agent_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def save_outline(cls, report_id: str, outline: ReportOutline) -> None:
        """
        Save outline
        
        Call after planning
        """
        cls._ensure_report_folder(report_id)
        
        with open(cls._get_outline_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(outline.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Outline saved: {report_id}")
    
    @classmethod
    def save_section(
        cls,
        report_id: str,
        section_index: int,
        section: ReportSection
    ) -> str:
        """
        Save one section file.

        Call immediately after each section is generated.

        Args:
            report_id: report id
            section_index: 1-based section index
            section: ReportSection

        Returns:
            Written path
        """
        cls._ensure_report_folder(report_id)

        # Build md; strip duplicate heading
        cleaned_content = cls._clean_section_content(section.content, section.title)
        md_content = f"## {section.title}\n\n"
        if cleaned_content:
            md_content += f"{cleaned_content}\n\n"

        # Write file
        file_suffix = f"section_{section_index:02d}.md"
        file_path = os.path.join(cls._get_report_folder(report_id), file_suffix)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(f"Section saved: {report_id}/{file_suffix}")
        return file_path
    
    @classmethod
    def _clean_section_content(cls, content: str, section_title: str) -> str:
        """
        Normalize section body
        
        1. Drop duplicate title lines
        2. Downgrade ###+ headings to bold
        
        Args:
            content: Raw content
            section_title: Section heading text
            
        Returns:
            Cleaned body
        """
        import re
        
        if not content:
            return content
        
        content = content.strip()
        lines = content.split('\n')
        cleaned_lines = []
        skip_next_empty = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Markdown heading?
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title_text = heading_match.group(2).strip()
                
                # Duplicate of section title in first lines?
                if i < 5:
                    if title_text == section_title or title_text.replace(' ', '') == section_title.replace(' ', ''):
                        skip_next_empty = True
                        continue
                
                # Convert heading to bold
                # System owns the ## title
                cleaned_lines.append(f"**{title_text}**")
                cleaned_lines.append("")  # blank line after bold lead
                continue
            
            # Skip blank after stripped title
            if skip_next_empty and stripped == '':
                skip_next_empty = False
                continue
            
            skip_next_empty = False
            cleaned_lines.append(line)
        
        # Trim leading blanks
        while cleaned_lines and cleaned_lines[0].strip() == '':
            cleaned_lines.pop(0)
        
        # Strip leading rules
        while cleaned_lines and cleaned_lines[0].strip() in ['---', '***', '___']:
            cleaned_lines.pop(0)
            # and blanks after rule
            while cleaned_lines and cleaned_lines[0].strip() == '':
                cleaned_lines.pop(0)
        
        return '\n'.join(cleaned_lines)
    
    @classmethod
    def update_progress(
        cls, 
        report_id: str, 
        status: str, 
        progress: int, 
        message: str,
        current_section: str = None,
        completed_sections: List[str] = None
    ) -> None:
        """
        Write progress.json
        
        UI polls progress.json
        """
        cls._ensure_report_folder(report_id)
        
        progress_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "current_section": current_section,
            "completed_sections": completed_sections or [],
            "updated_at": datetime.now().isoformat()
        }
        
        with open(cls._get_progress_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def get_progress(cls, report_id: str) -> Optional[Dict[str, Any]]:
        """Read progress"""
        path = cls._get_progress_path(report_id)
        
        if not os.path.exists(path):
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @classmethod
    def get_generated_sections(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        List section files
        
        Metadata + content
        """
        folder = cls._get_report_folder(report_id)
        
        if not os.path.exists(folder):
            return []
        
        sections = []
        for filename in sorted(os.listdir(folder)):
            if filename.startswith('section_') and filename.endswith('.md'):
                file_path = os.path.join(folder, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parse index from filename
                parts = filename.replace('.md', '').split('_')
                section_index = int(parts[1])

                sections.append({
                    "filename": filename,
                    "section_index": section_index,
                    "content": content
                })

        return sections
    
    @classmethod
    def assemble_full_report(cls, report_id: str, outline: ReportOutline) -> str:
        """
        Assemble full_report.md
        
        Concatenate saved section files into full_report.md and post-process headings.
        """
        folder = cls._get_report_folder(report_id)
        
        # Header block
        md_content = f"# {outline.title}\n\n"
        md_content += f"> {outline.summary}\n\n"
        md_content += f"---\n\n"
        
        # Concat sections
        sections = cls.get_generated_sections(report_id)
        for section_info in sections:
            md_content += section_info["content"]
        
        # Post-process headings
        md_content = cls._post_process_report(md_content, outline)
        
        # Write full_report.md
        full_path = cls._get_report_markdown_path(report_id)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"Assembled full report: {report_id}")
        return md_content
    
    @classmethod
    def _post_process_report(cls, content: str, outline: ReportOutline) -> str:
        """
        Post-process markdown
        
        1. Drop duplicate headings
        2. Keep # and ##; flatten deeper
        3. Normalize whitespace
        
        Args:
            content: Full markdown
            outline: Report outline
            
        Returns:
            Processed
        """
        import re
        
        lines = content.split('\n')
        processed_lines = []
        prev_was_heading = False
        
        # Known section titles
        section_titles = set()
        for section in outline.sections:
            section_titles.add(section.title)
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Heading line?
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                # Duplicate within window?
                is_duplicate = False
                for j in range(max(0, len(processed_lines) - 5), len(processed_lines)):
                    prev_line = processed_lines[j].strip()
                    prev_match = re.match(r'^(#{1,6})\s+(.+)$', prev_line)
                    if prev_match:
                        prev_title = prev_match.group(2).strip()
                        if prev_title == title:
                            is_duplicate = True
                            break
                
                if is_duplicate:
                    # Skip duplicate block
                    i += 1
                    while i < len(lines) and lines[i].strip() == '':
                        i += 1
                    continue
                
                # Heading levels:
                # - # (level=1) Keep report # title
                # - ## (level=2) Keep ## section
                # - ###+ (level>=3) -> bold
                
                if level == 1:
                    if title == outline.title:
                        # Keep report title
                        processed_lines.append(line)
                        prev_was_heading = True
                    elif title in section_titles:
                        # Normalize mistaken # to ##
                        processed_lines.append(f"## {title}")
                        prev_was_heading = True
                    else:
                        # Else flatten # to bold
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                elif level == 2:
                    if title in section_titles or title == outline.title:
                        # Keep ## section
                        processed_lines.append(line)
                        prev_was_heading = True
                    else:
                        # Other ## -> bold
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                else:
                    # ###+ -> bold
                    processed_lines.append(f"**{title}**")
                    processed_lines.append("")
                    prev_was_heading = False
                
                i += 1
                continue
            
            elif stripped == '---' and prev_was_heading:
                # Drop --- after heading
                i += 1
                continue
            
            elif stripped == '' and prev_was_heading:
                # Single blank after heading
                if processed_lines and processed_lines[-1].strip() != '':
                    processed_lines.append(line)
                prev_was_heading = False
            
            else:
                processed_lines.append(line)
                prev_was_heading = False
            
            i += 1
        
        # Collapse blank runs
        result_lines = []
        empty_count = 0
        for line in processed_lines:
            if line.strip() == '':
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    @classmethod
    def save_report(cls, report: Report) -> None:
        """Save meta + markdown"""
        cls._ensure_report_folder(report.report_id)
        
        # meta.json
        with open(cls._get_report_path(report.report_id), 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        # Save outline
        if report.outline:
            cls.save_outline(report.report_id, report.outline)
        
        # full markdown
        if report.markdown_content:
            with open(cls._get_report_markdown_path(report.report_id), 'w', encoding='utf-8') as f:
                f.write(report.markdown_content)
        
        logger.info(f"Saved report: {report.report_id}")
    
    @classmethod
    def get_report(cls, report_id: str) -> Optional[Report]:
        """Load report"""
        path = cls._get_report_path(report_id)
        
        if not os.path.exists(path):
            # legacy flat json
            old_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
            if os.path.exists(old_path):
                path = old_path
            else:
                return None
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # hydrate Report
        outline = None
        if data.get('outline'):
            outline_data = data['outline']
            sections = []
            for s in outline_data.get('sections', []):
                sections.append(ReportSection(
                    title=s['title'],
                    content=s.get('content', '')
                ))
            outline = ReportOutline(
                title=outline_data['title'],
                summary=outline_data['summary'],
                sections=sections
            )
        
        # backfill from full_report.md
        markdown_content = data.get('markdown_content', '')
        if not markdown_content:
            full_report_path = cls._get_report_markdown_path(report_id)
            if os.path.exists(full_report_path):
                with open(full_report_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
        
        return Report(
            report_id=data['report_id'],
            simulation_id=data['simulation_id'],
            graph_id=data['graph_id'],
            simulation_requirement=data['simulation_requirement'],
            status=ReportStatus(data['status']),
            outline=outline,
            markdown_content=markdown_content,
            created_at=data.get('created_at', ''),
            completed_at=data.get('completed_at', ''),
            error=data.get('error')
        )
    
    @classmethod
    def get_report_by_simulation(cls, simulation_id: str) -> Optional[Report]:
        """Find a report by simulation id."""
        cls._ensure_reports_dir()
        
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # folder layout
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report and report.simulation_id == simulation_id:
                    return report
            # legacy json file
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report and report.simulation_id == simulation_id:
                    return report
        
        return None
    
    @classmethod
    def list_reports(cls, simulation_id: Optional[str] = None, limit: int = 50) -> List[Report]:
        """List reports"""
        cls._ensure_reports_dir()
        
        reports = []
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # folder layout
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
            # legacy json file
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
        
        # newest first
        reports.sort(key=lambda r: r.created_at, reverse=True)
        
        return reports[:limit]
    
    @classmethod
    def delete_report(cls, report_id: str) -> bool:
        """Delete report folder"""
        import shutil
        
        folder_path = cls._get_report_folder(report_id)
        
        # rmtree folder
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            logger.info(f"Deleted folder: {report_id}")
            return True
        
        # legacy files
        deleted = False
        old_json_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
        old_md_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.md")
        
        if os.path.exists(old_json_path):
            os.remove(old_json_path)
            deleted = True
        if os.path.exists(old_md_path):
            os.remove(old_md_path)
            deleted = True
        
        return deleted
