import os
import autogen
from .Misty_society_of_mind import SocietyOfMindAgent  # noqa: E402
from .Misty_Plan_teachability import Teachability
from typing import List, Dict
import pdb

SystemMessage="""
ou are a professional system architect. All planning must follow software design principles and the Single Responsibility Principle. Your output must be one and only one JSON block, with no extra text, explanations, or Markdown.

[Fixed Rules]

If and only if the user message equals MEM (exact case match, no leading or trailing whitespace), your only reply is: PLANAPPROVED.

Each Agent may appear at most once in a single output (merge same-type high-level items into the corresponding array).

Decouple strictly: each Agent is responsible only for high-level goals within its own boundary—no cross-boundary work and no orchestration of other Agents.

Keep it strictly "high-level requirements"—no implementation details, parameters, numbers, step orders, callbacks, thresholds, color codes, angles, speeds, etc.

Misty_IP and API_KEY must be taken verbatim from user input; do not modify, mask, or generate them. If missing or invalid, enter the error branch.

Fields are restricted to the specified set with fixed order; select_agent must match the set of non-empty Agent arrays (deduplicated).

[Agent Boundaries]

ActionAgent: High-level intent for emotions and actions (arms/head/LED/facial expression/audio/TTS, etc.). No parameters or programmatic details.

EventAgent (touch sensor only): High-level description of subscribing/detecting touch events and trigger conditions. Must not dispatch/orchestrate other Agents.

PerceptionAgent: "Seeing and hearing" abstraction here. High-level goals for RTSP-based real-time audio/video acquisition and understanding (photo, recording, dialogue, image/text understanding, etc.). May state "use an LLM for understanding", but no model or pipeline details.

[Requirement Abstraction]

Read the user request and distill a one-sentence high-level intent into Task.

Split the intent into concise imperative high-level items for each applicable Agent (one per line, avoid details).

[Conflict Resolution]

If it includes "trigger + expression": triggering (touch) goes to EventAgent; expression goes to ActionAgent. Each writes its own high-level item, without "via/call/order" wording.

If it includes "understand first then express": PerceptionAgent writes the high-level goal of "acquire and understand"; ActionAgent writes the high-level goal of "express based on what has been understood"; do not describe ordering or orchestration.

[Success Path — the only permitted output JSON template (fixed field order)]
{
"Misty_IP": "xxx.xxx.xxx.xxx",
"API_KEY": "********",
"Task": "",
"ActionAgent": [],
"EventAgent": [],
"PerceptionAgent": [],
"select_agent": []
}

Task: one-sentence high-level restatement of the user intent.

Three Agent arrays: fill with corresponding high-level imperative items; arrays may be empty but fields must exist.

select_agent: choose from ["ActionAgent","EventAgent","PerceptionAgent"], and it must equal the set of all non-empty Agent arrays.

[Example 1] User: "I need you to act very excited"
{
"Misty_IP": "xxx.xxx.xxx.xxx",
"API_KEY": "********",
"Task": "Perform an excited action.",
"ActionAgent": [
"Express excitement."
],
"EventAgent": [],
"PerceptionAgent": [],
"select_agent": ["ActionAgent"]
}

[Example 2] User: "When I touch your head, act angry"
{
"Misty_IP": "xxx.xxx.xxx.xxx",
"API_KEY": "********",
"Task": "Express anger when the head is touched.",
"ActionAgent": [
"Express anger."
],
"EventAgent": [
"Detect head touch."
],
"PerceptionAgent": [],
"select_agent": ["ActionAgent", "EventAgent"]
}


"""

misty_plan_reflection_message="""
You are the JSON auditor for outputs produced by "misty_draft_plan_assistant".
Scope: static checks only; never request edits or masking.

— Output Rules —
• If EVERYTHING is compliant → reply ONLY: PLANAPPROVED
• If ANY issue exists → reply only the following sections (include only those that apply):
JSON format issues:
<list each violation>
Responsibility boundary issues:
<list each problem>

— Zero-Interference & Privacy —
• Treat "Misty_IP" and "API_KEY" as opaque strings. Verify TYPE & PRESENCE only.
• NEVER ask to modify, mask, or unmask values (e.g., do NOT say “replace with *****”).
• If the reviewed JSON already uses "********", judge it only by type/presence; do not demand changes.
• Any suggestion to alter values (masking/unmasking) = violation → report under JSON format issues: "越权修改值/要求掩码".

— I. Hard JSON Format Spec (all must pass) —
The reviewed value MUST be a single top-level JSON object with EXACT keys and order:
{
"Misty_IP": "xxx.xxx.xxx.xxx",
"API_KEY": "********",
"Task": "",
"ActionAgent": [],
"EventAgent": [],
"PerceptionAgent": [],
"select_agent": []
}
Details:
• Fixed key order exactly as above. No extra/missing keys; no nested structures.
• Types:
  - "Misty_IP": string
  - "API_KEY": string
  - "Task": string (one-sentence high-level intent)
  - "ActionAgent","EventAgent","PerceptionAgent": arrays of NON-EMPTY STRINGs
  - "select_agent": array of strings chosen from ["ActionAgent","EventAgent","PerceptionAgent"]
• Presence: all fields must exist. Arrays may be empty; null/undefined forbidden.
• Forbidden: nulls; non-string elements in arrays; objects/numbers inside arrays; any extra fields.

— II. select_agent Consistency —
Let S = {agent name | the corresponding Agent array has ≥1 non-empty string}.
The set in "select_agent" MUST equal S (order not enforced; no extras/missing).

— III. Responsibility Boundaries (high-level only) —
• ActionAgent: only high-level emotion/action intent (arms, head, LED, facial, audio/TTS). No capture/record/understanding.
• EventAgent (touch sensor only): high-level subscription/detection/trigger for touch events. No scheduling/controlling other agents.
• PerceptionAgent: RTSP-based real-time A/V acquisition & understanding (photo/record/dialogue/image/text understanding). May say “use an LLM for understanding”, but no model/param/pipeline specifics.



"""





import os
config_list = autogen.config_list_from_json(env_or_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "OAI_CONFIG_LIST.json"))
llm_config = {"config_list": config_list, "cache_seed": None}
misty_draft_plan_assistant = autogen.ConversableAgent(
    name="misty_draft_plan_assistant",
    llm_config=llm_config,
    system_message=SystemMessage
)
# Instantiate the Teachability capability. Its parameters are all optional.
misty_draft_plan_assistant_teachability = Teachability(
    verbosity=1,  # 0 for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
    reset_db=False,  # If True, clears the memory database.
    path_to_db_dir="./DB/misty_plan_db",
    recall_threshold=0.5,  # Higher numbers allow more (but less relevant) memos to be recalled.
)

misty_draft_plan_assistant_teachability.add_to_agent(misty_draft_plan_assistant)




misty_plan_reflection_assistant=autogen.ConversableAgent(
    name="misty_plan_reflection_assistant",
    llm_config=llm_config,
    system_message=misty_plan_reflection_message,
)

misty_draft_plan_groupchat = autogen.GroupChat(
    agents=[misty_draft_plan_assistant, misty_plan_reflection_assistant],
    messages=[],
    speaker_selection_method="round_robin", 
    allow_repeat_speaker=False,
    max_round=5,
)
#  llm_config=llm_config,
misty_draft_plan_manager = autogen.GroupChatManager(
    name="misty_draft_plan_manager",
    groupchat=misty_draft_plan_groupchat,
    llm_config=llm_config,
    is_termination_msg=lambda x: x.get("content", "").find("PLANAPPROVED") >= 0, 

)


draft_plan_response_preparer ="Extract the final generated json from our conversation and respond with it exactly as it is, WITHOUT MAKING ANY MODIFICATIONS.When you receive a 'MEM' message, you must reply with the previous json without any change."
Draft_Plan= SocietyOfMindAgent(
    "Draft_Plan",
    chat_manager=misty_draft_plan_manager,
    llm_config=llm_config,
    response_preparer=draft_plan_response_preparer,
    
)



misty_plan_auditor = autogen.UserProxyAgent(
    name="plan_auditor",
    human_input_mode="ALWAYS",
    code_execution_config={
        "work_dir": "code/mistyPy", 
        "use_docker": False,
    },
    default_auto_reply="",
)


misty_plan_groupchat = autogen.GroupChat(
    agents=[Draft_Plan, misty_plan_auditor],
    messages=[],
    speaker_selection_method="round_robin",  # With two agents, this is equivalent to a 1:1 conversation.
    allow_repeat_speaker=False,
    max_round=50,
)
misty_plan_manager = autogen.GroupChatManager(
   name="misty_plan_manager",
    groupchat=misty_plan_groupchat,
    is_termination_msg=lambda x: x.get("content", "").find("ALLSET") >= 0, 
    llm_config=llm_config,
)



PlanAgent_response_preparer ="Extract the final generated json from our conversation and respond with it exactly as it is, WITHOUT MAKING ANY MODIFICATIONS."
PlanAgent = SocietyOfMindAgent(
    "PlanAgent",
    chat_manager=misty_plan_manager,
    llm_config=llm_config,
    response_preparer=PlanAgent_response_preparer,
)



