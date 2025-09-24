import warnings
import shutil
warnings.filterwarnings("ignore", category=UserWarning, module="flaml")  # 忽略某些第三方库的 UserWarning
import pyfiglet
from termcolor import cprint
import autogen
import pdb
import json
from pyfiglet import Figlet

# 这四个 Agents 请确保在同路径或正确的 pythonpath 里
from Agents import PerceptionAgent
from Agents import EventAgent
from Agents import ActionAgent
from Agents import PlanAgent

# =================== 示例任务定义 =====================
example_tasks = {
    "1": "Create and recite a poem based on a theme “I love Research” while performing actions that align with the poem’s meaning.",
    "2": "I will speak in English, and you must translate it into Chinese.",
    "3": "Dance a rock-style dance for me. I want you to go all out and dance wildly!",
    "4": "Tell the story of The Three Little Pigs."
}

def print_welcome_banner():
    ascii_art = r"""
     █████╗ ██╗   ██╗████████╗ ██████╗ ███╗   ███╗██╗███████╗████████╗██╗   ██╗
    ██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗████╗ ████║██║██╔════╝╚══██╔══╝╚██╗ ██╔╝
    ███████║██║   ██║   ██║   ██║   ██║██╔████╔██║██║███████╗   ██║    ╚████╔╝
    ██╔══██║██║   ██║   ██║   ██║   ██║██║╚██╔╝██║██║╚════██║   ██║     ╚██╔╝
    ██║  ██║╚██████╔╝   ██║   ╚██████╔╝██║ ╚═╝ ██║██║███████║   ██║      ██║
    ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝ ╚═╝     ╚═╝╚═╝╚══════╝   ╚═╝      ╚═╝
    """
    width = shutil.get_terminal_size().columns
    centered = "\n".join(line.center(width) for line in ascii_art.split("\n"))
    cprint(centered, "white", attrs=["bold"])

    subtitle = " A Multi-Agent LLM Framework for Automated Code Generation in the Misty Social Robot"
    print(subtitle.center(width))
    print("\n" + "=" * width + "\n")

    # 打印使用说明（不带方框）
    print_usage_instructions()
    # 打印示例任务（不带方框）
    print_example_tasks()


def print_usage_instructions():
    from termcolor import colored, cprint

    lines = [
        "HOW TO USE IT",
        "",
        "1. Describe Your Task",
        "   Type in the task you want Misty to perform, using natural language.",
        "",
        "2. Planning Stage",
        "   Misty's Planner will create a high-level plan.",
        "   - If you’d like to revise the plan, just reply with your own version.",
        f"   - If satisfied, type: {colored('ALLSET', 'yellow', attrs=['bold'])}",
        "",
        "3. Agent Execution",
        "   For Action, Touch, and AudioVisual Agents:",
        "   - Code will be auto-generated and shown.",
        f"   - Press {colored('Enter', 'yellow', attrs=['bold'])} to execute it on Misty.",
        "   - The system auto-corrects bugs if errors occur.",
        "   - You may provide feedback to refine behavior.",
        f"   To proceed when satisfied: type {colored('ALLSET', 'yellow', attrs=['bold'])}",
        "",
        "4. AutoMisty Learns Your Personal Style",
        "   If you're especially happy with the result and want to learn the design:",
        f"   - Type {colored('MEM', 'yellow', attrs=['bold'])} to learn it",
        "   - AutoMisty will try to learn your personal preferences, including logic,",
        "     structure, and coding style, so future MistyCode can follow your preference.",
    ]

    # 直接逐行打印
    for line in lines:
        # 若是 "HOW TO USE IT"，则打印为红色粗体
        if line.strip() == "HOW TO USE IT":
            cprint(line, "red", attrs=["bold"])
        else:
            cprint(line, "white")
            

def print_example_tasks():
    from termcolor import cprint

    print("\n")  # 添加两行空行
    cprint("EXAMPLE TASKS (Type 1, 2, 3 or 4):", "green", attrs=["bold"])
    for key, value in example_tasks.items():
        cprint(f"{key} - {value}", "green")

# print_example_tasks()

# 这里调用一下，保证一启动就能看到欢迎横幅 & 示例任务
print_welcome_banner()

# =================== 屏蔽数据库加载日志 =====================
def silent_load_memory():
    pass

# =================== JSON 提取函数 =====================
def extract_json(content):
    if content.startswith("```json"):
        content = content.lstrip("```json").strip()
    if content.endswith("```"):
        content = content.rstrip("```").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("Failed to parse JSON. Content might be incorrectly formatted.")
        return None

# ================ (后续 LLM 配置、Agent 定义、对话逻辑等) ================
config_list = autogen.config_list_from_json(env_or_file="OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list, "cache_seed": None}

api_key = config_list[0].get("api_key", "YOUR_OPENAI_API_KEY_HERE") if config_list else "YOUR_OPENAI_API_KEY_HERE"
misty_ip = config_list[0].get("misty_ip", "67.20.195.181") if config_list else "67.20.195.181"

initializer = autogen.UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    code_execution_config=False,
    default_auto_reply="",
    is_termination_msg=lambda x: True,
)

def state_transition(last_speaker, groupchat):
    """
    决定下一个发言 Agent。
    规则基于 json_content['select_agent']，并按硬逻辑自动推导执行顺序：
      1) 如果包含 ActionAgent，则它必须在最前。
      2) 如果包含 PerceptionAgent，则它必须在最后。
      3) ActionAgent + EventAgent -> ["ActionAgent", "EventAgent"]
      4) EventAgent + PerceptionAgent -> ["EventAgent", "PerceptionAgent"]
      5) ActionAgent + PerceptionAgent -> ["ActionAgent", "PerceptionAgent"]
      6) 三者都有 -> ["ActionAgent", "EventAgent", "PerceptionAgent"]
    """
    messages = getattr(groupchat, "messages", [])
    json_content = None

    # 1) 初始化：initializer -> PlanAgent
    if last_speaker is initializer:
        return PlanAgent

    # 2) 解析第二条消息中的 JSON（你原逻辑如此；保持兼容 extract_json）
    if len(messages) >= 2:
        try:
            json_content = extract_json(messages[1].get("content", ""))
        except Exception:
            print("JSON parsing failed, unable to derive select_agent")
            return None

    # 3) 校验 select_agent
    if not json_content or "select_agent" not in json_content:
        print("Missing select_agent, unable to derive execution sequence")
        return None

    select_agent = json_content["select_agent"]
    if not isinstance(select_agent, list) or len(select_agent) == 0:
        print("select_agent is empty or incorrectly formatted")
        return None

    # 4) 只接受这三个合法 Agent 名称；忽略其他名称
    LEGAL = {"ActionAgent", "EventAgent", "PerceptionAgent"}
    sel_set = {str(x) for x in select_agent if str(x) in LEGAL}
    if not sel_set:
        print("No valid agents in select_agent")
        return None

    # 5) 根据硬规则推导顺序：
    #    简化实现：严格按 [ActionAgent] -> [EventAgent] -> [PerceptionAgent] 的优先级拼接，
    #    恰好满足你列出的 1~6 条硬约束。
    execution_order = []
    if "ActionAgent" in sel_set:
        execution_order.append("ActionAgent")
    if "EventAgent" in sel_set:
        execution_order.append("EventAgent")
    if "PerceptionAgent" in sel_set:
        execution_order.append("PerceptionAgent")

    if not execution_order:
        print("Derived execution_order is empty")
        return None

    # 6) 建立 name -> agent 映射，便于查找对象
    name_to_agent = {}
    for agent in getattr(groupchat, "agents", []):
        # 允许 agent.name 精确等于上述字符串
        name_to_agent[agent.name] = agent

    # 7) 特例：PlanAgent 说完后，交给推导顺序中的第一个
    if last_speaker is PlanAgent:
        first_name = execution_order[0]
        if first_name not in name_to_agent:
            print(f"Could not find `{first_name}` in groupchat.agents")
            return None
        return name_to_agent[first_name]

    # 8) 常规：如果 last_speaker 在推导顺序中，则给下一个；否则报错
    if getattr(last_speaker, "name", None) in execution_order:
        idx = execution_order.index(last_speaker.name)
        if idx == len(execution_order) - 1:
            # 到最后一个了，结束对话
            return None
        next_name = execution_order[idx + 1]
        if next_name not in name_to_agent:
            print(f"Could not find `{next_name}` in groupchat.agents")
            return None
        return name_to_agent[next_name]

    print(f"`{getattr(last_speaker, 'name', repr(last_speaker))}` is not in derived execution_order, unable to determine next speaker")
    return None

  

groupchat = autogen.GroupChat(
    agents=[initializer, PlanAgent, ActionAgent, EventAgent, PerceptionAgent],
    messages=[],
    max_round=20,
    speaker_selection_method=state_transition,
)

AUTOMISTY = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# =================== 任务输入 =====================
print()  # 插入一行空行
cprint("Please enter the task for Misty to perform (or type 1/2/3/4 to select an example task):", "red", attrs=["bold"])
user_input = input(">>> ").strip()

if user_input in example_tasks:
    mytask = example_tasks[user_input]
    print(f"Selected Example Task {user_input}: {mytask}")
else:
    mytask = user_input
    print(f"Custom task input: {mytask}")

system_message_template = (
    "{tasks}\n"
    "Misty_IP: ({ip_address})\n"
    "API_Key: ({api_key})\n"
)




chat_result = initializer.initiate_chat(
    AUTOMISTY,
    message=system_message_template.format(
        ip_address=misty_ip,
        tasks=mytask,
        api_key=api_key  # 从OAI_CONFIG_LIST.json加载
    )
)
