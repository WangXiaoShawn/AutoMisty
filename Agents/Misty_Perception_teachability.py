# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import os
import pickle
from typing import Optional, Union
import re
from .formatting_utils import colored
from .import_utils  import optional_import_block, require_optional_import
from autogen import ConversableAgent
from autogen import GroupChatManager
from .text_analyzer import TextAnalyzerAgent
from .agent_capability import AgentCapability
import pdb

with optional_import_block():
    import chromadb
    from chromadb.config import Settings



def extract_tasks_as_string(input_text):
   
    match = re.search(r"(?i)YOURTASK\s*:\s*(.*)", input_text, re.DOTALL)
    
    if match:
        tasks_text = match.group(1).strip()  # 提取任务部分并去除前后空格
        return tasks_text
    else:
        return None 
    

def extract_python_codeblock(text):
    """
    Extracts the code block enclosed within ```python and ``` from a given text.
    
    Parameters:
    text (str): Input text containing the code block.
    
    Returns:
    str: Extracted code block including the markdown formatting.
    """
    pattern = r'(```python[\s\S]*?```)'
    match = re.search(pattern, text)
    
    if match:
        return match.group(1)
    else:
        return None
    
    


class Teachability(AgentCapability):
    """Teachability uses a vector database to give an agent the ability to remember user teachings,
    where the user is any caller (human or not) sending messages to the teachable agent.
    Teachability is designed to be composable with other agent capabilities.
    To make any conversable agent teachable, instantiate both the agent and the Teachability class,
    then pass the agent to teachability.add_to_agent(agent).
    Note that teachable agents in a group chat must be given unique path_to_db_dir values.

    When adding Teachability to an agent, the following are modified:
    - The agent's system message is appended with a note about the agent's new ability.
    - A hook is added to the agent's `process_last_received_message` hookable method,
    and the hook potentially modifies the last of the received messages to include earlier teachings related to the message.
    Added teachings do not propagate into the stored message history.
    If new user teachings are detected, they are added to new memos in the vector database.
    """

    def __init__(
        self,
        verbosity: Optional[int] = 0,
        reset_db: Optional[bool] = False,
        path_to_db_dir: Optional[str] = "./tmp/teachable_agent_db",
        recall_threshold: Optional[float] = 1.5,
        max_num_retrievals: Optional[int] = 10,
        llm_config: Optional[Union[dict, bool]] = None,
    ):
        super().__init__()
        """Args:
        verbosity (Optional, int): # 0 (default) for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
        reset_db (Optional, bool): True to clear the DB before starting. Default False.
        path_to_db_dir (Optional, str): path to the directory where this particular agent's DB is stored. Default "./tmp/teachable_agent_db"
        recall_threshold (Optional, float): The maximum distance for retrieved memos, where 0.0 is exact match. Default 1.5. Larger values allow more (but less relevant) memos to be recalled.
        max_num_retrievals (Optional, int): The maximum number of memos to retrieve from the DB. Default 10.
        llm_config (dict or False): llm inference configuration passed to TextAnalyzerAgent.
            If None, TextAnalyzerAgent uses llm_config from the teachable agent.
        """
        self.verbosity = verbosity
        self.path_to_db_dir = path_to_db_dir
        self.recall_threshold = recall_threshold
        self.max_num_retrievals = max_num_retrievals
        self.llm_config = llm_config

        self.analyzer = None
        self.teachable_agent = None
        self.Perception_Agent_Task = None 
        self.has_done_retrieval = False # only retrival once
        self.expanded_text = None
        self.prev_agent_code = None
        self.memo_store = MemoStore(self.verbosity, reset_db, self.path_to_db_dir)

    def add_to_agent(self, agent: ConversableAgent):
        """Adds teachability to the given agent."""
        # pdb.set_trace()
        self.teachable_agent = agent

        # Register a hook for processing the last message.
        #process_all_messages_before_reply
        #agent.register_hook(hookable_method="process_last_received_message", hook=self.process_last_received_message)
        #  "process_all_messages_before_reply": [],
        #     "process_message_before_send": [],
        
        # agent.register_hook(hookable_method="process_last_received_message", hook=self.process_last_received_message) #changed here
        agent.register_hook(hookable_method="process_all_messages_before_reply", hook=self.process_all_messages_before_reply)

        # Was an llm_config passed to the constructor?
        if self.llm_config is None:
            # No. Use the agent's llm_config.
            self.llm_config = agent.llm_config
        assert self.llm_config, "Teachability requires a valid llm_config."

        # Create the analyzer agent.
        self.analyzer = TextAnalyzerAgent(llm_config=self.llm_config)

        # Append extra info to the system message.
        agent.update_system_message(
            agent.system_message
            + "\nYou've been given the special ability to remember user teachings from prior conversations."
        )

    def prepopulate_db(self):
        """Adds a few arbitrary memos to the DB."""
        self.memo_store.prepopulate()
        
    #
    def process_all_messages_before_reply(self, text: Union[dict, str, list]): # 这里的逻辑不对
        # 思路是获取任务 然后制造retrival 最后拼接成格式然后送出去 必须
        # pdb.set_trace() 
        #### get task
    
        if isinstance(text, list):
            formatted_text = " | ".join(
                f"{msg.get('role', 'unknown')}: {msg.get('name', 'unknown')} -> {msg.get('content', '')}"
                for msg in text if isinstance(msg, dict)
            )
            
            tasks= [
                msg.get('content', '') for msg in text if isinstance(msg, dict) and msg.get('name') == 'Draft_perception_Code'
            ]
             # **只在第一次赋值**
            if self.Perception_Agent_Task is None and tasks:
                self.Perception_Agent_Task = tasks[0]  # 只赋值一次，后续不修改
                self.prev_agent_code = extract_python_codeblock(self.Perception_Agent_Task)
                #pdb.set_trace()
            
        elif isinstance(text, dict):
            formatted_text = f"{text.get('role', 'unknown')}: {text.get('name', 'unknown')} -> {text.get('content', '')}"
             # **只在第一次赋值**
            if self.Perception_Agent_Task is None and text.get('name') == 'Draft_perception_Code':
                self.Perception_Agent_Task = text.get('content', '')  # 只赋值一次
        else:
            formatted_text = str(text)
            if self.Perception_Agent_Task is None :
                self.Perception_Agent_Task = formatted_text
        #### memory logic
        #pdb.set_trace()
        self._consider_memo_storage(formatted_text)
        #pdb.set_trace()
        #### retrieval logic
        if not self.has_done_retrieval and self.memo_store.last_memo_id> 0:
                self.expanded_text = self._consider_memo_retrieval( self.Perception_Agent_Task)
                text[1]['content'] = self.expanded_text
                self.has_done_retrieval = True
        elif self.has_done_retrieval:
            text[1]['content'] = self.expanded_text
            text[-1]['content'] += f"\nPrevious Agent code you must consider:\n{self.prev_agent_code}"

        return text
        
    def _consider_memo_storage(self, comment: Union[dict, str]): # 这里的逻辑不对 
        """Decides whether to store something from one user comment in the DB."""
        # pdb.set_trace()
        memo_added = False
        
       
        response = "yes" if "MEM" in comment else "no"
        # pdb.set_trace()        
        if "yes" in response.lower():
            # Can we extract advice?
            #pdb.set_trace()
            # Gen_code = self._analyze(
            #     comment,
            #     "Give me the final version of the code that satisfies / was approved by the user. Don't make any changes to the code. If there's no code, just respond with 'CodeNotExist1926817'",
            # ) # 这个会有个问题就是 None是代码的关键字 很多代码都有这样会导致失效
            Gen_code = self._analyze(
                comment,
                "Give me the final version of the code that satisfies / was approved by the user. Don't make any changes to the code. Make sure to enclose the code block with ```python. If there's no code, just respond with 'codenotexist1926817'.",
            )
            # pdb.set_trace()
            if "codenotexist1926817" not in Gen_code.lower():
                    #Summarize very briefly, in general terms, the type of task described in the TEXT. Leave out details that might not appear in a similar problem
                    # Generalize the task.
                    # pdb.set_trace()
                general_task = self._analyze(
                    self.Perception_Agent_Task,
                    "You must summarize the given Task concisely. You must provide a highly condensed summary. You must not analyze or mention any API_KEY, IP, or code. You must only focus on the core content of the Task，NO MORE THAN FIVE WORDS",
                                    )
                # Add the task-advice (problem-solution) pair to the vector DB.
                if self.verbosity >= 1:
                    print(colored("\nREMEMBER THIS TASK-ADVICE PAIR", "light_yellow"))
                self.memo_store.add_input_output_pair(general_task, Gen_code)
                memo_added = True
        if memo_added:
            # Yes. Save them to disk.
            self.memo_store._save_memos()
        

    def _consider_memo_retrieval(self, comment: Union[dict, str]):
        # pdb.set_trace()
        group_manager = None
        for agent, messages in self.teachable_agent.chat_messages.items():
            if isinstance(agent, GroupChatManager):
                group_manager = agent
                break
        """Decides whether to retrieve memos from the DB, and add them to the chat context."""
        # First, use the comment directly as the lookup key.
        if self.verbosity >= 1:
            print(colored("\nLOOK FOR RELEVANT MEMOS, AS QUESTION-ANSWER PAIRS", "light_yellow"))
            
        general_task = self._analyze(
                    comment,
                    "You must summarize the given Task concisely. You must provide a highly condensed summary. You must not analyze or mention any API_KEY, IP, or code. You must only focus on the core content of the Task, NO MORE THAN FIVE WORDS",
                                    )
        #pdb.set_trace()
        # 先泛化再retrieve
        print("******************************************************")
        print("General Task: ", general_task)
        print("******************************************************")
        memo_list = self._retrieve_relevant_memos(general_task)
        print("******************************************************")
        print("Example retreived help solve our task: ", comment + self._concatenate_memo_texts(memo_list))
        print("******************************************************")
        return comment + self._concatenate_memo_texts(memo_list)

    def _retrieve_relevant_memos(self, input_text: str) -> list:
        """Returns semantically related memos from the DB."""
        memo_list = self.memo_store.get_related_memos(
            input_text, n_results=self.max_num_retrievals, threshold=self.recall_threshold
        )

        if self.verbosity >= 1:  # noqa: SIM102
            # Was anything retrieved?
            if len(memo_list) == 0:
                # No. Look at the closest memo.
                print(colored("\nTHE CLOSEST MEMO IS BEYOND THE THRESHOLD:", "light_yellow"))
                self.memo_store.get_nearest_memo(input_text)
                print()  # Print a blank line. The memo details were printed by get_nearest_memo().

        # Create a list of just the memo output_text strings.
        memo_list = [memo[1] for memo in memo_list]
        return memo_list

    def _concatenate_memo_texts(self, memo_list: list) -> str:
        """Concatenates the memo texts into a single string for inclusion in the chat context."""
        memo_texts = ""
        if len(memo_list) > 0:
            info = "\n# Some Example may help you to finish you task：\n"
            for memo in memo_list:
                info = info + "- " + memo + "\n"
            if self.verbosity >= 1:
                print(colored("\nMEMOS APPENDED TO LAST MESSAGE...\n" + info + "\n", "light_yellow"))
            memo_texts = memo_texts + "\n" + info
        return memo_texts

    def _analyze(self, text_to_analyze: Union[dict, str], analysis_instructions: Union[dict, str]):
        """Asks TextAnalyzerAgent to analyze the given text according to specific instructions."""
        self.analyzer.reset()  # Clear the analyzer's list of messages.
        self.teachable_agent.send(
            recipient=self.analyzer, message=text_to_analyze, request_reply=False, silent=(self.verbosity < 2)
        )  # Put the message in the analyzer's list.
        self.teachable_agent.send(
            recipient=self.analyzer, message=analysis_instructions, request_reply=True, silent=(self.verbosity < 2)
        )  # Request the reply.
        return self.teachable_agent.last_message(self.analyzer)["content"]


@require_optional_import("chromadb", "teachable")
class MemoStore:
    """Provides memory storage and retrieval for a teachable agent, using a vector database.
    Each DB entry (called a memo) is a pair of strings: an input text and an output text.
    The input text might be a question, or a task to perform.
    The output text might be an answer to the question, or advice on how to perform the task.
    Vector embeddings are currently supplied by Chroma's default Sentence Transformers.
    """

    def __init__(
        self,
        verbosity: Optional[int] = 0,
        reset: Optional[bool] = False,
        path_to_db_dir: Optional[str] = "./tmp/teachable_agent_db",
    ):
        """Args:
        - verbosity (Optional, int): 1 to print memory operations, 0 to omit them. 3+ to print memo lists.
        - reset (Optional, bool): True to clear the DB before starting. Default False.
        - path_to_db_dir (Optional, str): path to the directory where the DB is stored.
        """
        self.verbosity = verbosity
        self.path_to_db_dir = path_to_db_dir

        # Load or create the vector DB on disk.
        settings = Settings(
            anonymized_telemetry=False, allow_reset=True, is_persistent=True, persist_directory=path_to_db_dir
        )
        self.db_client = chromadb.Client(settings)
        self.vec_db = self.db_client.create_collection("memos", get_or_create=True)  # The collection is the DB.
        # Load or create the associated memo dict on disk.
        self.path_to_dict = os.path.join(path_to_db_dir, "uid_text_dict.pkl")
        self.uid_text_dict = {}
        self.last_memo_id = 0
        if (not reset) and os.path.exists(self.path_to_dict):
            # print(colored("\nLOADING MEMORY FROM DISK", "light_green"))
            # print(colored(f"    Location = {self.path_to_dict}", "light_green"))
            with open(self.path_to_dict, "rb") as f:
                self.uid_text_dict = pickle.load(f)
                self.last_memo_id = len(self.uid_text_dict)
                if self.verbosity >= 3:
                    self.list_memos()

        # Clear the DB if requested.
        if reset:
            self.reset_db()

    def list_memos(self):
        """Prints the contents of MemoStore."""
        print(colored("LIST OF MEMOS", "light_green"))
        for uid, text in self.uid_text_dict.items():
            input_text, output_text = text
            print(
                colored(
                    f"  ID: {uid}\n    INPUT TEXT: {input_text}\n    OUTPUT TEXT: {output_text}",
                    "light_green",
                )
            )

    def _save_memos(self):
        """Saves self.uid_text_dict to disk."""
        with open(self.path_to_dict, "wb") as file:
            pickle.dump(self.uid_text_dict, file)

    def reset_db(self):
        """Forces immediate deletion of the DB's contents, in memory and on disk."""
        print(colored("\nCLEARING MEMORY", "light_green"))
        self.db_client.delete_collection("memos")
        self.vec_db = self.db_client.create_collection("memos")
        self.uid_text_dict = {}
        self._save_memos()

    def add_input_output_pair(self, input_text: str, output_text: str):
        """Adds an input-output pair to the vector DB."""
        self.last_memo_id += 1
        self.vec_db.add(documents=[input_text], ids=[str(self.last_memo_id)])
        self.uid_text_dict[str(self.last_memo_id)] = input_text, output_text
        if self.verbosity >= 1:
            print(
                colored(
                    f"\nINPUT-OUTPUT PAIR ADDED TO VECTOR DATABASE:\n  ID\n    {self.last_memo_id}\n  INPUT\n    {input_text}\n  OUTPUT\n    {output_text}\n",
                    "light_yellow",
                )
            )
        if self.verbosity >= 3:
            self.list_memos()

    def get_nearest_memo(self, query_text: str):
        """Retrieves the nearest memo to the given query text."""
        results = self.vec_db.query(query_texts=[query_text], n_results=1)
        uid, input_text, distance = results["ids"][0][0], results["documents"][0][0], results["distances"][0][0]
        input_text_2, output_text = self.uid_text_dict[uid]
        assert input_text == input_text_2
        if self.verbosity >= 1:
            print(
                colored(
                    f"\nINPUT-OUTPUT PAIR RETRIEVED FROM VECTOR DATABASE:\n  INPUT1\n    {input_text}\n  OUTPUT\n    {output_text}\n  DISTANCE\n    {distance}",
                    "light_yellow",
                )
            )
        return input_text, output_text, distance

    def get_related_memos(self, query_text: str, n_results: int, threshold: Union[int, float]):
        """Retrieves memos that are related to the given query text within the specified distance threshold."""
        if n_results > len(self.uid_text_dict):
            n_results = len(self.uid_text_dict)
        results = self.vec_db.query(query_texts=[query_text], n_results=n_results)
        memos = []
        num_results = len(results["ids"][0])
        for i in range(num_results):
            uid, input_text, distance = results["ids"][0][i], results["documents"][0][i], results["distances"][0][i]
            if distance < threshold:
                input_text_2, output_text = self.uid_text_dict[uid]
                assert input_text == input_text_2
                if self.verbosity >= 1:
                    print(
                        colored(
                            f"\nINPUT-OUTPUT PAIR RETRIEVED FROM VECTOR DATABASE:\n  INPUT1\n    {input_text}\n  OUTPUT\n    {output_text}\n  DISTANCE\n    {distance}",
                            "light_yellow",
                        )
                    )
                memos.append((input_text, output_text, distance))
        return memos

    def prepopulate(self):
        """Adds a few arbitrary examples to the vector DB, just to make retrieval less trivial."""
        if self.verbosity >= 1:
            print(colored("\nPREPOPULATING MEMORY", "light_green"))
        examples = []
        examples.append({"text": "When I say papers I mean research papers, which are typically pdfs.", "label": "yes"})
        examples.append({"text": "Please verify that each paper you listed actually uses langchain.", "label": "no"})
        examples.append({"text": "Tell gpt the output should still be latex code.", "label": "no"})
        examples.append({"text": "Hint: convert pdfs to text and then answer questions based on them.", "label": "yes"})
        examples.append(
            {"text": "To create a good PPT, include enough content to make it interesting.", "label": "yes"}
        )
        examples.append(
            {
                "text": "No, for this case the columns should be aspects and the rows should be frameworks.",
                "label": "no",
            }
        )
        examples.append({"text": "When writing code, remember to include any libraries that are used.", "label": "yes"})
        examples.append({"text": "Please summarize the papers by Eric Horvitz on bounded rationality.", "label": "no"})
        examples.append({"text": "Compare the h-index of Daniel Weld and Oren Etzioni.", "label": "no"})
        examples.append(
            {
                "text": "Double check to be sure that the columns in a table correspond to what was asked for.",
                "label": "yes",
            }
        )
        for example in examples:
            self.add_input_output_pair(example["text"], example["label"])
        self._save_memos()
