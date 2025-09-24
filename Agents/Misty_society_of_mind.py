# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# ruff: noqa: E722
import copy
import traceback
from contextlib import suppress
from typing import Callable, Literal, Optional, Union

from autogen import Agent, ConversableAgent, GroupChat, GroupChatManager, OpenAIWrapper
import pdb

class SocietyOfMindAgent(ConversableAgent):
    """(In preview) A single agent that runs a Group Chat as an inner monologue.
    At the end of the conversation (termination for any reason), the SocietyOfMindAgent
    applies the response_preparer method on the entire inner monologue message history to
    extract a final answer for the reply.

    Most arguments are inherited from ConversableAgent. New arguments are:
        chat_manager (GroupChatManager): the group chat manager that will be running the inner monologue
        response_preparer (Optional, Callable or String): If response_preparer is a callable function, then
                it should have the signature:
                    f( self: SocietyOfMindAgent, messages: List[Dict])
                where `self` is this SocietyOfMindAgent, and `messages` is a list of inner-monologue messages.
                The function should return a string representing the final response (extracted or prepared)
                from that history.
                If response_preparer is a string, then it should be the LLM prompt used to extract the final
                message from the inner chat transcript.
                The default response_preparer depends on if an llm_config is provided. If llm_config is False,
                then the response_preparer deterministically returns the last message in the inner-monolgue. If
                llm_config is set to anything else, then a default LLM prompt is used.
    """

    def __init__(
        self,
        name: str,
        chat_manager: GroupChatManager,
        response_preparer: Optional[Union[str, Callable]] = None,
        is_termination_msg: Optional[Callable[[dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = "TERMINATE",
        function_map: Optional[dict[str, Callable]] = None,
        code_execution_config: Union[dict, Literal[False]] = False,
        llm_config: Optional[Union[dict, Literal[False]]] = False,
        default_auto_reply: Optional[Union[str, dict, None]] = "",
        **kwargs,
    ):
        super().__init__(
            name=name,
            system_message="When you recive a message MEMDONE you must respond with only one work ALLSET.",
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            function_map=function_map,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply,
            **kwargs,
        )

        self.update_chat_manager(chat_manager)

        # response_preparer default depends on if the llm_config is set, and if a client was created
        if response_preparer is None:
            if self.client is not None:
                response_preparer = "Output a standalone response to the original request, without mentioning any of the intermediate discussion."
            else:

                def response_preparer(agent, messages):
                    return messages[-1]["content"].replace("TERMINATE", "").strip() #没有llm直接处理最后一条信息

        # Create the response_preparer callable, if given only a prompt string
        # self.response_preparer 一个函数，它接收两个参数：agent 和 messages。其中：
        if isinstance(response_preparer, str):
            self.response_preparer = lambda agent, messages: agent._llm_response_preparer(response_preparer, messages)#它接收两个参数：agent 和 messages。其中：
        else:
            self.response_preparer = response_preparer

        # NOTE: Async reply functions are not yet supported with this contrib agent
        #self._reply_func_list = []
        # self.register_reply([Agent, None], SocietyOfMindAgent.generate_inner_monologue_reply)
        #[Agent, None] trigger
        self.register_reply([Agent, None], SocietyOfMindAgent.generate_inner_monologue_reply) #独立的
        self.register_reply([Agent, None], ConversableAgent.generate_code_execution_reply)
        self.register_reply([Agent, None], ConversableAgent.generate_function_call_reply)
        self.register_reply([Agent, None], ConversableAgent.check_termination_and_human_reply) #这个是继承的

    def _llm_response_preparer(self, prompt, messages):
        """Default response_preparer when provided with a string prompt, rather than a callable.
            这是一个私有方法，用于处理当 response_preparer 是字符串（而非自定义函数）时的默认处理方式。
        Args:
            prompt (str): The prompt used to extract the final response from the transcript.
            messages (list): The messages generated as part of the inner monologue group chat.
        """
        _messages = [
            {
                "role": "system",
                "content": """Earlier you were asked to fulfill a request. You and your team worked diligently to address that request. Here is a transcript of that conversation:""",
            }
        ]
        # pdb.set_trace()
        for message in messages:
            message = copy.deepcopy(message)
            message["role"] = "user"
            # Convert tool and function calls to basic messages to avoid an error on the LLM call
            if "content" not in message:
                message["content"] = ""
            if "tool_calls" in message:
                del message["tool_calls"]
            if "tool_responses" in message:
                del message["tool_responses"]
            if "function_call" in message and message["content"] == "":
                with suppress(KeyError):
                    message["content"] = (
                        message["function_call"]["name"] + "(" + message["function_call"]["arguments"] + ")"
                    )
                del message["function_call"]
            # Add the modified message to the transcript
            # pdb.set_trace()
            _messages.append(message)
# _messages = [
#         {
#             "role": "system",
#             "content": "Earlier you were asked to fulfill a request. You and your team worked diligently to address that request. Here is a transcript of that conversation:"
#         },
#         {
#             "role": "user",
#             "name": "society_of_mind",
#             "content": "On which days in 2024 was Microsoft Stock higher than $370?"
#         },
#         {
#             "role": "user",
#             "name": "inner-assistant",
#             "content": "To determine on which days in 2024 Microsoft's stock was higher than $370, you'll need access to historical stock price data for that year. One way to get this data is through financial websites or APIs. However, I can guide you on using Python code with a library like `pandas` or financial data APIs like Yahoo Finance to get this data.\n\nHere’s a plan:\n\n1. **Use Python with a Financial Data Library:**\n   - We'll use the `yfinance` library to fetch historical stock data for Microsoft.\n   - Filter the data for entries where the stock price was higher than $370.\n\nLet's execute the code:\n\n```python\n# filename: msft_stock_above_370.py\n\nimport yfinance as yf\nimport pandas as pd\n\n# Get Microsoft historical stock data for 2024\nmsft = yf.Ticker(\"MSFT\")\nhist = msft.history(start=\"2024-01-01\", end=\"2024-12-31\")\n\n# Filter the data for days when closing price was higher than $370\nover_370 = hist[hist['Close'] > 370]\n\n# Print the dates and closing prices\nprint(over_370[['Close']])\n```\n\n1. Run this code to get a list of dates when Microsoft's stock closed above $370 in 2024.\n2. Check the output to confirm the specific days.\n\nExecute the code and provide the result, then we can analyze the output to give a concise answer."
#         },
#         {
#             "role": "user",
#             "name": "inner-code-interpreter",
#             "content": "exitcode: 0 (execution succeeded)\nCode output: \n                                Close\nDate                                 \n2024-01-08 00:00:00-05:00  371.908905\n2024-01-09 00:00:00-05:00  373.000732\n2024-01-10 00:00:00-05:00  379.928894\n2024-01-11 00:00:00-05:00  381.775146\n2024-01-12 00:00:00-05:00  385.586609\n...                               ...\n2024-12-23 00:00:00-05:00  435.250000\n2024-12-24 00:00:00-05:00  439.329987\n2024-12-26 00:00:00-05:00  438.109985\n2024-12-27 00:00:00-05:00  430.529999\n2024-12-30 00:00:00-05:00  424.829987\n\n[247 rows x 1 columns]\n"
#         },
#         {
#             "role": "user",
#             "name": "inner-assistant",
#             "content": "Based on the fetched data, Microsoft stock was higher than $370 on the following dates in 2024:\n\n- January 8, 2024\n- January 9, 2024\n- January 10, 2024\n- January 11, 2024\n- January 12, 2024\n- ... and many other days throughout the year.\n\nIn total, there were 247 days where Microsoft's stock closed above $370 in 2024. The data would include all such instances across the year.\n\nIf you need any more specific analysis, feel free to ask!\n\nTERMINATE"
#         }
#     ]

      
        _messages.append(
            {
                "role": "system",
                "content": prompt,
            }
        )
#         [
#   {
#     "role": "system",
#     "content": "Earlier you were asked to fulfill a request. You and your team worked diligently to address that request. Here is a transcript of that conversation:"
#   },
#   {
#     "role": "user",
#     "name": "society_of_mind",
#     "content": "On which days in 2024 was Microsoft Stock higher than $370?"
#   },
#   {
#     "role": "user",
#     "name": "inner-assistant",
#     "content": "To determine on which days in 2024 Microsoft's stock was higher than $370, you'll need access to historical stock price data for that year. One way to get this data is through financial websites or APIs. However, I can guide you on using Python code with a library like pandas or financial data APIs like Yahoo Finance to get this data.\n\nHere’s a plan:\n\n1. **Use Python with a Financial Data Library:**\n   - We'll use the yfinance library to fetch historical stock data for Microsoft.\n   - Filter the data for entries where the stock price was higher than $370.\n\nLet's execute the code:\n\n```python\n# filename: msft_stock_above_370.py\n\nimport yfinance as yf\nimport pandas as pd\n\n# Get Microsoft historical stock data for 2024\nmsft = yf.Ticker(\"MSFT\")\nhist = msft.history(start=\"2024-01-01\", end=\"2024-12-31\")\n\n# Filter the data for days when closing price was higher than $370\nover_370 = hist[hist['Close'] > 370]\n\n# Print the dates and closing prices\nprint(over_370[['Close']])\n```\n\n1. Run this code to get a list of dates when Microsoft's stock closed above $370 in 2024.\n2. Check the output to confirm the specific days.\n\nExecute the code and provide the result, then we can analyze the output to give a concise answer."
#   },
#   {
#     "role": "user",
#     "name": "inner-code-interpreter",
#     "content": "exitcode: 0 (execution succeeded)\nCode output: \n                                Close\nDate                                 \n2024-01-08 00:00:00-05:00  371.908905\n2024-01-09 00:00:00-05:00  373.000763\n2024-01-10 00:00:00-05:00  379.928925\n2024-01-11 00:00:00-05:00  381.775116\n2024-01-12 00:00:00-05:00  385.586609\n...                               ...\n2024-12-23 00:00:00-05:00  435.250000\n2024-12-24 00:00:00-05:00  439.329987\n2024-12-26 00:00:00-05:00  438.109985\n2024-12-27 00:00:00-05:00  430.529999\n2024-12-30 00:00:00-05:00  424.829987\n\n[247 rows x 1 columns]"
#   },
#   {
#     "role": "user",
#     "name": "inner-assistant",
#     "content": "Based on the executed code, Microsoft's stock price was higher than $370 on 247 days in 2024. Here are the dates for a few instances when this occurred:\n\n- January 8, 2024 (Close: $371.91)\n- January 9, 2024 (Close: $373.00)\n- January 10, 2024 (Close: $379.93)\n- January 11, 2024 (Close: $381.78)\n- January 12, 2024 (Close: $385.59)\n\n...and so on until December 30, 2024 (Close: $424.83).\n\nThis data gives us a clear picture of the days when Microsoft's stock closed above $370. The data spans from January 8 to December 30, 2024, with varying closing prices all above the specified value of $370.\n\nTERMINATE"
#   },
#   {
#     "role": "system",
#     "content": "Output a standalone response to the original request, without mentioning any of the intermediate discussion."
#   }
# ]
        

        response = self.client.create(context=None, messages=_messages, cache=self.client_cache, agent=self.name)#这里有cache
# {
#   "id": "chatcmpl-AxH63gwyrSrmbzjopVJ3VVHMUGG4r",
#   "model": "gpt-4o-2024-08-06",
#   "created": 1738690435,
#   "object": "chat.completion",
#   "service_tier": "default",
#   "system_fingerprint": "fp_50cad350e4",
#   "choices": [
#     {
#       "finish_reason": "stop",
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "content": "In 2024, Microsoft's stock price was higher than $370 on 247 days. Some examples include:\n\n- January 8, 2024\n- January 9, 2024\n- January 10, 2024\n- January 11, 2024\n- January 12, 2024\n\nThe pattern continued until December 30, 2024. The stock price exceeded $370 on numerous occasions throughout the year."
#       }
#     }
#   ],
#   "usage": {
#     "prompt_tokens": 851,
#     "completion_tokens": 93,
#     "total_tokens": 944
#   },
#   "cost": 0.0030575
# }
        extracted_response = self.client.extract_text_or_completion_object(response)[0]
        if not isinstance(extracted_response, str):
            return str(extracted_response.model_dump(mode="dict"))
        else:
            return extracted_response # 一个字符串

    @property
    def chat_manager(self) -> Union[GroupChatManager, None]:
        """Return the group chat manager."""
        return self._chat_manager

    def update_chat_manager(self, chat_manager: Union[GroupChatManager, None]):
        """Update the chat manager.

        Args:
            chat_manager (GroupChatManager): the group chat manager
        """
        self._chat_manager = chat_manager # 

        # Awkward, but due to object cloning, there's no better way to do this
        # Read the GroupChat object from the callback
        """ 
        self._reply_func_list.insert(
            position,
            {
                "trigger": trigger,
                "reply_func": reply_func,
                "config": copy.copy(config),
                "init_config": config,
                "reset_config": reset_config,
                "ignore_async_in_sync_chat": ignore_async_in_sync_chat and inspect.iscoroutinefunction(reply_func),
            },
        )
        """
        self._group_chat = None
        if self._chat_manager is not None:
            for item in self._chat_manager._reply_func_list: #直接copy过来的
                if isinstance(item["config"], GroupChat):
                    #pdb.set_trace()
                    self._group_chat = item["config"]
                    break
    # righst trigger的标准函数             
    def generate_inner_monologue_reply(
        self,
        messages: Optional[list[dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[OpenAIWrapper] = None,
    ) -> tuple[bool, Union[str, dict, None]]:
        # pdb.set_trace()
        """Generate a reply by running the group chat"""
        if self.chat_manager is None:
            return False, None #如果当前代理没有绑定任何群聊管理器（chat_manager），就无法启动内心讨论，直接返回 (False, None) 表示无法生成回复。

        if messages is None: # 这个就是给这个societyAgent发出问询的Agent的指令
            messages = self._oai_messages[sender]

        # We want to clear the inner monolgue, keeping only the exteranl chat for context.
        # Reset all the counters and histories, then populate agents with necessary context from the external chat
        #self.chat_manager.reset() # 重置所有的计数器和历史记录，然后从外部聊天中填充代理所需的上下文
        #self.update_chat_manager(self.chat_manager) #填充新的信息

        external_history = []
        if len(messages) > 1:
            external_history = messages[0 : len(messages) - 1]  # All but the current message
        #如果消息列表中有多条消息，就将除最后一条（最新消息）之外的消息作为“外部对话历史”。
        for agent in self._group_chat.agents:
            agent.reset() #清除该代理在内聊中的状态
            for message in external_history: #这里会遍历所有外部历史消息。外部历史消息是指当前对话中除了最新一条消息之外的所有消息，这些消息作为上下文，会传递给内聊中的各个代理。
                # Assign each message a name
                attributed_message = message.copy() #这里对每条消息做了深拷贝（复制一份），以防止对原消息数据产生修改，从而影响外部对话历史。
                if "name" not in attributed_message: #在内聊中，各个代理需要知道每条消息的来源，这样才能正确地模拟对话角色和区分不同说话者。如果消息中没有“name”字段，就需要手动添加。
                    if attributed_message["role"] == "assistant":
                        attributed_message["name"] = self.name
                    else:
                        attributed_message["name"] = sender.name #如果角色不是 "assistant"（比如是用户）：用发送者的名称（sender.name）。
                self.chat_manager.send(attributed_message, agent, request_reply=False, silent=True)#把历史信息apend上去了
                #send 方法：
                    # 将处理后的消息发送到当前内聊代理（agent）。

                    # 参数说明：

                    # request_reply=False 表示发送这条消息时不要求该代理立即回复。
                    # silent=True 表示这条消息发送时不会在外部显示，只是在内部群聊中传递上下文信息。


        try:
            # pdb.set_trace()
            self.initiate_chat(self.chat_manager, message=messages[-1], clear_history=False) #这里传入最新的消息
        except:
            #pdb.set_trace()
            traceback.print_exc() #如果启动过程中出现异常，则捕获并打印异常信息，避免整个回复生成流程崩溃。

        response_preparer = self.response_preparer #这里获取之前设置好的 response_preparer 方法或 lambda 函数，该函数负责对整个内聊消息记录进行整理和提取最终答案。

        return True, response_preparer(self, self._group_chat.messages) #返回 True 和 response_preparer 处理后的结果
