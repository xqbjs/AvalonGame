# games/web/web_user_input.py
# -*- coding: utf-8 -*-
"""Unified web user input handler."""
import json
from typing import Any, Type
from pydantic import BaseModel

from agentscope.agent._user_input import UserInputBase, UserInputData
from agentscope.message import TextBlock

from games.web.game_state_manager import GameStateManager


class WebUserInput(UserInputBase):   
    
    def __init__(self, state_manager: GameStateManager, player_id: int):
        self.state_manager = state_manager
        self.player_id = player_id
    
    async def __call__(
        self,
        agent_id: str,
        agent_name: str,
        *args: Any,
        structured_model: Type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> UserInputData:
        # 强制使用字符串格式的 player_id 作为 Key
        queue_key = str(self.player_id)
        
        prompt = f"[{agent_name}] Please provide your input:"
        if structured_model is not None:
            prompt += f"\nStructured input required: {structured_model.model_json_schema()}"
        
        request_msg = self.state_manager.format_user_input_request(queue_key, prompt)
        
        # ================= [修改点 1：请求输入改为私聊] =================
        # 原代码：await self.state_manager.broadcast_message(request_msg)
        # 修正后：只发给当前需要输入的玩家。
        # 解决问题：多个人类玩家并发投票时，不再互相干扰（窗口闪退问题）。
        
        if hasattr(self.state_manager, "send_personal_message"):
            await self.state_manager.send_personal_message(self.player_id, request_msg)
        else:
            # 兜底：如果方法不存在（虽然Server已注入），则回退到广播
            await self.state_manager.broadcast_message(request_msg)
        # ==========================================================
        
        try:
            # 等待前端发送 WebSocket 消息填入队列
            content = await self.state_manager.get_user_input(queue_key, timeout=None)
            
            # ================= [修改点 2：回显改为私聊] =================
            # 逻辑：只给发送者自己看“我刚才说了啥”，防止其他人看到双重消息。
            # 仅在非结构化输入（聊天）时回显；结构化输入（投票/刺杀）通常是秘密的，不回显。
            
            if structured_model is None:
                echo_msg = self.state_manager.format_message(
                    sender=agent_name,
                    content=content,
                    role="user"
                )
                if hasattr(self.state_manager, "send_personal_message"):
                    await self.state_manager.send_personal_message(self.player_id, echo_msg)
            # ==========================================================
            
            structured_input = None
            if structured_model is not None:
                try:
                    structured_input = json.loads(content)
                except json.JSONDecodeError:
                    structured_input = {"content": content}
            
            return UserInputData(
                blocks_input=[TextBlock(type="text", text=content)],
                structured_input=structured_input,
            )
        except Exception:
            # 简单的错误处理
            return UserInputData(
                blocks_input=[TextBlock(type="text", text="")],
                structured_input=None,
            )