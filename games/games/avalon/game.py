# -*- coding: utf-8 -*-
"""Avalon game implemented by agentscope."""
from typing import Any

from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.pipeline import MsgHub, fanout_pipeline

from loguru import logger

from games.games.avalon.engine import AvalonGameEnvironment, AvalonBasicConfig
from games.games.avalon.utils import Parser, GameLogger, LanguageFormatter
from games.agents.echo_agent import EchoAgent


class AvalonGame:
    """Main Avalon game class that integrates all game functionality."""
    
    def __init__(
        self,
        agents: list[AgentBase],
        config: AvalonBasicConfig,
        log_dir: str | None = None,
        language: str = "en",
        observe_agent: AgentBase | None = None,
        state_manager: Any = None,
        preset_roles: list[tuple[int, str, bool]] | None = None,
        timestamp: str | None = None,
    ):
        """Initialize Avalon game."""
        
        self.agents = agents
        self.config = config
        self.log_dir = log_dir
        self.language = language
        self.observe_agent = observe_agent
        self.state_manager = state_manager
        
        # Initialize utilities
        self.localizer = LanguageFormatter(language)
        self.parser = Parser()
        self.game_logger = GameLogger()
        
        # Initialize moderator
        self.moderator = EchoAgent()
        self.moderator.set_console_output_enabled(True)
        
        # Import prompts based on language
        if self.localizer.is_zh:
            from games.games.avalon.prompt import ChinesePrompts as Prompts
        else:
            from games.games.avalon.prompt import EnglishPrompts as Prompts
        self.Prompts = Prompts
        
        # Initialize game environment with preset roles if provided
        if preset_roles is not None:
            # Use preset roles - create environment with presets
            # Extract role names (they should already be in correct format from get_roles())
            role_names = [role_name for _, role_name, _ in preset_roles]
            import numpy as np
            quest_leader = np.random.randint(0, config.num_players - 1)
            presets = {
                'num_players': config.num_players,
                'quest_leader': quest_leader,
                'role_names': role_names,
            }
            self.env = AvalonGameEnvironment.from_presets(presets)
            # Fix: from_presets uses cls variables, so we need to set instance variables
            # Convert preset_roles to the format needed by env (role_ids array)
            role_ids = [role_id for role_id, _, _ in preset_roles]
            is_good_list = [is_good for _, _, is_good in preset_roles]
            #import numpy as np
            self.env.roles = np.array(role_ids)
            self.env.is_good = np.array(is_good_list)
            self.env.quest_leader = quest_leader
            # Use the preset roles directly
            self.roles = preset_roles
        else:
            # Use default random role assignment
            self.env = AvalonGameEnvironment(config)
            self.roles = self.env.get_roles()
        
        # Initialize game log
        self.game_logger.create_game_log_dir(log_dir, timestamp)
        self.game_logger.initialize_game_log(self.roles, config.num_players)
        
        assert len(agents) == config.num_players, f"The Avalon game needs exactly {config.num_players} players."
        
    # [NEW] 核心辅助函数：提取 AI 的完整思考过程 (Prompt + Thinking)
    def _extract_rich_log(self, agent: AgentBase, msg: Msg) -> dict:
        """
        从 Agent 的调用历史中提取 '发送给模型的 Prompt' 和 '模型的原始 Response'。
        这样就能看到 System Prompt 和 Thinking 过程了。
        """
        log_payload = msg.to_dict()
        
        # 检查 Agent 是否有模型调用记录 (UserAgent 或 EchoAgent 可能没有)
        if hasattr(agent, "model_call_history") and agent.model_call_history:
            # 获取最近一次调用
            last_call = agent.model_call_history[-1]
            
            # 1. 提取 Full Prompt (系统指令 + 历史对话 + 当前输入)
            # 这就是你想要的 "System对AI的提示词"
            if "prompt" in last_call:
                log_payload["llm_full_prompt"] = last_call["prompt"]
            
            # 2. 提取 Raw Response (AI 的原始回复，通常包含 <think> 或思维链)
            # 这就是你想要的 "AI Thinking"
            if "response" in last_call:
                log_payload["llm_raw_response"] = last_call["response"]
            
        return log_payload
    
    def _get_hub_participants(self) -> list[AgentBase]:
        """Get participants list for hub, including observe_agent if present."""
        participants = self.agents.copy()
        if self.observe_agent is not None:
            participants.append(self.observe_agent)
        return participants
    
    
    async def run(self) -> bool:
        """Run the Avalon game."""
        
        # [关键] 使用 try...finally 结构包裹整个游戏逻辑
        try:
            # --- 1. 游戏启动广播 ---
            async with MsgHub(participants=self._get_hub_participants()) as greeting_hub:
                system_prompt_content = self.localizer.format_system_prompt(self.config, self.Prompts)
                system_prompt_msg = await self.moderator(system_prompt_content)
                await greeting_hub.broadcast(system_prompt_msg)
                
                # [NEW Log] 广播：所有人记录“系统规则”
                # 这会让 PlayerX_stream.jsonl 的第一条记录变成规则介绍
                self.game_logger.log_broadcast_event(system_prompt_msg.to_dict(), self.config.num_players)
                
                # ================= [新增：详细身份介绍 (竖线分隔版)] =================
                from collections import Counter
                # 统计角色
                all_role_names = [self.localizer.format_role_name(r[1]) for r in self.roles]
                role_counts = Counter(all_role_names)
                
                if self.localizer.is_zh:
                    # 1. 标题和总人数
                    intro_text = [f"📢 **本局身份配置** ({self.config.num_players}人)"]
                    intro_text.append(f"✅ 好人: {self.config.num_good} | ❌ 坏人: {self.config.num_evil}")
                    
                    # 2. [核心修改] 将所有角色拼接成一行，用 " | " 分隔
                    # 例如: 梅林: 1名 | 派西维尔: 1名 | 刺客: 1名 ...
                    role_items = [f"{name}: {count}" for name, count in role_counts.items()]
                    intro_text.append("----------------")
                    intro_text.append(" | ".join(role_items))
                    intro_text.append("----------------")
                    
                    full_intro_str = "\n".join(intro_text)
                else:
                    intro_text = [f"📢 **Role Config** ({self.config.num_players} Players)"]
                    intro_text.append(f"✅ Good: {self.config.num_good} | ❌ Evil: {self.config.num_evil}")
                    
                    role_items = [f"{name}: {count}" for name, count in role_counts.items()]
                    intro_text.append("----------------")
                    intro_text.append(" | ".join(role_items))
                    intro_text.append("----------------")
                    
                    full_intro_str = "\n".join(intro_text)

                # 发送消息
                role_intro_msg = await self.moderator(full_intro_str)
                await greeting_hub.broadcast(role_intro_msg)
                self.game_logger.log_broadcast_event(role_intro_msg.to_dict(), self.config.num_players)
                # ================= [新增结束] =================
                
                new_game_msg = await self.moderator(
                    self.Prompts.to_all_new_game.format(self.localizer.format_agents_names(self.agents))
                )
                await greeting_hub.broadcast(new_game_msg)
                
                # [NEW Log] 广播：所有人记录“新游戏开始”
                self.game_logger.log_broadcast_event(new_game_msg.to_dict(), self.config.num_players)

            # --- 2. 分配角色 ---
            # 注意：这里面的私有日志已经在 _assign_roles_to_agents 里加过了
            await self._assign_roles_to_agents()
            
            # --- 3. 更新前端状态 (Web Mode) ---
            if self.state_manager:
                roles_data = [
                    {"role_id": int(rid), "role_name": str(rn), "is_good": bool(ig)}
                    for rid, rn, ig in self.roles
                ]
                self.state_manager.update_game_state(roles=roles_data)
                await self.state_manager.broadcast_message(self.state_manager.format_game_state())

            # --- 4. 主游戏循环 ---
            game_stopped = False
            while not self.env.done:
                # 检查前端停止信号
                if self.state_manager and self.state_manager.should_stop:
                    logger.info("Game stopped by user request")
                    game_stopped = True
                    self.env.done = True
                    break
                
                phase, _ = self.env.get_phase()
                leader = self.env.get_quest_leader()
                mission_id = self.env.turn
                round_id = self.env.round

                # 更新前端
                if self.state_manager:
                    self.state_manager.update_game_state(
                        phase=phase, mission_id=mission_id, round_id=round_id, leader=leader
                    )
                    await self.state_manager.broadcast_message(self.state_manager.format_game_state())

                async with MsgHub(participants=self._get_hub_participants(), enable_auto_broadcast=False, name="all_players") as all_players_hub:
                    # 再次检查停止信号
                    if self.state_manager and self.state_manager.should_stop:
                        game_stopped = True
                        self.env.done = True
                        break
                        
                    # 处理各个阶段 (这些函数内部我们已经加了 Log)
                    if phase == 0:
                        await self._handle_team_selection_phase(all_players_hub, mission_id, round_id, leader)
                    elif phase == 1:
                        await self._handle_team_voting_phase(all_players_hub)
                    elif phase == 2:
                        await self._handle_quest_voting_phase(all_players_hub, mission_id)
                    elif phase == 3:
                        await self._handle_assassination_phase(all_players_hub)

            # --- 5. 游戏正常结束处理 ---
            if not game_stopped:
                async with MsgHub(participants=self._get_hub_participants()) as end_hub:
                    end_message = self.localizer.format_game_end_message(
                        self.env.good_victory, self.roles, self.Prompts
                    )
                    end_msg = await self.moderator(end_message)
                    await end_hub.broadcast(end_msg)
                    
                    # [NEW Log] 广播：所有人记录“游戏大结局”
                    self.game_logger.log_broadcast_event(end_msg.to_dict(), self.config.num_players)

                logger.info(f"Game finished. Good wins: {self.env.good_victory}")
                return self.env.good_victory
            else:
                logger.info("Game was stopped by user")
                return None
        
        # [关键] 捕获所有异常，包括 Ctrl+C
        except KeyboardInterrupt:
            logger.info("--- User pressed Ctrl+C, stopping game... ---")
            raise 
        except Exception as e:
            logger.error(f"Game runtime error: {e}")
            raise e
            
        # [核心] 无论上面发生了什么，这里一定会执行！
        finally:
            logger.info("--- [FINAL SAVE] Saving Game Logs & Agent Memories... ---")
            try:
                # 尝试保存最终的汇总文件 (虽然流式日志已经有了，但这个作为备份)
                await self.game_logger.save_game_logs(self.agents, self.env, self.roles)
                logger.info("--- [FINAL SAVE] Success! ---")
            except Exception as log_err:
                logger.error(f"Failed to save logs in finally block: {log_err}")
    
    
    
    async def _assign_roles_to_agents(self) -> None:
        """Assign roles to agents with FULL Avalon Logic and I18N Support."""
        
        logger.info("--- Starting Advanced Role Assignment (I18N) ---")

        for i, (my_role_id, my_role_name, my_side) in enumerate(self.roles):
            agent = self.agents[i]
            
            # 1. 基础信息本地化 (复用旧代码逻辑)
            localized_role_name = self.localizer.format_role_name(my_role_name)
            side_name = self.localizer.format_side_name(my_side) # "Good"/"Evil" or "好人"/"坏人"
            localized_agent_name = self.localizer.format_player_name(agent.name)
            
            # 2. 计算视野 (Visibility Logic)
            known_info_list = []
            
            # 辅助函数：生成 "Player X 是 坏人" 的本地化句子
            def format_knowledge(target_idx, target_role_str):
                # target_role_str 应该是本地化后的词，比如 "Evil"(坏人) 或 "Merlin"(梅林)
                t_pname = self.localizer.format_player_id(target_idx)
                if self.localizer.is_zh:
                    return f"{t_pname} 是 {target_role_str}"
                else:
                    return f"{t_pname} is {target_role_str}"

            # 获取本地化的 "Evil" 单词 (用于梅林和坏人视野)
            # 0 代表坏人阵营
            evil_word = self.localizer.format_side_name(0) 

            # === 逻辑 A: 梅林 (Merlin) ===
            # 能看到所有坏人，除了 莫德雷德 (Mordred)
            if my_role_name == "Merlin":
                for target_i, (t_id, t_name, t_side) in enumerate(self.roles):
                    if target_i == i: continue
                    # 看到坏人(t_side=0)，但看不到 Mordred
                    if (not t_side) and (t_name != "Mordred"):
                        known_info_list.append(format_knowledge(target_i, evil_word))

            # === 逻辑 B: 派西维尔 (Percival) ===
            # 能看到 梅林 (Merlin) 和 莫甘娜 (Morgana)，但分不清谁是谁
            elif my_role_name == "Percival":
                # 获取本地化的 "Merlin" 和 "Morgana"
                merlin_str = self.localizer.format_role_name("Merlin")
                morgana_str = self.localizer.format_role_name("Morgana")
                # 构造 "梅林 或 莫甘娜"
                if self.localizer.is_zh:
                    or_str = f"{merlin_str} 或 {morgana_str}"
                else:
                    or_str = f"{merlin_str} or {morgana_str}"
                
                for target_i, (t_id, t_name, t_side) in enumerate(self.roles):
                    if target_i == i: continue
                    if t_name in ["Merlin", "Morgana"]:
                        known_info_list.append(format_knowledge(target_i, or_str))

            # === 逻辑 C: 坏人阵营 (Evil) ===
            # 坏人互认，但 奥伯伦 (Oberon) 除外
            elif not my_side: 
                # 如果我是奥伯伦，我看不到队友
                if my_role_name == "Oberon":
                    pass 
                else:
                    # 我是普通坏人/莫甘娜/莫德雷德/刺客
                    for target_i, (t_id, t_name, t_side) in enumerate(self.roles):
                        if target_i == i: continue
                        # 看到其他坏人，但看不到 Oberon
                        if (not t_side) and (t_name != "Oberon"):
                            # 坏人看坏人，通常只知道是“同伙”，这里依然用 evil_word ("Bad"/"Evil")
                            known_info_list.append(format_knowledge(target_i, evil_word))

            # === 逻辑 D: 忠臣 (Servant) ===
            else:
                pass

            # 3. 构造 Prompt (使用 self.Prompts 模板)
            if known_info_list:
                sides_str = ", ".join(known_info_list)
                # 使用模板注入 sides_info
                additional_info = self.Prompts.to_agent_role_with_visibility.format(sides_info=sides_str)
            else:
                additional_info = self.Prompts.to_agent_role_no_visibility
            
            # 4. 组装最终消息
            role_info = self.Prompts.to_agent_role_assignment.format(
                agent_name=localized_agent_name,
                role_name=localized_role_name,
                side_name=side_name,
                additional_info=additional_info,
            )
            
            # 5. 发送私信
            role_msg = Msg(
                name="Moderator",
                content=role_info,
                role="assistant",
            )
            
            # 简单的日志打印
            log_type = "AI" if (hasattr(agent, 'model') and agent.model) else "Human"
            logger.info(f"Assigning role to {log_type} Agent {i}: {my_role_name} (Lang: {self.language})")

            await agent.observe(role_msg)
            # [Log] 私有记录：只写入当前玩家 i 的日志文件
            # 这样 Player2 的日志里就有 "你是派西维尔"，而 Player3 的日志里看不到这条
            self.game_logger.log_agent_event(i, agent.name, role_msg.to_dict())
    
    
    async def _handle_team_selection_phase(
        self,
        all_players_hub: MsgHub,
        mission_id: int,
        round_id: int,
        leader: int,
    ) -> None:
        """Handle Team Selection Phase."""
        self.game_logger.add_mission(mission_id, round_id, leader)
        
        # 1. 广播阶段开始
        phase_msg = await self.moderator(self.Prompts.to_all_team_selection_discuss.format(
            mission_id=mission_id,
            round_id=round_id,
            leader_id=leader,
            team_size=self.env.get_team_size(),
        ))
        await all_players_hub.broadcast(phase_msg)
        
        # [Log] 广播
        self.game_logger.log_broadcast_event(phase_msg.to_dict(), self.config.num_players)

        # 2. 讨论环节
        leader_agent = self.agents[leader]
        all_players_hub.set_auto_broadcast(True)
        self.game_logger.add_discussion_messages([]) 
        discussion_msgs = []
        
        # === 队长发言 ===
        leader_msg = await leader_agent()
        
        # [修改] 提取富文本日志 (含 Thinking)
        rich_leader_msg = self._extract_rich_log(leader_agent, leader_msg)
        
        # 存入总流水 (存简单版即可，省空间，或者也存 rich 版)
        self.game_logger.add_single_dialogue(rich_leader_msg)
        # 存入个人日志 (必须存 Rich 版！)
        self.game_logger.log_agent_event(leader, leader_agent.name, rich_leader_msg)
        
        discussion_msgs.append(leader_msg)
        
        # === 其他人发言 ===
        for i in range(1, self.config.num_players):
            current_speaker_idx = (leader + i) % self.config.num_players
            agent = self.agents[current_speaker_idx]
            
            msg = await agent()
            
            # [修改] 提取富文本日志
            rich_msg = self._extract_rich_log(agent, msg)
            
            self.game_logger.add_single_dialogue(rich_msg)
            self.game_logger.log_agent_event(current_speaker_idx, agent.name, rich_msg)
            
            discussion_msgs.append(msg)
        
        all_players_hub.set_auto_broadcast(False)

        # 3. 队长提议
        propose_prompt = await self.moderator(self.Prompts.to_leader_propose_team.format(
            mission_id=mission_id,
            team_size=self.env.get_team_size(),
            max_player_id=self.config.num_players - 1,
        ))
        # [Log] 记录发给队长的指令
        self.game_logger.log_agent_event(leader, leader_agent.name, propose_prompt.to_dict())

        team_response = await leader_agent(propose_prompt)
        
        # [修改] 提取队长的思考 (为什么选这几个人?)
        rich_team_response = self._extract_rich_log(leader_agent, team_response)
        self.game_logger.log_agent_event(leader, leader_agent.name, rich_team_response)
        
        # 广播给其他人看的是结果，不是思考，所以这里广播用普通的就行
        # 但为了 Log 完整，我们这里也可以广播 Rich 的，或者只广播结果
        # 这里建议只广播结果，个人 Log 存 Rich
        self.game_logger.log_broadcast_event(team_response.to_dict(), self.config.num_players)

        team = self.parser.parse_team_from_response(team_response.content)
        
        # Normalize team size
        team = list(set(team))[:self.env.get_team_size()]
        if len(team) < self.env.get_team_size():
            remaining = [i for i in range(self.config.num_players) if i not in team]
            team.extend(remaining[:self.env.get_team_size() - len(team)])
        
        self.env.choose_quest_team(team=frozenset(team), leader=leader)
        self.game_logger.add_team_proposal(list(team))
    
    
    
    async def _handle_team_voting_phase(self, all_players_hub: MsgHub) -> None:
        """Handle Team Voting Phase."""
        current_team = self.env.get_current_quest_team()
        
        # 1. 发送投票指令
        vote_prompt = await self.moderator(self.Prompts.to_all_team_vote.format(team=list(current_team)))
        # [Log] 广播
        self.game_logger.log_broadcast_event(vote_prompt.to_dict(), self.config.num_players)

        # 2. 收集投票
        msgs_vote = await fanout_pipeline(self.agents, msg=[vote_prompt], enable_gather=True)
        
        # [修改] 记录每个人的思考过程
        for i, msg in enumerate(msgs_vote):
            agent = self.agents[i]
            # 提取 Thinking
            rich_vote_msg = self._extract_rich_log(agent, msg)
            self.game_logger.log_agent_event(i, agent.name, rich_vote_msg)

        votes = [self.parser.parse_vote_from_response(msg.content) for msg in msgs_vote]
        outcome = self.env.gather_team_votes(votes)
        
        # 3. 广播结果
        approved = bool(outcome[2])
        votes_detail, result_text, outcome_text = self.localizer.format_vote_details(votes, approved)
        
        result_msg = await self.moderator(self.Prompts.to_all_team_vote_result.format(
            result=result_text,
            team=list(current_team),
            outcome=outcome_text,
            votes_detail=votes_detail,
        ))
        await all_players_hub.broadcast([result_msg])
        
        # [Log] 广播结果
        self.game_logger.log_broadcast_event(result_msg.to_dict(), self.config.num_players)
        
        self.game_logger.add_team_voting(list(current_team), votes, approved)
    
    
    
    async def _handle_quest_voting_phase(self, all_players_hub: MsgHub, mission_id: int) -> None:
        """Handle Quest Voting Phase."""
        current_team = self.env.get_current_quest_team()
        team_agents = [self.agents[i] for i in current_team]
        
        # 1. 发送指令
        vote_prompt = await self.moderator(self.Prompts.to_all_quest_vote.format(team=list(current_team)))
        
        # [Log] 记录发给做任务人的私有指令
        for i in current_team:
            self.game_logger.log_agent_event(i, self.agents[i].name, vote_prompt.to_dict())

        # 2. 收集出票
        msgs_vote = await fanout_pipeline(team_agents, msg=[vote_prompt], enable_gather=True)
        
        # [修改] 记录做任务人的思考
        for idx, agent_idx in enumerate(current_team):
            agent = self.agents[agent_idx]
            msg = msgs_vote[idx]
            
            # 提取 Thinking
            rich_quest_msg = self._extract_rich_log(agent, msg)
            self.game_logger.log_agent_event(agent_idx, agent.name, rich_quest_msg)

        votes = [self.parser.parse_vote_from_response(msg.content) for msg in msgs_vote]
        outcome = self.env.gather_quest_votes(votes)
        
        # 3. 广播结果
        result_msg = await self.moderator(self.Prompts.to_all_quest_result.format(
            mission_id=mission_id,
            outcome="succeeded" if outcome[2] else "failed",
            team=list(current_team),
            num_fails=outcome[3],
        ))
        await all_players_hub.broadcast(result_msg)
        
        # [Log] 广播结果
        self.game_logger.log_broadcast_event(result_msg.to_dict(), self.config.num_players)
        
        self.game_logger.add_quest_voting(list(current_team), votes, int(outcome[3]), bool(outcome[2]))
    
    
    async def _handle_assassination_phase(
        self,
        all_players_hub: MsgHub,
    ) -> None:
        """Handle Assassination Phase."""
        # Broadcast phase
        assassination_msg = await self.moderator(self.Prompts.to_all_assassination)
        await all_players_hub.broadcast(assassination_msg)
        # [Log]
        self.game_logger.log_broadcast_event(assassination_msg.to_dict(), self.config.num_players)

        # Assassin chooses target
        assassin_id = self.env.get_assassin()
        assassin_agent = self.agents[assassin_id]
        
        assassinate_prompt = await self.moderator(
            self.Prompts.to_assassin_choose.format(max_player_id=self.config.num_players - 1)
        )
        # [Log] 记录发给刺客的指令
        self.game_logger.log_agent_event(assassin_id, assassin_agent.name, assassinate_prompt.to_dict())

        target_response = await assassin_agent(assassinate_prompt)
        
        # [修改] 记录刺客的思考
        rich_response = self._extract_rich_log(assassin_agent, target_response)
        self.game_logger.log_agent_event(assassin_id, assassin_agent.name, rich_response)
        
        target = self.parser.parse_player_id_from_response(target_response.content, self.config.num_players - 1)
        _, _, good_wins = self.env.choose_assassination_target(assassin_id, target)
        
        # Broadcast result
        assassin_name = self.localizer.format_player_id(assassin_id)
        target_name = self.localizer.format_player_id(target)
        result_text = self.Prompts.to_all_good_wins if good_wins else self.Prompts.to_all_evil_wins
        
        if self.localizer.is_zh:
            result_msg = await self.moderator(f"刺客{assassin_name} 选择刺杀{target_name}。{result_text}")
        else:
            result_msg = await self.moderator(f"Assassin {assassin_name} has chosen to assassinate {target_name}. {result_text}")
        await all_players_hub.broadcast(result_msg)
        
        # [Log]
        self.game_logger.log_broadcast_event(result_msg.to_dict(), self.config.num_players)
        
        self.game_logger.add_assassination(assassin_id, target, bool(good_wins))


# ============================================================================
# Convenience Function
# ============================================================================

async def avalon_game(
    agents: list[AgentBase],
    config: AvalonBasicConfig,
    log_dir: str | None = None,
    language: str = "en",
    web_mode: str | None = None,
    web_observe_agent: AgentBase | None = None,
    state_manager: Any = None,
    preset_roles: list[tuple[int, str, bool]] | None = None,  # added gpt
) -> bool:
    """Convenience function to run Avalon game.
    
    This is a wrapper around AvalonGame class for backward compatibility.
    
    Args:
        agents: List of agents (5-10 players). Can be ReActAgent, ThinkingReActAgent, or UserAgent.
        config: Game configuration.
        log_dir: Directory to save game logs. If None, logs are not saved.
        language: Language for prompts. "en" for English, "zh" or "cn" for Chinese.
        web_mode: Web mode ("observe" or "participate"). If None, runs in normal mode.
        web_observe_agent: Observer agent for web observe mode. Only used when web_mode="observe".
        state_manager: Optional state manager for web mode to check stop flag.
    Returns:
        True if good wins, False otherwise.
    """
    # Create AvalonGame instance
    game = AvalonGame(
        agents=agents,
        config=config,
        log_dir=log_dir,
        language=language,
        observe_agent=web_observe_agent if web_mode == "observe" else None,
        state_manager=state_manager,
        preset_roles=preset_roles,  # added gpt
    )
    
    # Run the game
    return await game.run()
