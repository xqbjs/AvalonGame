# -*- coding: utf-8 -*-
"""Unified web game launcher for Avalon + Diplomacy."""
import argparse
import asyncio
import os
import sys
import json  # [NEW]
import glob  # [NEW]
import threading
from pathlib import Path
from typing import Dict, Any, List
import copy
from agentscope.message import Msg

# Add repo root for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import agentscope
from agentscope.model import OpenAIChatModel
from agentscope.memory import InMemoryMemory
from agentscope.formatter import DashScopeMultiAgentFormatter
from agentscope.tool import Toolkit

from games.web.game_state_manager import GameStateManager
from games.web.web_agent import WebUserAgent, ObserveAgent

# Avalon imports
from games.agents.thinking_react_agent import ThinkingReActAgent
from games.utils import load_agent_class
from games.games.avalon.game import avalon_game
from games.games.avalon.engine import AvalonBasicConfig
from games.games.avalon.workflows.eval_workflow import RoleManager
from games.games.avalon.engine import AvalonGameEnvironment


from games.games.diplomacy.engine import DiplomacyConfig
from games.games.diplomacy.game import diplomacy_game
from games.utils import (
    load_config,
    create_agent_from_config,
    create_model_from_config,
    deep_merge,
)


def _get_role_config(
    config_dict: Dict[str, Any],
    role_identifiers: list[str],
    frontend_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Get complete role configuration with priority-based lookup.
    Includes robust fallback to Environment Variables and URL sanitization.
    """
    default_role = config_dict.get('default_role', {})
    roles_config = config_dict.get('roles', {})
    
    if not isinstance(default_role, dict):
        default_role = {}
    if not isinstance(roles_config, dict):
        roles_config = {}
    
    # Start with default_role
    role_config = copy.deepcopy(default_role)
    
    # Find role-specific config by priority
    specific_role_config = None
    for identifier in role_identifiers:
        if identifier in roles_config:
            specific_role_config = roles_config[identifier]
            break
        identifier_upper = identifier.upper()
        for k, v in roles_config.items():
            if k.upper() == identifier_upper:
                specific_role_config = v
                break
        if specific_role_config:
            break
    
    if specific_role_config and isinstance(specific_role_config, dict):
        role_config = deep_merge(role_config, specific_role_config)
    
    # Apply frontend config override
    if frontend_cfg and isinstance(frontend_cfg, dict):
        frontend_role_config = {}
        # Handle Model Config
        if frontend_cfg.get("base_model"):
            frontend_role_config["model"] = {
                "model_name": frontend_cfg.get("base_model"),
                "url": frontend_cfg.get("api_base", "").strip(),
                "api_key": frontend_cfg.get("api_key", "").strip() or os.getenv("OPENAI_API_KEY", ""),
                "model_type": "openai_chat", 
            }
        
        if frontend_cfg.get("agent_class"):
            frontend_role_config["agent"] = {
                "type": frontend_cfg.get("agent_class"),
                "kwargs": {}
            }
        
        if frontend_role_config:
            role_config = deep_merge(role_config, frontend_role_config)
            
    # [SAFETY NET] Ensure model config dict exists
    if "model" not in role_config:
        role_config["model"] = {}
    
    # ========================== https://www.uho.com.tw/article-61362.html ==========================
    # 1. 获取当前 URL (可能是 YAML 里的，也可能是前端传的)
    current_url = role_config["model"].get("url")
    
    # 2. 如果当前 URL 无效（空或 None），尝试读取环境变量
    if not current_url:
        current_url = os.getenv("OPENAI_BASE_URL", "")
        # 如果读到了，赋值回去
        if current_url:
            role_config["model"]["url"] = current_url

    # 3. [CRITICAL FIX] 再次检查最终的 URL，强制补充协议头
    # 这一步是为了防止：
    #   a. YAML 里写了 "dashscope.aliyuncs.com" (缺 https)
    #   b. 环境变量读出来即使是对的，也再确认一遍
    final_url = role_config["model"].get("url")
    
    if final_url and isinstance(final_url, str):
        final_url = final_url.strip() # 去除首尾空格
        if final_url: # 确保不是空字符串
            # 如果没有 http 或 https 开头，强制加上 https://
            if not (final_url.startswith("http://") or final_url.startswith("https://")):
                print(f"⚠️ [Config Fix] URL '{final_url}' missing protocol. Auto-fixing to 'https://{final_url}'")
                final_url = "https://" + final_url
            
            # 写回配置
            role_config["model"]["url"] = final_url
    # ======================================================================

    # Ensure API Key exists
    if not role_config["model"].get("api_key"):
        role_config["model"]["api_key"] = os.getenv("OPENAI_API_KEY", "")

    return role_config

async def run_avalon(
    state_manager: GameStateManager,
    num_players: int,
    language: str,
    user_agent_id: int,
    mode: str,
    preset_roles: list[tuple[int, str, bool]] | None = None,
    selected_portrait_ids: list[int] | None = None,
    agent_configs: Dict[int, Dict[str, str]] | None = None,
    ai_ids: List[int] | None = None,
):
    """Run Avalon game"""
    config = AvalonBasicConfig.from_num_players(num_players)
    
    yaml_path = os.environ.get("AVALON_CONFIG_YAML", "games/games/avalon/configs/default_config.yaml")
    task_cfg = load_config(yaml_path)
    
    if not selected_portrait_ids:
        selected_portrait_ids = list(range(1, num_players + 1))

    agents = []
    observe_agent = None
    if mode == "observe":
        observe_agent = ObserveAgent(name="Observer", state_manager=state_manager)
    
    ai_portrait_index = 0

    if preset_roles is None:
        print("No preset roles, using AvalonGameEnvironment to generate default roles")
        env = AvalonGameEnvironment(config)
        assigned_roles = env.get_roles()
    else:
        assigned_roles = preset_roles
    
    role_manager = RoleManager(assigned_roles)

    for i in range(num_players):
        # Determine Human vs AI
        is_human = False
        if ai_ids is not None:
            if i not in ai_ids:
                is_human = True
        else:
            if mode == "participate" and i == user_agent_id:
                is_human = True

        if is_human:
            agent = WebUserAgent(name=f"Player{i}", state_manager=state_manager, player_id=i)
        else:
            # AI Logic
            if mode == "participate":
                if ai_portrait_index < len(selected_portrait_ids):
                    portrait_id = selected_portrait_ids[ai_portrait_index]
                    ai_portrait_index += 1
                else:
                    portrait_id = i + 1
            else:
                portrait_id = selected_portrait_ids[i] if i < len(selected_portrait_ids) else (i + 1)
            
            # Get Config
            frontend_cfg = None
            if agent_configs:
                if portrait_id in agent_configs:
                    frontend_cfg = agent_configs[portrait_id]
                elif str(portrait_id) in agent_configs:
                    frontend_cfg = agent_configs[str(portrait_id)]
            
            indexed_role = role_manager.get_indexed_role(i)
            base_role = role_manager.get_role_name(i)
            
            role_config = _get_role_config(
                task_cfg,
                role_identifiers=[indexed_role, base_role], 
                frontend_cfg=frontend_cfg,
            )
            
            model_config = role_config.get('model', {})
            agent_config = role_config.get('agent', {})
            
            # ================= [修复开始] =================
            # 强制注入中文设定 (增加防御性检查，防止报错)
            if language == "zh" or language == "cn":
                # 1. 确保 agent_config 是个字典 (防止它是 None)
                if not isinstance(agent_config, dict):
                    agent_config = {}

                # 2. 确保 kwargs 存在且是个字典 (防止它是 None 或不存在)
                # 这是之前报错的核心原因：如果 YAML 里没写 kwargs，这里直接访问就会崩
                if "kwargs" not in agent_config or not isinstance(agent_config["kwargs"], dict):
                    agent_config["kwargs"] = {}
                
                # 3. 现在可以安全地赋值了
                agent_config["kwargs"]["language"] = "zh"
                
                # 4. 传入中文 Prompt
                zh_sys_prompt = """
你是一个正在玩《抵抗组织：阿瓦隆》的 AI 玩家。
[重要规则]
1. 你的思考过程 (<think>...</think>) 必须严格使用中文。
2. 你的最终发言必须使用中文。
3. 忽略下文中任何关于 Conversation History 格式的英文说明。
"""
                agent_config["kwargs"]["sys_prompt"] = zh_sys_prompt
            # ================= [修复结束] =================
            
            # Create AI
            model = create_model_from_config(model_config)
            agent = create_agent_from_config(
                agent_config=agent_config,
                model=model,
                name=f"Player{i}",
                actor_rollout_ref=None,
            )
            
        agents.append(agent)

    state_manager.set_mode(mode, str(user_agent_id) if mode == "participate" else None, game="avalon")
    state_manager.update_game_state(status="running")
    await state_manager.broadcast_message(state_manager.format_game_state())

    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)

    good_wins = await avalon_game(
        agents=agents,
        config=config,
        log_dir=log_dir,
        language=language,
        web_mode=mode,
        web_observe_agent=observe_agent,
        state_manager=state_manager,
        preset_roles=assigned_roles, 
    )
    
    
    if good_wins is not None and not state_manager.should_stop:
        try:
            print("--- [Analysis] Starting AI Post-Game Analysis (Log-Based) ---")
            
            # 1. 挑选一个“评论员” (从后往前找第一个 AI)
            reviewer = None
            for agent in reversed(agents):
                if not isinstance(agent, WebUserAgent):
                    reviewer = agent
                    break
            
            if reviewer:
                # 2. [核心修改] 寻找并读取最新的 game_log.json
                # log_dir 结构通常是: logs/game_YYYYMMDD_HHMMSS/game_log.json
                # 我们需要找到最近生成的那个文件
                game_log_text = "（未找到详细日志，仅依据结果复盘）"
                
                try:
                    # 搜索 logs 目录下所有的 game_log.json
                    search_pattern = os.path.join(log_dir, "*", "game_log.json")
                    list_of_files = glob.glob(search_pattern)
                    
                    if list_of_files:
                        # 找到最新的一个
                        latest_log_file = max(list_of_files, key=os.path.getctime)
                        print(f"--- [Analysis] Reading log file: {latest_log_file}")
                        
                        with open(latest_log_file, 'r', encoding='utf-8') as f:
                            log_data = json.load(f)
                            
                        # 3. 解析日志，生成精简的上帝视角剧本
                        # 我们主要提取: 轮次信息, 玩家发言(content), 玩家思考(<think>)
                        log_entries = []
                        for entry in log_data:
                            # 过滤掉一些无关紧要的系统日志，只保留关键交互
                            # 通常日志里有 'agent_name', 'content', 'mode' 等字段
                            agent_name = entry.get("agent_name", "System")
                            content = entry.get("content", "")
                            
                            # 如果内容太长（比如 SVG 地图数据），截断它
                            if len(str(content)) > 2000: 
                                content = "[Data too long, omitted]"
                                
                            # 格式化: [Player0 (Merlin)] 说/想: ...
                            # 尝试从 assigned_roles 里找身份信息辅助 (可选)
                            role_info = ""
                            # 这里简单的用名字做前缀
                            log_entries.append(f"【{agent_name}】: {content}")
                        
                        # 将列表合并成文本
                        game_log_text = "\n".join(log_entries)
                        
                        # 截断保护：如果日志太长超过 LLM 上下文，取最后 15000 字符 (通常包含最关键的后期辩论)
                        # 或者取首尾。这里简单处理：保留所有，依靠 LLM 的长窗口
                        if len(game_log_text) > 50000:
                            game_log_text = game_log_text[:20000] + "\n...[中间省略]...\n" + game_log_text[-20000:]
                            
                except Exception as log_err:
                    print(f"Error reading game log: {log_err}")

                # 4. 准备复盘 Prompt
                winner_text = "好人方 (Merlin's Team)" if good_wins else "坏人方 (Assassin's Team)"
                roles_reveal_str = "\n".join([f"Player {i}: {r[1]}" for i, r in enumerate(assigned_roles)])
                
                analysis_prompt = (
                    f"游戏结束了！最终获胜的是：【{winner_text}】。\n"
                    f"这是所有玩家的真实身份揭秘：\n{roles_reveal_str}\n\n"
                    f"以下是本局游戏的【完整全知视角记录】，包含了所有玩家的公开用语以及他们的【内心思考(<think>标签)】。\n"
                    f"================ 游戏记录开始 ================\n"
                    f"{game_log_text}\n"
                    f"================ 游戏记录结束 ================\n\n"
                    f"请你作为本局游戏的“首席评论员”，结合所有人的内心真实想法(Think)和实际行动(Speak)，"
                    f"进行一次深度的复盘点评。\n"
                    f"要求：\n"
                    f"1. 语言风格风趣冷静，一针见血。\n"
                    f"2. 重点分析：谁是 MVP？谁是“猪队友”？\n"
                    f"3. 结合 <think> 内容，通过对比“心里想的”和“嘴上说的”，揭穿某些玩家的伪装或失误。\n"
                    f"4. 先给结合<think>的内容,从player0到最后的playern(n根据实际的游玩人数来看),做出对每个玩家的分析,然后进行总结"
                    f"5. 直接输出复盘内容，不要使用 Markdown 标题，不要带 <think> 标签。"
                )
                
                # 通知前端
                await state_manager.broadcast_message(state_manager.format_message(
                    sender="System",
                    content="正在读取完整游戏日志进行深度复盘，请稍候...",
                    role="system"
                ))

                # 5. 调用 AI 生成
                trigger_msg = Msg(name="System", content=analysis_prompt, role="system")
                response_msg = await reviewer.reply(trigger_msg)
                
                # 6. 清洗内容格式
                import re 
                raw_content = response_msg.content
                analysis_text = ""
                if isinstance(raw_content, list):
                    for item in raw_content:
                        if isinstance(item, dict): analysis_text += item.get("text", "")
                        elif isinstance(item, str): analysis_text += item
                elif isinstance(raw_content, str):
                    analysis_text = raw_content
                else:
                    analysis_text = str(raw_content)

                # 去掉 AI 自己的思考过程，只保留点评结果
                analysis_text = re.sub(r'<think>.*?</think>', '', analysis_text, flags=re.DOTALL).strip()
                
                if not analysis_text.strip():
                     analysis_text = "（AI 阅读了日志，但未输出有效评价）"
                
                final_content = f"【上帝视角复盘 - {reviewer.name}】\n\n{analysis_text}"
                
                await state_manager.broadcast_message(state_manager.format_message(
                    sender=reviewer.name,
                    content=final_content,
                    role="assistant"
                ))
                
        except Exception as e:
            print(f"❌ [Analysis Error] AI 复盘失败: {e}")
            import traceback
            traceback.print_exc()
    # ================= [END 新增功能] =================

    if good_wins is None or state_manager.should_stop:
        state_manager.update_game_state(status="stopped")
        await state_manager.broadcast_message(state_manager.format_game_state())
        return

    state_manager.update_game_state(status="finished", good_wins=good_wins)
    await state_manager.broadcast_message(state_manager.format_game_state())
    result_msg = state_manager.format_message(
        sender="System",
        content=f"Game finished! {'Good wins!' if good_wins else 'Evil wins!'}",
        role="assistant",
    )
    await state_manager.broadcast_message(result_msg)


async def run_diplomacy(
    state_manager: GameStateManager,
    config: DiplomacyConfig,
    mode: str,
    selected_portrait_ids: list[int] | None = None,
    agent_configs: Dict[int, Dict[str, str]] | None = None,
    ai_ids: List[int] | None = None, 
):
    """Run Diplomacy game."""
    agentscope.init()

    yaml_path = os.environ.get("DIPLOMACY_CONFIG_YAML", "games/games/diplomacy/configs/default_config.yaml")
    task_cfg = load_config(yaml_path)

    agents = []
    observe_agent = None
    if mode == "observe":
        observe_agent = ObserveAgent(name="Observer", state_manager=state_manager)

    for power_idx, power in enumerate(config.power_names):
        if mode == "participate" and config.human_power and power == config.human_power:
            agent = WebUserAgent(name=power, state_manager=state_manager, player_id=power_idx) 
            state_manager.user_agent_id = agent.id
        else:
            portrait_id = None
            frontend_cfg = None
            if selected_portrait_ids and power_idx < len(selected_portrait_ids):
                portrait_id = selected_portrait_ids[power_idx]
                if portrait_id is not None and portrait_id != -1:
                    if agent_configs:
                        if portrait_id in agent_configs:
                            frontend_cfg = agent_configs[portrait_id]
                        elif str(portrait_id) in agent_configs:
                            frontend_cfg = agent_configs[str(portrait_id)]

            role_config = _get_role_config(
                task_cfg,
                role_identifiers=[power],
                frontend_cfg=frontend_cfg,
            )
            
            model_config = role_config.get('model', {})
            agent_config = role_config.get('agent', {})
            
            model = create_model_from_config(model_config)
            agent = create_agent_from_config(
                agent_config=agent_config,
                model=model,
                name=power,
                actor_rollout_ref=None,
            )
            agent.power_name = power
            agent.set_console_output_enabled(True)
        agents.append(agent)
        
    state_manager.set_mode(mode, config.human_power if mode == "participate" else None, game="diplomacy")
    state_manager.update_game_state(status="running", human_power=config.human_power if mode == "participate" else None)
    await state_manager.broadcast_message(state_manager.format_game_state())

    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)

    result = await diplomacy_game(
        agents=agents,
        config=config,
        state_manager=state_manager,
        log_dir=log_dir,
        observe_agent=observe_agent,
    )

    if result is None or state_manager.should_stop:
        state_manager.update_game_state(status="stopped")
        await state_manager.broadcast_message(state_manager.format_game_state())
        return

    state_manager.update_game_state(status="finished", result=result)
    await state_manager.broadcast_message(state_manager.format_game_state())
    end_msg = state_manager.format_message(sender="System", content=f"Diplomacy finished: {result}", role="assistant")
    await state_manager.broadcast_message(end_msg)


def start_game_thread(
    state_manager: GameStateManager,
    game: str,
    mode: str,
    language: str = "en",
    num_players: int = 5,
    user_agent_id: int = 0,
    preset_roles: list[dict] | None = None,
    selected_portrait_ids: list[int] | None = None,
    agent_configs: Dict[int, Dict[str, str]] | None = None,
    ai_ids: List[int] | None = None, 
    human_power: str | None = None,
    max_phases: int = 20,
    negotiation_rounds: int = 3,
    power_names: list[str] | None = None,
    power_models: Dict[str, str] | None = None,
):
    def run():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            if game == "avalon":
                preset_roles_tuples: list[tuple[int, str, bool]] | None = None
                if preset_roles:
                    try:
                        from games.games.avalon.engine import AvalonBasicConfig
                        preset_roles_tuples = [
                            (AvalonBasicConfig.ROLES_REVERSE.get(str(x.get("role_name")), 0), str(x.get("role_name")), bool(x.get("is_good")))
                            for x in preset_roles
                            if isinstance(x, dict)
                        ]
                    except Exception:
                        pass
                
                portrait_ids = selected_portrait_ids if selected_portrait_ids else list(range(1, num_players + 1))
                task = loop.create_task(run_avalon(
                    state_manager=state_manager,
                    num_players=num_players,
                    language=language,
                    user_agent_id=user_agent_id,
                    mode=mode,
                    preset_roles=preset_roles_tuples,
                    selected_portrait_ids=portrait_ids,
                    agent_configs=agent_configs,
                    ai_ids=ai_ids,
                ))
                state_manager._game_task = task
                
                try:
                    loop.run_until_complete(task)
                except asyncio.CancelledError:
                    pass
                finally:
                    pending = asyncio.all_tasks(loop)
                    for t in pending:
                        t.cancel()
                    if pending:
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            else:
                cfg = DiplomacyConfig.default()
                cfg.max_phases = max_phases
                cfg.negotiation_rounds = negotiation_rounds
                cfg.language = language
                cfg.human_power = human_power
                if power_names:
                    cfg.power_names = list(power_names)
                
                task = loop.create_task(run_diplomacy(
                    state_manager=state_manager,
                    config=cfg,
                    mode=mode,
                    selected_portrait_ids=selected_portrait_ids,
                    agent_configs=agent_configs,
                    ai_ids=ai_ids,
                ))
                state_manager._game_task = task
                
                try:
                    loop.run_until_complete(task)
                except asyncio.CancelledError:
                    pass
                finally:
                    pending = asyncio.all_tasks(loop)
                    for t in pending:
                        t.cancel()
                    if pending:
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            try:
                loop.close()
            except Exception:
                pass
    
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    state_manager.set_game_thread(thread)
    return thread


def main():
    parser = argparse.ArgumentParser(description="Run web game (avalon|diplomacy)")
    parser.add_argument("--game", type=str, default="avalon", choices=["avalon", "diplomacy"])
    parser.add_argument("--mode", type=str, default="observe", choices=["observe", "participate"])
    parser.add_argument("--user-agent-id", type=int, default=0)
    parser.add_argument("--num-players", type=int, default=5)
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--human-power", type=str, default=None)
    parser.add_argument("--max-phases", type=int, default=20)
    parser.add_argument("--negotiation-rounds", type=int, default=3)
    args = parser.parse_args()

    state_manager = GameStateManager()
    start_game_thread(
        state_manager=state_manager,
        game=args.game,
        mode=args.mode,
        language=args.language,
        num_players=args.num_players,
        user_agent_id=args.user_agent_id,
        human_power=args.human_power,
        max_phases=args.max_phases,
        negotiation_rounds=args.negotiation_rounds,
        power_models={},
        ai_ids=[], 
    )
    while True:
        asyncio.sleep(1)


if __name__ == "__main__":
    main()