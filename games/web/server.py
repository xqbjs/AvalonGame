# -*- coding: utf-8 -*-
"""Unified web server for Avalon + Diplomacy."""
import asyncio
import json
import uuid
import sys
import os
from typing import Optional, Dict, List, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from pathlib import Path
import uvicorn

# [CRITICAL FIX] Windows asyncio policy fix
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from games.web.game_state_manager import GameStateManager
from games.web.run_web_game import start_game_thread
from games.utils import load_config

# --- GLOBAL STATE & LOOP CAPTURE ---
state_manager = GameStateManager()
MAIN_LOOP = None # Will store the main asyncio loop

# [NEW] Global store for current game metadata (portraits, names)
# Because WebSocket clients (guests) don't have this info in their sessionStorage
CURRENT_GAME_METADATA = {
    "num_players": 5,
    "players": [] 
}

app = FastAPI(title="Games Web Interface")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# [CRITICAL FIX] Capture Main Loop for Thread-Safe Scheduling
@app.on_event("startup")
async def startup_event():
    global MAIN_LOOP
    MAIN_LOOP = asyncio.get_running_loop()
    print(f"Main Event Loop Captured: {MAIN_LOOP}")


# --- MONKEY PATCHING FOR THREAD SAFETY ---

# 1. Patch Websocket Storage
if not hasattr(state_manager, "websockets"):
    state_manager.websockets = {}

_original_add = state_manager.add_websocket_connection
_original_remove = state_manager.remove_websocket_connection

def _add_wrapper(connection_id: str, websocket: WebSocket):
    state_manager.websockets[connection_id] = websocket
    return _original_add(connection_id, websocket)

def _remove_wrapper(connection_id: str):
    if connection_id in state_manager.websockets:
        del state_manager.websockets[connection_id]
    return _original_remove(connection_id)

state_manager.add_websocket_connection = _add_wrapper
state_manager.remove_websocket_connection = _remove_wrapper

# 2. Patch Player Mapping
if not hasattr(state_manager, "player_connections"):
    state_manager.player_connections = {}

    def register_player_connection(player_id: int, connection_id: str):
        state_manager.player_connections[player_id] = connection_id

    def get_player_connection_id(player_id: int) -> Optional[str]:
        return state_manager.player_connections.get(player_id)

    state_manager.register_player_connection = register_player_connection
    state_manager.get_player_connection_id = get_player_connection_id

# 3. [CRITICAL FIX] Thread-Safe Send Methods
async def _safe_send_json(websocket: WebSocket, message: dict):
    """Helper to send JSON safely, handling thread boundaries."""
    try:
        # If we are in a different loop (Game Thread), schedule on Main Loop
        if MAIN_LOOP and asyncio.get_running_loop() != MAIN_LOOP:
            future = asyncio.run_coroutine_threadsafe(websocket.send_json(message), MAIN_LOOP)
            # Optional: future.result() if needed
        else:
            # We are on Main Loop
            await websocket.send_json(message)
    except Exception as e:
        print(f"Send Error: {e}")

async def send_personal_message(player_id: int, message: dict):
    """Thread-safe personal message."""
    conn_id = state_manager.player_connections.get(player_id)
    if conn_id and conn_id in state_manager.websockets:
        ws = state_manager.websockets[conn_id]
        await _safe_send_json(ws, message)

async def broadcast_message_safe(message: dict):
    """Thread-safe broadcast (Monkey patch replacement)."""
    for conn_id, ws in state_manager.websockets.items():
        await _safe_send_json(ws, message)

# Apply the Safe Methods
state_manager.send_personal_message = send_personal_message
state_manager.broadcast_message = broadcast_message_safe


# --- LOBBY MANAGER ---
class LobbyManager:
    def __init__(self):
        self.active_players: List[Dict[str, Any]] = []
        self.current_ai_ids: List[int] = [] 
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()

    def disconnect(self, websocket: WebSocket):
        self.active_players = [p for p in self.active_players if p['ws'] != websocket]
        if self.active_players and not any(p['is_host'] for p in self.active_players):
            if len(self.active_players) > 0:
                self.active_players[0]['is_host'] = True

    async def handle_message(self, websocket: WebSocket, data: dict):
        msg_type = data.get("type")
        
        if msg_type == "LOGIN":
            name = data.get("name", "Unknown")
            # [NEW] Receive avatar_id from client
            avatar_id = data.get("avatar_id", "human") 
            
            is_host = len(self.active_players) == 0
            player_info = {
                "ws": websocket,
                "name": name,
                "id": str(uuid.uuid4()),
                "is_host": is_host,
                "avatar_id": avatar_id # Store avatar preference
            }
            self.active_players.append(player_info)
            await self.broadcast_lobby_state()
            
        elif msg_type == "SYNC_AI":
            player = next((p for p in self.active_players if p['ws'] == websocket), None)
            if player and player['is_host']:
                self.current_ai_ids = data.get("ai_ids", [])
                await self.broadcast_lobby_state()

        elif msg_type == "START_GAME":
            player = next((p for p in self.active_players if p['ws'] == websocket), None)
            if player and player['is_host']:
                game_config = data.get("game_config", {})
                
                try:
                    req = StartGameRequest(**game_config)
                    req.mode = "participate" 
                    ai_ids = req.ai_ids or []
                    
                    # Capture current lobby state to pass to game init
                    lobby_snapshot = self.active_players
                    
                    # Start Game with snapshot
                    await start_game_implementation(req, lobby_snapshot)
                    
                    # Redirect Players
                    all_ids = set(range(req.num_players))
                    ai_ids_set = set(ai_ids)
                    available_human_ids = sorted(list(all_ids - ai_ids_set))
                    
                    for i, p in enumerate(self.active_players):
                        if i < len(available_human_ids):
                            assigned_id = available_human_ids[i]
                            url = f"/avalon/participate?uid={assigned_id}"
                            await p['ws'].send_json({
                                "type": "GAME_START",
                                "url": url,
                                "player_id": assigned_id
                            })
                        else:
                            await p['ws'].send_json({
                                "type": "GAME_START",
                                "url": "/avalon/observe",
                                "player_id": -1
                            })
                            
                except Exception as e:
                    print(f"Start Game Error: {e}")
                    import traceback
                    traceback.print_exc()
                    await websocket.send_json({"type": "ERROR", "message": str(e)})

    async def broadcast_lobby_state(self):
        # [NEW] Include avatar_id in broadcast so lobby sees avatars
        players_list = [{"name": p["name"], "is_host": p["is_host"], "avatar_id": p.get("avatar_id")} for p in self.active_players]
        msg = {
            "type": "LOBBY_UPDATE", 
            "players": players_list,
            "ai_ids": self.current_ai_ids
        }
        for p in self.active_players:
            try:
                await _safe_send_json(p['ws'], msg)
            except:
                pass

lobby_manager = LobbyManager()


WEB_DIR = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("<h1>Games Web Interface</h1><p>index.html missing</p>")


def _page(path: str):
    f = STATIC_DIR / path
    if f.exists():
        return FileResponse(str(f))
    return HTMLResponse(f"<h1>Not found: {path}</h1>")

@app.get("/favicon.ico")
async def favicon():
    favicon_png = STATIC_DIR / "favicon.png"
    if favicon_png.exists():
        return FileResponse(str(favicon_png), media_type="image/png")

@app.get("/avalon/observe")
async def avalon_observe_page():
    return _page("avalon/observe.html")

@app.get("/avalon/participate")
async def avalon_participate_page():
    return _page("avalon/participate.html")

@app.get("/diplomacy/observe")
async def dip_observe_page():
    return _page("diplomacy/observe.html")

@app.get("/diplomacy/participate")
async def dip_participate_page():
    return _page("diplomacy/participate.html")


# --- WEBSOCKET HANDLERS ---

@app.websocket("/ws/lobby")
async def websocket_lobby(websocket: WebSocket):
    await lobby_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            await lobby_manager.handle_message(websocket, data)
    except WebSocketDisconnect:
        lobby_manager.disconnect(websocket)
        await lobby_manager.broadcast_lobby_state()
    except Exception as e:
        print(f"Lobby Error: {e}")
        lobby_manager.disconnect(websocket)


async def _handle_game_websocket(websocket: WebSocket, uid: Optional[int] = None):
    connection_id = str(uuid.uuid4())
    state_manager.add_websocket_connection(connection_id, websocket)
    
    if uid is not None:
        state_manager.register_player_connection(uid, connection_id)
    
    try:
        await _safe_send_json(websocket, state_manager.format_game_state())
        
        # [NEW] Send Global Metadata (Names/Avatars) to Client
        await _safe_send_json(websocket, {
            "type": "game_metadata",
            "metadata": CURRENT_GAME_METADATA,
            "my_id": uid
        })
        
        await _safe_send_json(websocket, {
            "type": "mode_info",
            "mode": state_manager.mode,
            "user_agent_id": uid if uid is not None else -1,
            "game": state_manager.game_state.get("game"),
        })
        
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "user_input":
                    agent_id = message.get("agent_id")
                    
                    # if uid is not None and int(agent_id) == int(uid):
                    #     content = message.get("content", "")
                    #     await state_manager.put_user_input(agent_id, content)
                    # elif state_manager.mode == "participate" and str(agent_id) == str(state_manager.user_agent_id):
                    #      content = message.get("content", "")
                    #      await state_manager.put_user_input(agent_id, content)

                    if uid is not None and int(agent_id) == int(uid):
                        content = message.get("content", "")
                        await state_manager.put_user_input(str(agent_id), content)
                    elif state_manager.mode == "participate" and str(agent_id) == str(state_manager.user_agent_id):
                        content = message.get("content", "")
                        await state_manager.put_user_input(str(agent_id), content)
     
            except WebSocketDisconnect:
                break
            except Exception as e:
                break
                
    except WebSocketDisconnect:
        pass
    finally:
        state_manager.remove_websocket_connection(connection_id)


@app.websocket("/ws/game")
async def websocket_game_endpoint(websocket: WebSocket, uid: Optional[int] = Query(None)):
    try:
        await websocket.accept()
    except Exception:
        return
    await _handle_game_websocket(websocket, uid)

@app.websocket("/ws")
async def websocket_legacy(websocket: WebSocket):
    try:
        await websocket.accept()
    except Exception:
        return
    await _handle_game_websocket(websocket, None)


# --- API MODELS & ENDPOINTS ---

# class StartGameRequest(BaseModel):
#     game: str = "avalon"
#     mode: str = "observe"
#     language: str = "en"
#     agent_configs: Dict[int | str, Dict[str, str]] | None = None
#     num_players: int = 5
#     user_agent_id: int = 0
#     preset_roles: list[dict] | None = None
#     selected_portrait_ids: list[int] | None = None
#     ai_ids: list[int] | None = None 
#     human_power: Optional[str] = None
#     max_phases: int = 20
#     negotiation_rounds: int = 3
#     power_names: list[str] | None = None
#     power_models: Dict[str, str] | None = None
class StartGameRequest(BaseModel):
    game: str = "avalon"
    mode: str = "observe"
    language: str = "en"
    agent_configs: Dict[int | str, Dict[str, str]] | None = None
    num_players: int = 5
    user_agent_id: int = 0
    preset_roles: list[dict] | None = None
    
    # [修改点] 将 list[int] 改为 list[int | str] 以支持 'h11' 这种 ID
    selected_portrait_ids: list[int | str] | None = None
    
    ai_ids: list[int] | None = None 
    human_power: Optional[str] = None
    max_phases: int = 20
    negotiation_rounds: int = 3
    power_names: list[str] | None = None
    power_models: Dict[str, str] | None = None

# async def start_game_implementation(request: StartGameRequest, lobby_players: List[Dict] = None):
#     print(f"--- [Game Init] Starting Game Check ---")
#     if state_manager.game_state.get("status") == "running":
#         state_manager.stop_game()
        
#     state_manager.reset()
#     state_manager.set_mode(request.mode, str(request.user_agent_id) if request.mode == "participate" else None, game=request.game)
    

    
#     # ========================== [智能补位逻辑 START] ==========================
#     # 1. 初始化 AI 列表
#     if request.ai_ids is None:
#         request.ai_ids = []
    
#     # 2. 找出所有坑位
#     all_slots = list(range(request.num_players)) # [0, 1, 2, 3, 4]
    
#     # 3. 找出用户已明确指定的 AI (绝对不让活人坐)
#     fixed_ai_set = set(request.ai_ids)
    
#     # 4. 找出理论上留给人类的坑位 (例如 [0, 1, 2, 4])
#     available_human_slots = [i for i in all_slots if i not in fixed_ai_set]
#     available_human_slots.sort()
    
#     # 5. 计算实际有多少个活人
#     # 如果 lobby_players 为空 (API直接调用)，假设只有 1 个活人
#     real_human_count = len(lobby_players) if lobby_players else 1
#     print(f"--- [Game Init] Total Slots: {request.num_players}, Fixed AI: {fixed_ai_set}")
#     print(f"--- [Game Init] Available Human Slots: {available_human_slots}")
#     print(f"--- [Game Init] Real Humans Connected: {real_human_count}")

#     # 6. 分配座位：
#     # 真正的人类 ID = 预留坑位中的前 N 个 (N = 活人数)
#     assigned_human_ids = available_human_slots[:real_human_count]
    
#     # 剩余没坐满的坑位 = 预留坑位中的后半部分 -> 必须转为 AI
#     empty_slots_to_fill = available_human_slots[real_human_count:]
    
#     if empty_slots_to_fill:
#         print(f"⚠️ [Auto-Fill] Not enough humans! Converting slots {empty_slots_to_fill} to AI.")
#         request.ai_ids.extend(empty_slots_to_fill)
#         # 重新排序并去重
#         request.ai_ids = sorted(list(set(request.ai_ids)))
#     else:
#         print(f"--- [Game Init] Human slots match player count perfectly.")

#     # ========================== [智能补位逻辑 END] ==========================

#     # [CRITICAL] Populate Global Metadata
#     # 注意：这里的 ai_ids_set 必须使用我们刚刚更新过的 request.ai_ids
#     ai_ids_set = set(request.ai_ids)
    
#     players_meta = []
#     selected_portrait_ids = request.selected_portrait_ids or list(range(1, request.num_players + 1))
#     ai_portrait_idx = 0
#     human_idx_counter = 0 # 用来遍历 lobby_players
    
#     for i in all_slots:
#         is_human = (i not in ai_ids_set)
#         p_data = {"id": i, "is_human": is_human}
        
#         if is_human:
#             # Map to Lobby Player info
#             # 只有当该 ID 确实被分配给人类时，才从 lobby_players 取名字
#             if lobby_players and human_idx_counter < len(lobby_players):
#                 lp = lobby_players[human_idx_counter]
#                 p_data["name"] = lp["name"]
#                 p_data["portrait_id"] = lp.get("avatar_id", "human")
#                 human_idx_counter += 1
#             else:
#                 p_data["name"] = f"Player {i}"
#                 p_data["portrait_id"] = "human"
#         else:
#             # AI Logic
#             if ai_portrait_idx < len(selected_portrait_ids):
#                 pid = selected_portrait_ids[ai_portrait_idx]
#                 ai_portrait_idx += 1
#             else:
#                 pid = (i % 15) + 1
#             p_data["portrait_id"] = pid
            
#             agent_name = f"Agent {i}"
#             if request.agent_configs:
#                 cfg = request.agent_configs.get(pid) or request.agent_configs.get(str(pid))
#                 if cfg and cfg.get("base_model"):
#                     agent_name = cfg.get("base_model")
#             p_data["name"] = agent_name
            
#         players_meta.append(p_data)
        
#     CURRENT_GAME_METADATA["num_players"] = request.num_players
#     CURRENT_GAME_METADATA["players"] = players_meta
    
#     start_game_thread(
#         state_manager=state_manager,
#         game=request.game,
#         mode=request.mode,
#         language=request.language,
#         num_players=request.num_players,
#         user_agent_id=request.user_agent_id,
#         preset_roles=request.preset_roles,
#         selected_portrait_ids=request.selected_portrait_ids,
#         agent_configs=request.agent_configs or {},
#         ai_ids=request.ai_ids, # 传入更新后的 AI 列表
#         human_power=request.human_power,
#         max_phases=request.max_phases,
#         negotiation_rounds=request.negotiation_rounds,
#         power_names=request.power_names,
#         power_models=request.power_models or {},
#     )

#     # 重定向逻辑修正：准确告诉每个 WebSocket 它们是谁
#     if lobby_players:
#         # assigned_human_ids 是我们在上面计算出来的，例如 [0, 1]
#         for idx, p in enumerate(lobby_players):
#             if idx < len(assigned_human_ids):
#                 # 这是一个参与游戏的玩家
#                 assigned_id = assigned_human_ids[idx]
#                 url = f"/avalon/participate?uid={assigned_id}"
#                 await _safe_send_json(p['ws'], {
#                     "type": "GAME_START",
#                     "url": url,
#                     "player_id": assigned_id
#                 })
#             else:
#                 # 观众
#                 await _safe_send_json(p['ws'], {
#                     "type": "GAME_START",
#                     "url": "/avalon/observe",
#                     "player_id": -1
#                 })
async def start_game_implementation(request: StartGameRequest, lobby_players: List[Dict] = None):
    print(f"--- [Game Init] Starting Game Check ---")
    if state_manager.game_state.get("status") == "running":
        state_manager.stop_game()
        
    state_manager.reset()
    state_manager.set_mode(request.mode, str(request.user_agent_id) if request.mode == "participate" else None, game=request.game)
    
    # 1. 确定有多少个真人，多少个 AI
    # 如果 lobby_players 存在，以它为准；否则默认为 1 个真人
    real_human_count = len(lobby_players) if lobby_players else 1
    total_slots = request.num_players
    
    # 确保真人数量不超过总人数
    if real_human_count > total_slots:
        real_human_count = total_slots

    # 2. 准备两个“头像队列”
    # 队列 A: 真人头像 (从 Lobby 数据获取)
    human_avatars_queue = []
    if lobby_players:
        for p in lobby_players:
            human_avatars_queue.append(p.get("avatar_id", "human"))
    else:
        # 单机模式/API调用兜底：使用 request 里的第一个头像，或者默认
        first_portrait = request.selected_portrait_ids[0] if request.selected_portrait_ids else "human"
        human_avatars_queue.append(first_portrait)

    # 队列 B: AI 头像 (从 Host 选择的 ai_ids 获取)
    # 注意：前端发来的 request.ai_ids 实际上是 Host 选中的那些“头像ID” (例如 [7, 8, 9])
    ai_avatars_queue = request.ai_ids if request.ai_ids else []
    
    # 3. 开始分配座位 (0 到 N-1)
    full_portrait_list = []
    players_meta = []
    
    # 我们不仅要收集 ID，还要收集正确的 Slot 分配给 request.ai_ids 以便引擎识别身份
    # 重置 request.ai_ids 为“座位号列表”，而不是“头像ID列表”
    final_ai_slot_indices = []

    for i in range(total_slots):
        p_data = {"id": i}
        final_pid = 1 # 默认值
        
        # 规则：前 N 个位置给真人，剩下的给 AI
        is_human_slot = (i < real_human_count)
        
        if is_human_slot:
            # --- 分配给真人 ---
            p_data["is_human"] = True
            
            # 从真人队列取头像
            if len(human_avatars_queue) > 0:
                final_pid = human_avatars_queue.pop(0)
            else:
                final_pid = "human" # 理论上不应发生
                
            # 名字
            if lobby_players and i < len(lobby_players):
                p_data["name"] = lobby_players[i]["name"]
            else:
                p_data["name"] = f"Player {i}"
                
        else:
            # --- 分配给 AI ---
            p_data["is_human"] = False
            final_ai_slot_indices.append(i) # 记录这个座位号是 AI
            
            # 从 AI 队列取头像
            if len(ai_avatars_queue) > 0:
                final_pid = ai_avatars_queue.pop(0)
            else:
                # 如果 Host 选的头像不够用了，就按座位号取模生成
                final_pid = (i % 15) + 1
            
            # AI 名字 (模型名)
            agent_name = f"Agent {i}"
            if request.agent_configs:
                # 尝试用头像ID或座位号查找配置
                cfg = request.agent_configs.get(i) or request.agent_configs.get(str(i))
                if not cfg:
                    cfg = request.agent_configs.get(final_pid) or request.agent_configs.get(str(final_pid))
                if cfg and cfg.get("base_model"):
                    agent_name = cfg.get("base_model")
            p_data["name"] = agent_name

        # 存入列表
        p_data["portrait_id"] = final_pid
        full_portrait_list.append(final_pid)
        players_meta.append(p_data)

    # 4. 更新全局元数据
    CURRENT_GAME_METADATA["num_players"] = total_slots
    CURRENT_GAME_METADATA["players"] = players_meta
    
    print(f"--- [Game Init] Humans: {real_human_count}, AI Slots: {final_ai_slot_indices}")
    print(f"--- [Game Init] Full Portrait List: {full_portrait_list}")

    # 5. 启动引擎
    # 注意：这里我们将 ai_ids 参数更新为真正的“AI座位号列表”，
    # 这样引擎就知道哪些位置该由电脑托管了。
    
    start_game_thread(
        state_manager=state_manager,
        game=request.game,
        mode=request.mode,
        language=request.language,
        num_players=request.num_players,
        user_agent_id=request.user_agent_id,
        preset_roles=request.preset_roles,
        selected_portrait_ids=full_portrait_list, # <--- 传入这个完美拼接的列表
        agent_configs=request.agent_configs or {},
        ai_ids=final_ai_slot_indices,             # <--- 传入计算好的 AI 座位号
        human_power=request.human_power,
        max_phases=request.max_phases,
        negotiation_rounds=request.negotiation_rounds,
        power_names=request.power_names,
        power_models=request.power_models or {},
    )

    # 6. 广播跳转 (保持不变)
    if lobby_players:
        for idx, p in enumerate(lobby_players):
            if idx < real_human_count:
                # 分配座位号，就是 idx 本身 (因为我们是按顺序排的)
                assigned_id = idx
                url = f"/avalon/participate?uid={assigned_id}"
                await _safe_send_json(p['ws'], {
                    "type": "GAME_START",
                    "url": url,
                    "player_id": assigned_id
                })
            else:
                await _safe_send_json(p['ws'], {
                    "type": "GAME_START",
                    "url": "/avalon/observe",
                    "player_id": -1
                })
                
@app.post("/api/start-game")
async def start_game_api(request: StartGameRequest):
    try:
        await start_game_implementation(request)
        return {"status": "ok", "message": "Game started"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/stop-game")
async def stop_game():
    if state_manager.game_state.get("status") != "running":
        raise HTTPException(status_code=400, detail="No game is currently running")
    
    state_manager.stop_game()
    if hasattr(state_manager, '_game_task') and state_manager._game_task:
        try:
            state_manager._game_task.cancel()
        except Exception:
            pass
            
    return {"status": "ok", "message": "Game stopped"}


@app.get("/api/options")
async def get_options(game: str | None = None):
    # Same as before, keeping file structure valid
    import os
    def _to_ui_lang(raw: str | None) -> str:
        lang = (raw or "").lower().strip()
        return "zh" if lang in {"zh", "zn", "cn", "zh-cn", "zh_cn", "chinese"} else "en"

    if not game:
        web_config_path = Path(__file__).parent / "web_config.yaml"
        result = {"portraits": {}, "default_model": {}}
        if web_config_path.exists():
            web_cfg = load_config(web_config_path)
            if isinstance(web_cfg, dict):
                result["portraits"] = web_cfg.get('portraits', {})
                default_role = web_cfg.get("default_role", {})
                
                default_model = {}
                if isinstance(default_role, dict):
                    model_cfg = default_role.get("model", {})
                    agent_cfg = default_role.get("agent", {})
                    default_model["model_name"] = model_cfg.get("model_name", "")
                    default_model["api_base"] = model_cfg.get("url", "") or model_cfg.get("api_base", "")
                    default_model["api_key"] = model_cfg.get("api_key", "")
                    default_model["agent_class"] = agent_cfg.get("type", "")
                    if not default_model.get("api_key"):
                        default_model["api_key"] = os.getenv("OPENAI_API_KEY", "")
                    if not default_model.get("api_base"):
                        default_model["api_base"] = os.getenv("OPENAI_BASE_URL", "")
                result["default_model"] = default_model
        return result

    if game == "diplomacy":
        yaml_path = os.environ.get("DIPLOMACY_CONFIG_YAML", "games/games/diplomacy/configs/default_config.yaml")
        diplomacy_cfg = load_config(yaml_path)['game']
        lang = _to_ui_lang(diplomacy_cfg['language'])
        return {
            "powers": diplomacy_cfg['power_names'],
            "defaults": {
                "mode": "observe",
                "human_power": (diplomacy_cfg['power_names'][0] if diplomacy_cfg['power_names'] else "ENGLAND"),
                "max_phases": diplomacy_cfg['max_phases'],
                "map_name": diplomacy_cfg['map_name'],
                "negotiation_rounds": diplomacy_cfg['negotiation_rounds'],
                "language": lang,
            },
        }

    if game == "avalon":
        yaml_path = os.environ.get("AVALON_CONFIG_YAML", "games/games/avalon/configs/default_config.yaml")
        avalon_cfg = load_config(yaml_path)['game']
        return {
            "roles": avalon_cfg.get("roles_name", []),
            "defaults": {
                "num_players": int(avalon_cfg.get("num_players", 5) or 5),
                "language": _to_ui_lang(str(avalon_cfg.get("language", "en"))),
            },
        }

    raise HTTPException(status_code=404, detail="options only for avalon/diplomacy")


def run_server(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()