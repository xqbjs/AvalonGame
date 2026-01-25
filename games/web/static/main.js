const CONFIG = {
  portraitsBase: "/static/portraits/",
  portraitCount: 15,      // AI 随机池 (portrait_1 ~ portrait_15)
  humanAvatarCount: 11,    // [NEW] 真人专用池数量 (portrait_h1 ~ portrait_h6) - 请根据你实际图片数量修改这里
  travelDuration: 260,
};

const STORAGE_KEYS = {
  AGENT_CONFIGS: "AgentConfigs.v1",
  GAME_OPTIONS: "GameOptions.v1",
  WEB_CONFIG_LOADED: "WebConfigLoaded.v1",
  LAST_GAME_OPTIONS: "LastGameOptions.v1",
  CONFIG_UPDATE_TIME: "ConfigUpdateTime.v1",
};

// Local Avatar State
let myAvatarId = "human";

// --- WEBSOCKET LOBBY CLIENT ---
const LobbyClient = {
  ws: null,
  connected: false,
  isHost: false,
  myNickname: "",

  connect(nickname) {
    if (this.ws) this.ws.close();

    this.myNickname = nickname.trim();

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const url = `${protocol}//${host}/ws/lobby`;

    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      this.connected = true;
      addStatusMessage("Connected to lobby server.");

      if (DOM.connectionStatus) DOM.connectionStatus.classList.add("connected");
      if (DOM.connectBtn) {
        DOM.connectBtn.textContent = "Joined";
        DOM.connectBtn.disabled = true;
      }
      if (DOM.nicknameInput) DOM.nicknameInput.disabled = true;

      if (DOM.avatarPreview) DOM.avatarPreview.style.pointerEvents = "none";
      if (DOM.avatarPreview) DOM.avatarPreview.style.opacity = "0.7";

      this.send({
        type: "LOGIN",
        name: this.myNickname,
        avatar_id: myAvatarId
      });
    };

    this.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        this.handleMessage(msg);
      } catch (e) {
        console.error("Lobby msg parse error:", e);
      }
    };

    this.ws.onclose = () => {
      this.connected = false;
      this.isHost = false;

      if (DOM.connectionStatus) DOM.connectionStatus.classList.remove("connected");
      if (DOM.connectBtn) {
        DOM.connectBtn.textContent = "Connect";
        DOM.connectBtn.disabled = false;
      }
      if (DOM.nicknameInput) DOM.nicknameInput.disabled = false;
      if (DOM.avatarPreview) DOM.avatarPreview.style.pointerEvents = "auto";
      if (DOM.avatarPreview) DOM.avatarPreview.style.opacity = "1";

      addStatusMessage("Disconnected from lobby.");

      state.lobbyPlayers = [];
      disableHostControls(false);
      updateCounter();
      layoutTablePlayers();
    };

    this.ws.onerror = (err) => {
      addStatusMessage("Connection error.");
      console.error(err);
    };
  },

  send(data) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  },

  handleMessage(msg) {
    switch (msg.type) {
      case "LOBBY_UPDATE":
        state.lobbyPlayers = msg.players || [];

        const me = state.lobbyPlayers.find(p => p.name === this.myNickname);
        const oldIsHost = this.isHost;

        if (me && me.is_host) {
          this.isHost = true;
          if (!oldIsHost) addStatusMessage("You are the Host 👑");

          const serverIds = (msg.ai_ids || []).map(Number).sort((a, b) => a - b);
          const localIds = Array.from(state.selectedIds).map(Number).sort((a, b) => a - b);

          if (localIds.length > 0 && JSON.stringify(serverIds) !== JSON.stringify(localIds)) {
            console.log("Syncing my AI selection to server...");
            this.send({
              type: "SYNC_AI",
              ai_ids: localIds
            });
          }

        } else {
          this.isHost = false;
          if (oldIsHost) addStatusMessage("You are now a Guest");

          if (msg.ai_ids && Array.isArray(msg.ai_ids)) {
            const serverIds = msg.ai_ids.map(id => parseInt(id, 10));
            state.selectedIds = new Set(serverIds);
            state.selectedIdsOrder = [...serverIds];

            renderPortraits();
            updateCounter();
            layoutTablePlayers();
          }
        }

        if (!state.selectedGame || state.selectedGame === "") {
          setGame('avalon');
        }
        const tablePreview = document.getElementById("table-preview");
        if (tablePreview) tablePreview.classList.add("has-game");

        disableHostControls(!this.isHost);
        updateCounter();
        layoutTablePlayers();
        break;

      case "GAME_START":
        addStatusMessage("🚀 Game starting! Redirecting...");
        sessionStorage.setItem("my_player_id", msg.player_id);
        window.location.href = msg.url;
        break;

      case "ERROR":
        addStatusMessage(`❌ Error: ${msg.message}`);
        if (DOM.startBtn) {
          DOM.startBtn.disabled = false;
          updateCounter();
        }
        break;
    }
  }
};

function disableHostControls(disabled) {
  const els = [
    document.getElementById("avalon-num-players"),
    document.getElementById("avalon-language"),
    document.getElementById("random-select-btn"),
    document.getElementById("avalon-reroll-roles")
  ];

  els.forEach(el => {
    if (el) {
      el.disabled = disabled;
      el.style.opacity = disabled ? "0.6" : "1";
      el.style.cursor = disabled ? "not-allowed" : "pointer";
    }
  });

  const startBtn = document.getElementById("start-btn");
  if (startBtn && disabled) {
    startBtn.disabled = true;
    startBtn.textContent = "Waiting for Host...";
    startBtn.classList.remove("highlight");
  }

  const portraits = document.querySelectorAll(".portrait-card");
  portraits.forEach(p => {
    p.style.pointerEvents = disabled ? "none" : "auto";
    p.style.opacity = disabled ? "0.7" : "1";
  });
}

function loadAgentConfigs() {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.AGENT_CONFIGS);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function saveAgentConfigs(configs) {
  try {
    localStorage.setItem(STORAGE_KEYS.AGENT_CONFIGS, JSON.stringify(configs));
  } catch (e) {
    console.error("Failed to save agent configs:", e);
  }
}

function loadGameOptions() {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.GAME_OPTIONS);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function saveGameOptions(options) {
  try {
    localStorage.setItem(STORAGE_KEYS.GAME_OPTIONS, JSON.stringify(options));
  } catch (e) {
    console.error("Failed to save game options:", e);
  }
}

function shouldLoadWebConfig() {
  const loaded = localStorage.getItem(STORAGE_KEYS.WEB_CONFIG_LOADED);
  return !loaded || loaded !== "true";
}

function markWebConfigLoaded() {
  localStorage.setItem(STORAGE_KEYS.WEB_CONFIG_LOADED, "true");
}

async function loadWebConfig() {
  if (window.location.protocol === "file:") return;

  const fromCharacterConfig = sessionStorage.getItem('fromCharacterConfig');
  if (fromCharacterConfig === 'true') {
    sessionStorage.removeItem('fromCharacterConfig');

    const hasLoaded = shouldLoadWebConfig() === false;
    if (hasLoaded) {
      const existingConfigs = loadAgentConfigs();
      if (Object.keys(existingConfigs).length >= 1) return;
      localStorage.removeItem(STORAGE_KEYS.WEB_CONFIG_LOADED);
    }
  } else {
    localStorage.removeItem(STORAGE_KEYS.AGENT_CONFIGS);
    localStorage.removeItem(STORAGE_KEYS.WEB_CONFIG_LOADED);
  }

  try {
    const resp = await fetch("/api/options");
    if (!resp.ok) return;

    const webOpts = await resp.json();
    const portraits = webOpts.portraits || {};
    const defaultModel = webOpts.default_model || {};

    const existingConfigs = loadAgentConfigs();
    let updated = false;

    for (let id = 1; id <= CONFIG.portraitCount; id++) {
      if (!existingConfigs[id]) existingConfigs[id] = {};

      const portraitCfg = portraits[id];
      const hasPortraitCfg = portraitCfg && typeof portraitCfg === "object";

      if (hasPortraitCfg && portraitCfg.name) {
        if (!existingConfigs[id].name) {
          existingConfigs[id].name = portraitCfg.name;
          updated = true;
        }
      }

      let modelName = null;
      if (hasPortraitCfg && portraitCfg.model && portraitCfg.model.model_name) {
        modelName = portraitCfg.model.model_name;
      } else if (defaultModel.model_name) {
        modelName = defaultModel.model_name;
      }
      if (modelName && !existingConfigs[id].base_model) {
        existingConfigs[id].base_model = modelName;
        updated = true;
      }

      let apiBase = null;
      if (hasPortraitCfg && portraitCfg.model) {
        apiBase = portraitCfg.model.url || portraitCfg.model.api_base || null;
      }
      if (!apiBase && defaultModel.api_base) {
        apiBase = defaultModel.api_base;
      }
      if (apiBase && !existingConfigs[id].api_base) {
        existingConfigs[id].api_base = apiBase;
        updated = true;
      }

      let apiKey = null;
      if (hasPortraitCfg && portraitCfg.model && portraitCfg.model.api_key) {
        apiKey = portraitCfg.model.api_key;
      } else if (defaultModel.api_key) {
        apiKey = defaultModel.api_key;
      }
      if (apiKey && !existingConfigs[id].api_key) {
        existingConfigs[id].api_key = apiKey;
        updated = true;
      }

      let agentClass = null;
      if (hasPortraitCfg && portraitCfg.agent && portraitCfg.agent.type) {
        agentClass = portraitCfg.agent.type;
      } else if (defaultModel.agent_class) {
        agentClass = defaultModel.agent_class;
      }
      if (agentClass && !existingConfigs[id].agent_class) {
        existingConfigs[id].agent_class = agentClass;
        updated = true;
      }
    }

    if (updated) {
      saveAgentConfigs(existingConfigs);
    }

    markWebConfigLoaded();
  } catch (e) {
    console.warn("Failed to load web config:", e);
  }
}

const AVALON_ROLE_MAP = {
  "Merlin": 0,
  "Percival": 1,
  "Servant": 2,
  "Minion": 3,
  "Assassin": 4,
};
const AVALON_GOOD_ROLES = ["Merlin", "Percival", "Servant"];

const state = {
  selectedIds: new Set(),
  selectedIdsOrder: [],
  selectedGame: "",
  selectedMode: "observe",
  diplomacyOptions: null,
  avalonOptions: null,
  diplomacyPowerOrder: null,
  avalonRoleOrder: null,
  avalonPreviewRoles: null,
  diplomacyPreviewPowers: null,
  lobbyPlayers: [],
};

let DOM = {};

function polarPositions(count, radiusX, radiusY) {
  return Array.from({ length: count }).map((_, i) => {
    const angle = (Math.PI * 2 * i) / count - Math.PI / 2;
    return { x: radiusX * Math.cos(angle), y: radiusY * Math.sin(angle) };
  });
}

function computeRedirectUrl(game, mode) {
  if (window.location.protocol !== "file:") return `/${game}/${mode}`;
  return `./static/${game}/${mode}.html`;
}

// [JS FIX] 替换 main.js 中的 addStatusMessage 函数
function addStatusMessage(text) {
  if (!DOM.statusLog) return;

  // --- Mobile Logic: 单条弹幕模式 ---
  if (window.innerWidth <= 768) {
    // 1. 强制清空旧消息 (单例模式)
    DOM.statusLog.innerHTML = "";

    // 2. 创建新气泡
    const bubble = document.createElement("div");
    bubble.className = "status-bubble";
    bubble.textContent = text;
    DOM.statusLog.appendChild(bubble);

    // 3. 2.5秒后自动移除，保持界面清爽
    setTimeout(() => {
      if (bubble.parentNode) bubble.parentNode.removeChild(bubble);
    }, 2500);
    return;
  }

  // --- Desktop Logic (原有逻辑) ---
  const bubble = document.createElement("div");
  bubble.className = "status-bubble";
  bubble.textContent = text;
  DOM.statusLog.appendChild(bubble);

  while (DOM.statusLog.children.length > 20) {
    DOM.statusLog.removeChild(DOM.statusLog.firstChild);
  }

  setTimeout(() => {
    if (DOM.statusLog) {
      DOM.statusLog.scrollTop = DOM.statusLog.scrollHeight;
    }
  }, 50);
}

function ensureSeat(id) {
  if (!DOM.tablePlayers) return null;

  let seat = DOM.tablePlayers.querySelector(`.seat[data-id="${id}"]`);
  if (seat) return seat;

  seat = document.createElement("div");
  seat.className = "seat enter";
  seat.dataset.id = String(id);

  let src = "";
  let alt = "";
  let modelLabel = "";
  let nameLabel = "";

  if (String(id).startsWith("human_")) {
    const pName = String(id).replace("human_", "");
    src = `${CONFIG.portraitsBase}portrait_human.png`;

    // Try to find custom avatar from lobby data
    const lobbyPlayer = state.lobbyPlayers.find(p => p.name === pName);
    if (lobbyPlayer && lobbyPlayer.avatar_id && lobbyPlayer.avatar_id !== "human") {
      src = `${CONFIG.portraitsBase}portrait_${lobbyPlayer.avatar_id}.png`;
    }

    alt = pName;
    nameLabel = `<div class="player-name">${pName}</div>`;

  } else if (id === "human") {
    src = `${CONFIG.portraitsBase}portrait_human.png`;
    if (myAvatarId !== "human") {
      src = `${CONFIG.portraitsBase}portrait_${myAvatarId}.png`;
    }
    alt = "Human";

  } else {
    src = `${CONFIG.portraitsBase}portrait_${id}.png`;
    alt = `Agent ${id}`;

    const AgentConfigs = loadAgentConfigs();
    const cfg = AgentConfigs[id] || {};
    const baseModel = cfg.base_model || "";
    modelLabel = baseModel ? `<div class="seat-model">${baseModel}</div>` : "";
  }

  seat.innerHTML = `
    ${nameLabel}
    ${modelLabel}
    <div class="seat-label"></div>
    <img src="${src}" alt="${alt}">
  `;
  seat.style.left = "50%";
  seat.style.top = "50%";
  seat.style.transform = "translate(-50%, -50%) scale(0.8)";
  seat.style.pointerEvents = "auto";

  const isAI = !String(id).startsWith("human");
  seat.style.cursor = isAI ? "pointer" : "default";

  DOM.tablePlayers.appendChild(seat);

  requestAnimationFrame(() => seat.classList.remove("enter"));
  return seat;
}

function checkRoleConflict(seatId, newRole, game) {
  if (!game) return null;
  const seats = DOM.tablePlayers.querySelectorAll(".seat");
  const currentSelections = [];
  seats.forEach(seat => {
    const select = seat.querySelector(".seat-label select");
    if (!select) return;
    let value = select.value;
    if (seat.dataset.id === seatId) value = newRole;
    if (value && value !== "") currentSelections.push(value);
  });

  let expectedList = [];
  if (game === "avalon") {
    if (state.avalonOptions && Array.isArray(state.avalonOptions.roles)) {
      expectedList = state.avalonOptions.roles.slice();
    } else {
      expectedList = ["Merlin", "Servant", "Servant", "Minion", "Assassin"];
    }
  } else if (game === "diplomacy") {
    if (state.diplomacyOptions && Array.isArray(state.diplomacyOptions.powers)) {
      expectedList = state.diplomacyOptions.powers.slice();
    }
  }

  if (expectedList.length === 0) return null;

  const counts = {};
  currentSelections.forEach(role => { counts[role] = (counts[role] || 0) + 1; });

  for (const [role, count] of Object.entries(counts)) {
    if (count > 1) {
      return { hasConflict: true, message: `${role} appears ${count} times!`, conflicts: [] };
    }
  }

  const missing = expectedList.filter(role => !currentSelections.includes(role));
  if (missing.length > 0) {
    return { hasConflict: true, message: `Missing roles: ${missing.join(", ")}`, conflicts: [] };
  }
  return null;
}

function setSeatLabelBySeatId(seatId, text, options = []) {
  const el = DOM.tablePlayers && DOM.tablePlayers.querySelector(`.seat[data-id="${seatId}"]`);
  if (!el) return;
  const labelContainer = el.querySelector(".seat-label");
  if (!labelContainer) return;

  if (!text && options.length === 0) {
    el.classList.remove("has-label");
    labelContainer.innerHTML = "";
    return;
  }

  if (options.length > 0) {
    const select = document.createElement("select");
    let currentValue = text || options[0];
    options.forEach(opt => {
      const option = document.createElement("option");
      option.value = opt;
      option.textContent = opt;
      if (opt === currentValue) option.selected = true;
      select.appendChild(option);
    });

    select.addEventListener("click", (e) => { e.stopPropagation(); });
    select.addEventListener("mousedown", (e) => { e.stopPropagation(); });
    select.addEventListener("change", (e) => {
      e.stopPropagation();
      const newRole = e.target.value;
      const conflict = checkRoleConflict(seatId, newRole, state.selectedGame);
      if (conflict && conflict.hasConflict) addStatusMessage(`⚠ ${conflict.message}`);

      currentValue = newRole;
      if (state.selectedGame === "avalon") {
        const ids = getTablePlayerIds();
        const idx = ids.indexOf(seatId);
        if (idx !== -1 && state.avalonRoleOrder) {
          state.avalonRoleOrder[idx] = newRole;
          state.avalonPreviewRoles = state.avalonRoleOrder.map((roleName) => ({
            role_id: AVALON_ROLE_MAP[roleName] || 0,
            role_name: roleName,
            is_good: AVALON_GOOD_ROLES.includes(roleName),
          }));
        }
      }
      updateSelectionHint();
      updateTableRoleStats();
    });

    labelContainer.innerHTML = "";
    labelContainer.appendChild(select);
    el.classList.add("has-label");
  } else {
    labelContainer.textContent = String(text);
    el.classList.add("has-label");
  }
}

function getTablePlayerIds() {
  return Array.from(DOM.tablePlayers.querySelectorAll(".seat")).map(s => s.dataset.id);
}

function shouldShowPreview() {
  return state.selectedMode !== "participate";
}

function setRandomButtonsEnabled() {
  const disabled = state.selectedMode === "participate";
  const aBtn = document.getElementById("avalon-reroll-roles");
  const dBtn = document.getElementById("diplomacy-shuffle-powers");
  const isHost = LobbyClient.connected && LobbyClient.isHost;

  if (aBtn) aBtn.disabled = disabled && !isHost;
  if (dBtn) dBtn.disabled = disabled;
}

// function requiredCountForPreview() {
//   const game = state.selectedGame;
//   if (!game) return 0;
//   if (game === "avalon") return 5;
//   if (game === "diplomacy") return 7;
//   return 0;
// }
function requiredCountForPreview() {
  const game = state.selectedGame;
  if (!game) return 0;

  if (game === "avalon") {
    // [修改点] 动态读取下拉框的值，而不是写死 5
    const numPlayersEl = document.getElementById("avalon-num-players");
    return numPlayersEl ? parseInt(numPlayersEl.value, 10) : 5;
  }

  if (game === "diplomacy") return 7; // Diplomacy 暂时还是固定 7 人
  return 0;
}

// function avalonAssignRolesFor5() {
//   if (state.avalonOptions && Array.isArray(state.avalonOptions.roles)) {
//     const roles = state.avalonOptions.roles.slice();
//     return shuffleInPlace(roles);
//   }
//   const roles = ["Merlin", "Servant", "Servant", "Minion", "Assassin"];
//   return shuffleInPlace(roles.slice());
// }
// [修改点] 重命名并增强逻辑，支持 5-10 人
function avalonAssignRoles() {
  // 如果后端给了配置，优先用后端的
  if (state.avalonOptions && Array.isArray(state.avalonOptions.roles)) {
    const roles = state.avalonOptions.roles.slice();
    return shuffleInPlace(roles);
  }

  // 获取当前选择的人数
  const numPlayersEl = document.getElementById("avalon-num-players");
  const num = numPlayersEl ? parseInt(numPlayersEl.value, 10) : 5;

  // 根据人数生成标准配置 (Avalon Standard Setup)
  let roles = [];
  if (num === 5) roles = ["Merlin", "Percival", "Servant", "Minion", "Assassin"]; // 改回标准 3好2坏 (含Percival)
  else if (num === 6) roles = ["Merlin", "Percival", "Servant", "Servant", "Minion", "Assassin"];
  else if (num === 7) roles = ["Merlin", "Percival", "Servant", "Servant", "Minion", "Assassin", "Minion"]; // 4好3坏
  else if (num === 8) roles = ["Merlin", "Percival", "Servant", "Servant", "Servant", "Minion", "Assassin", "Minion"];
  else if (num === 9) roles = ["Merlin", "Percival", "Servant", "Servant", "Servant", "Servant", "Minion", "Assassin", "Mordred"];
  else if (num === 10) roles = ["Merlin", "Percival", "Servant", "Servant", "Servant", "Servant", "Minion", "Assassin", "Mordred", "Oberon"];
  else roles = Array(num).fill("Servant"); // 兜底

  // 如果你想用简单版（只有 Merlin + Assassin + Minion），可以简化上面的列表
  // 这里使用的是包含 Percival 的标准版，你可以根据需要调整

  return shuffleInPlace(roles);
}

function updateTableHeadPreview() {
  if (!DOM.tablePlayers) return;

  Array.from(DOM.tablePlayers.querySelectorAll(".seat")).forEach(seat => {
    const label = seat.querySelector(".seat-label");
    if (label) label.innerHTML = "";
    seat.classList.remove("has-label");
  });

  setRandomButtonsEnabled();

  const game = state.selectedGame;
  const aiCount = state.selectedIds.size;
  const humanCount = LobbyClient.connected ? state.lobbyPlayers.length : (state.selectedMode === "participate" ? 1 : 0);
  const totalCount = aiCount + humanCount;
  const isParticipate = state.selectedMode === "participate";

  if (game === "avalon") {
    const numPlayersEl = document.getElementById("avalon-num-players");
    const numPlayers = numPlayersEl ? parseInt(numPlayersEl.value, 10) : 5;

    if (LobbyClient.connected) {
      if (totalCount !== numPlayers) {
        updateSelectionHint();
        return;
      }
    } else {
      const required = (isParticipate) ? (numPlayers - 1) : numPlayers;
      const ids = state.selectedIdsOrder.filter(id => state.selectedIds.has(id));
      if (ids.length !== required) {
        updateSelectionHint();
        return;
      }
    }
  }

  if (game === "avalon") {
    const numPlayersEl = document.getElementById("avalon-num-players");
    const numPlayers = numPlayersEl ? parseInt(numPlayersEl.value, 10) : 5;
    if (!state.avalonRoleOrder || state.avalonRoleOrder.length !== numPlayers) {
      if (numPlayers === 5) {
        // state.avalonRoleOrder = avalonAssignRolesFor5();
        state.avalonRoleOrder = avalonAssignRoles();
      } else {
        state.avalonRoleOrder = Array(numPlayers).fill("Servant");
      }
    }
    state.avalonPreviewRoles = state.avalonRoleOrder.map((roleName, idx) => ({
      role_id: AVALON_ROLE_MAP[roleName] || 0,
      role_name: roleName,
      is_good: AVALON_GOOD_ROLES.includes(roleName),
    }));

    if (shouldShowPreview()) {
      let allRoles = [];
      if (state.avalonOptions && Array.isArray(state.avalonOptions.roles)) {
        allRoles = [...new Set(state.avalonOptions.roles)];
      } else {
        // allRoles = numPlayers === 5 ? ["Merlin", "Servant", "Minion", "Assassin"] : ["Servant"];
        allRoles = ["Merlin", "Percival", "Servant", "Minion", "Assassin", "Mordred", "Morgana", "Oberon"];
      }


      const tableIds = getTablePlayerIds();
      tableIds.forEach((sid, idx) => {
        if (idx < state.avalonRoleOrder.length) {
          setSeatLabelBySeatId(sid, state.avalonRoleOrder[idx], allRoles);
        }
      });
    }
  }

  updateSelectionHint();
  updateTableRoleStats();
}

function updateTableRoleStats() {
  const statsEl = document.getElementById("table-role-stats");
  if (!statsEl) return;
  const game = state.selectedGame;
  if (!game) {
    statsEl.classList.remove("show");
    return;
  }
  statsEl.classList.remove("show");
}

function layoutTablePlayers() {
  if (!DOM.tablePlayers) return;

  // 1. 过滤出需要显示的 ID 列表
  const aiIds = state.selectedIdsOrder.filter(id => state.selectedIds.has(id));
  let humanIds = [];
  if (state.selectedMode === "participate") {
    if (LobbyClient.connected) {
      humanIds = state.lobbyPlayers.map(p => `human_${p.name}`);
    } else {
      if (myAvatarId !== "human") {
        humanIds = ["human"];
      }
    }
  }

  const keys = [...humanIds, ...aiIds.map(String)];
  const keySet = new Set(keys);

  // 2. 清理不在列表中的座位
  Array.from(DOM.tablePlayers.querySelectorAll(".seat")).forEach(el => {
    const key = String(el.dataset.id || "");
    if (!keySet.has(key)) {
      el.classList.add("leave");
      el.addEventListener("transitionend", () => el.remove(), { once: true });
    }
  });

  if (!keys.length) return;
  keys.forEach(id => ensureSeat(id));

  // 3. 计算布局几何参数 (Responsive Geometry)
  const rect = DOM.tablePlayers.getBoundingClientRect();
  const cx = rect.width / 2;
  const cy = rect.height / 2;

  // 检测是否为移动端 (断点 768px)
  const isMobile = window.innerWidth <= 768;

  // 移动端使用更紧凑的缩放比例 (0.65) 以容纳更多人，桌面端保持 0.85
  const scale = isMobile ? 0.65 : 0.85;
  const seatBaseSize = 70; // CSS 中定义的 .seat 宽度

  let radiusX, radiusY;

  if (isMobile) {
    // [V10 修复] 大幅减小半径，让人物紧贴桌子边缘
    // 之前的计算结果大约是 135px，现在我们强制限制在 95px-105px 左右
    radiusX = Math.min(rect.width * 0.3, 105);
    radiusY = Math.min(rect.height * 0.3, 110);
  } else {
    // Desktop Algo
    radiusX = Math.min(280, Math.max(150, rect.width * 0.35));
    radiusY = Math.min(125, Math.max(90, rect.height * 0.35));
  }

  const positions = polarPositions(keys.length, radiusX, radiusY);

  // 4. 应用位置
  keys.forEach((id, i) => {
    const el = DOM.tablePlayers.querySelector(`.seat[data-id="${id}"]`);
    if (!el) return;

    // 设置中心点坐标 (修正：减去 seatBaseSize / 2 以居中)
    el.style.left = `${cx + positions[i].x - seatBaseSize / 2}px`;
    el.style.top = `${cy + positions[i].y - seatBaseSize / 2}px`;

    // [Fix] 组合 scale 和 rotate，确保移动端头像变小且有些许随机旋转感
    const rot = (i % 2 ? 1 : -1) * 2;
    el.style.transform = `scale(${scale}) rotate(${rot}deg)`;
    el.style.zIndex = "1";

    // 绑定点击事件 (保持原有逻辑)
    if (!String(id).startsWith("human") && !el.dataset.hasEvents) {
      el.dataset.hasEvents = "true";
      el.addEventListener("click", (e) => {
        if (state.selectedMode === "participate" && LobbyClient.connected && !LobbyClient.isHost) {
          return;
        }
        if (e.target.closest(".seat-label") || e.target.closest("select")) return;
        e.stopPropagation();
        e.preventDefault();
        const portraitId = parseInt(id, 10);
        if (!isNaN(portraitId)) {
          toggleAgent({ id: portraitId }, null);
        }
      });
    }
  });

  updateTableHeadPreview();
}

function updateCounter() {
  if (!DOM.counterEl) return;

  const game = state.selectedGame;
  const mode = state.selectedMode;

  const aiCount = state.selectedIds.size;
  const humanCount = LobbyClient.connected ? state.lobbyPlayers.length : (mode === 'participate' ? 1 : 0);
  const total = aiCount + humanCount;

  let required = 0;
  if (game === 'avalon') {
    const numPlayersEl = document.getElementById("avalon-num-players");
    const numPlayers = numPlayersEl ? parseInt(numPlayersEl.value, 10) : 5;
    required = numPlayers;
  }

  DOM.counterEl.textContent = `${total}/${required}`;

  const startBtn = document.getElementById("start-btn");
  if (startBtn && LobbyClient.connected) {
    if (LobbyClient.isHost) {
      if (total === required) {
        startBtn.disabled = false;
        startBtn.textContent = `Start Game (${total}/${required})`;
        startBtn.style.opacity = "1";
        startBtn.style.cursor = "pointer";
        startBtn.classList.add("highlight");
      } else {
        startBtn.disabled = true;
        const diff = required - total;
        const msg = diff > 0 ? `${diff} more...` : "Too many";
        startBtn.textContent = `Wait (${msg})`;
        startBtn.style.opacity = "0.6";
        startBtn.style.cursor = "not-allowed";
        startBtn.classList.remove("highlight");
      }
    } else {
      startBtn.disabled = true;
      startBtn.textContent = "Waiting for Host...";
      startBtn.style.opacity = "0.6";
      startBtn.classList.remove("highlight");
    }
  }
  updateSelectionHint();
  updateTableRoleStats();
}

function checkConfigError() {
  return null;
}

function updateSelectionHint() {
  const game = state.selectedGame;
  const hintPill = document.getElementById('selection-hint-pill');
  const hintEl = document.getElementById('selection-hint');

  if (!hintPill || !hintEl) return;

  let hint = '';
  let showHint = false;
  let required = 0;

  const aiCount = state.selectedIds.size;
  const mode = state.selectedMode;
  const humanCount = LobbyClient.connected ? state.lobbyPlayers.length : (mode === 'participate' ? 1 : 0);
  const total = aiCount + humanCount;

  if (game === 'avalon') {
    const numPlayersEl = document.getElementById("avalon-num-players");
    const numPlayers = numPlayersEl ? parseInt(numPlayersEl.value, 10) : 5;
    required = numPlayers;
    showHint = true;
  }

  if (showHint) {
    if (total < required) {
      hint = `${required - total} more`;
      hintPill.style.borderColor = '#ffdd57';
    } else if (total === required) {
      hint = `✓ Correct`;
      hintPill.style.borderColor = '#51f6a5';
    } else {
      hint = `⚠ Exceed ${total - required}`;
      hintPill.style.borderColor = '#ff6b6b';
    }

    hintEl.textContent = hint;
    hintPill.style.display = 'inline-flex';
  } else {
    hintPill.style.display = 'none';
  }
}

function updatePortraitCardActiveState(portraitId, isActive) {
  if (!DOM.strip) return;
  const card = DOM.strip.querySelector(`.portrait-card[data-id="${portraitId}"]`);
  if (card) {
    if (isActive) {
      card.classList.add("active");
    } else {
      card.classList.remove("active");
    }
  }
}

function toggleAgent(person, card) {
  if (state.selectedMode === "participate" && LobbyClient.connected && !LobbyClient.isHost) {
    addStatusMessage("Only Host can manage bots.");
    return;
  }

  const existed = state.selectedIds.has(person.id);

  if (existed) {
    state.selectedIds.delete(person.id);
    const idx = state.selectedIdsOrder.indexOf(person.id);
    if (idx !== -1) {
      state.selectedIdsOrder.splice(idx, 1);
    }
    if (card) {
      card.classList.remove("active");
    } else {
      updatePortraitCardActiveState(person.id, false);
    }

    if (LobbyClient.connected && LobbyClient.isHost) {
      LobbyClient.send({
        type: "SYNC_AI",
        ai_ids: Array.from(state.selectedIds).map(id => parseInt(id, 10))
      });
    }

    updateCounter();
    layoutTablePlayers();
    addStatusMessage(`${person.name} left the team!`);
    return;
  }

  state.selectedIds.add(person.id);
  if (!state.selectedIdsOrder.includes(person.id)) {
    state.selectedIdsOrder.push(person.id);
  }
  if (card) {
    card.classList.add("active");
  } else {
    updatePortraitCardActiveState(person.id, true);
  }

  if (LobbyClient.connected && LobbyClient.isHost) {
    LobbyClient.send({
      type: "SYNC_AI",
      ai_ids: Array.from(state.selectedIds).map(id => parseInt(id, 10))
    });
  }

  updateCounter();
  layoutTablePlayers();
  addStatusMessage(`${person.name} joined the team!`);
}

function renderPortraits() {
  if (!DOM.portraitsGrid) return;
  DOM.portraitsGrid.innerHTML = "";
  const AgentConfigs = loadAgentConfigs();
  const portraits = Array.from({ length: CONFIG.portraitCount }).map((_, i) => {
    const id = i + 1;
    const cfg = AgentConfigs[id] || {};
    return {
      id,
      name: cfg.name || `Agent ${id}`,
      src: `${CONFIG.portraitsBase}portrait_${id}.png`,
      base_model: cfg.base_model || "",
    };
  });
  portraits.forEach(p => {
    const card = document.createElement("div");
    card.className = "portrait-card";
    if (state.selectedIds.has(p.id)) {
      card.classList.add("active");
    }
    card.dataset.id = String(p.id);
    const modelLabel = p.base_model
      ? `<div class="portrait-model">${p.base_model}</div>`
      : "";
    card.innerHTML = `
      ${modelLabel}
      <img src="${p.src}" alt="${p.name}">
      <div class="portrait-name">${p.name}</div>
    `;
    card.addEventListener("click", () => toggleAgent(p, card));
    DOM.portraitsGrid.appendChild(card);
  });
}

function focusGame(game) {
  if (!DOM.gameCards) return;
  DOM.gameCards.forEach(c => c.classList.toggle("active", c.dataset.game === game));
}

function setGame(game) {
  state.selectedGame = game || "";
  focusGame(state.selectedGame);

  if (!state.selectedGame) {
    if (DOM.avalonFields) DOM.avalonFields.classList.remove("show");
    if (DOM.diplomacyFields) DOM.diplomacyFields.classList.remove("show");
    updateCounter();
    updateTableRoleStats();
    const tablePreview = document.getElementById("table-preview");
    if (tablePreview) {
      tablePreview.classList.remove("has-game");
    }
    return;
  }

  addStatusMessage(`Selected game: ${state.selectedGame}`);

  if (state.selectedGame === "diplomacy") {
    fetchDiplomacyOptions().then(() => updateCounter());
  } else if (state.selectedGame === "avalon") {
    fetchAvalonOptions().then(() => updateCounter());
  }

  updateConfigVisibility();
  updateSelectionHint();
  updateTableHeadPreview();
  updateTableRoleStats();

  const tablePreview = document.getElementById("table-preview");
  if (tablePreview) {
    if (state.selectedGame) {
      tablePreview.classList.add("has-game");
      setTimeout(() => {
        layoutTablePlayers();
      }, 0);
    } else {
      tablePreview.classList.remove("has-game");
    }
  }
}

function setMode(mode) {
  state.selectedMode = mode || "observe";

  if (DOM.modeLabelEl) {
    DOM.modeLabelEl.textContent = state.selectedMode === "observe" ? "Observer" : "Participate";
  }

  if (DOM.modeToggle) {
    DOM.modeToggle.querySelectorAll(".mode-opt").forEach(opt => {
      opt.classList.toggle("active", opt.dataset.mode === state.selectedMode);
    });
  }

  updateConfigVisibility();
  updateSelectionHint();
  layoutTablePlayers();
  updateTableHeadPreview();
  updateTableRoleStats();
}

function updateConfigVisibility() {
  const game = state.selectedGame;
  const mode = state.selectedMode;

  if (DOM.avalonFields) {
    DOM.avalonFields.classList.toggle("show", game === "avalon" && !!mode);
  }
  if (DOM.diplomacyFields) {
    DOM.diplomacyFields.classList.toggle("show", game === "diplomacy" && !!mode);
  }

  if (DOM.connectionUI) {
    if (game === "avalon" && mode === "participate") {
      DOM.connectionUI.classList.add("active");
    } else {
      DOM.connectionUI.classList.remove("active");
    }
  }

  document.querySelectorAll(".avalon-participate-only").forEach(el => {
    el.style.display = (game === "avalon" && mode === "participate") ? "flex" : "none";
  });
  document.querySelectorAll(".diplomacy-participate-only").forEach(el => {
    el.style.display = (game === "diplomacy" && mode === "participate") ? "flex" : "none";
  });
  if (DOM.powerModelsSection) {
    DOM.powerModelsSection.style.display = (game === "diplomacy" && state.diplomacyOptions) ? "block" : "none";
  }
}

function shuffleInPlace(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// function buildMultiplayerPayload(game) {
//     const numPlayersEl = document.getElementById("avalon-num-players");
//     const languageEl = document.getElementById("avalon-language");

//     const aiIds = state.selectedIdsOrder.filter(id => state.selectedIds.has(id));
//     const agentConfigs = loadAgentConfigs();
//     const aiConfigPayload = {};
//     aiIds.forEach(id => {
//         if (agentConfigs[id]) aiConfigPayload[id] = agentConfigs[id];
//     });

//     return {
//         type: "START_GAME",
//         game_config: {
//             game: game,
//             num_players: numPlayersEl ? parseInt(numPlayersEl.value, 10) : 5,
//             language: languageEl ? languageEl.value : "en",
//             ai_ids: aiIds,
//             agent_configs: aiConfigPayload
//         }
//     };
// }
function buildMultiplayerPayload(game) {
  const numPlayersEl = document.getElementById("avalon-num-players");
  const languageEl = document.getElementById("avalon-language");

  // 1. 获取 AI 的 ID 列表
  const aiIds = state.selectedIdsOrder.filter(id => state.selectedIds.has(id));

  // 2. 获取 AI 配置
  const agentConfigs = loadAgentConfigs();
  const aiConfigPayload = {};
  aiIds.forEach(id => {
    if (agentConfigs[id]) aiConfigPayload[id] = agentConfigs[id];
  });

  // 3. [核心修复] 构建完整的头像列表
  // 列表结构必须是: [房主头像ID, AI_1_ID, AI_2_ID, ...]
  // 这里的 myAvatarId 就是你在大厅选的 "human" 或 "h1" 等
  const fullPortraitList = [myAvatarId, ...aiIds];

  console.log("正在发送多人游戏配置, 头像列表:", fullPortraitList); // 方便调试

  return {
    type: "START_GAME",
    game_config: {
      game: game,
      num_players: numPlayersEl ? parseInt(numPlayersEl.value, 10) : 5,
      language: languageEl ? languageEl.value : "en",
      ai_ids: aiIds,
      agent_configs: aiConfigPayload,

      // [必须加上这一行] 把完整的头像列表传给 Server
      selected_portrait_ids: fullPortraitList
    }
  };
}

function closeModeDropdown() {
  if (DOM.modeToggle) {
    DOM.modeToggle.classList.remove("open");
  }
}

// [NEW] Avatar Selection Logic
function initAvatarSelection() {
  const preview = document.getElementById("my-avatar-preview");
  const modal = document.getElementById("avatar-modal");
  const closeBtn = document.getElementById("close-avatar-modal");
  const grid = document.getElementById("avatar-selection-grid");

  if (!preview || !modal || !grid) return;

  preview.addEventListener("click", () => {
    if (DOM.connectBtn.disabled) return;
    modal.classList.add("show");
  });

  closeBtn.addEventListener("click", () => modal.classList.remove("show"));

  // 1. Human default
  const humanOpt = document.createElement("div");
  humanOpt.className = "avatar-option selected";
  humanOpt.dataset.id = "human";
  humanOpt.innerHTML = `<img src="${CONFIG.portraitsBase}portrait_human.png">`;
  humanOpt.addEventListener("click", () => selectAvatar("human", humanOpt));
  grid.appendChild(humanOpt);

  // 2. Custom Humans (h1...hN)
  console.log("正在渲染头像数量:", CONFIG.humanAvatarCount); // <--- 加入这一行测试
  if (CONFIG.humanAvatarCount) {
    for (let i = 1; i <= CONFIG.humanAvatarCount; i++) {
      const opt = document.createElement("div");
      opt.className = "avatar-option";
      opt.dataset.id = `h${i}`;
      opt.innerHTML = `<img src="${CONFIG.portraitsBase}portrait_h${i}.png">`;
      opt.addEventListener("click", () => selectAvatar(`h${i}`, opt));
      grid.appendChild(opt);
    }
  }

  function selectAvatar(id, el) {
    myAvatarId = id;
    const src = id === "human" ? "portrait_human.png" : `portrait_${id}.png`;
    preview.querySelector("img").src = CONFIG.portraitsBase + src;
    grid.querySelectorAll(".avatar-option").forEach(o => o.classList.remove("selected"));
    el.classList.add("selected");
    modal.classList.remove("show");
  }
}

function initEventListeners() {
  if (DOM.gameCards) {
    DOM.gameCards.forEach(card => {
      card.addEventListener("click", () => {
        setGame(card.dataset.game);
      });
    });
  }

  if (DOM.connectBtn) {
    DOM.connectBtn.addEventListener("click", () => {
      const nick = DOM.nicknameInput.value.trim();
      if (!nick) {
        addStatusMessage("Please enter a nickname.");
        return;
      }
      LobbyClient.connect(nick);
    });
  }

  if (DOM.modeToggle) {
    const pill = DOM.modeToggle.querySelector(".pill-mode");
    if (pill) {
      pill.addEventListener("click", (e) => {
        e.stopPropagation();
        DOM.modeToggle.classList.toggle("open");
      });
    }
    DOM.modeToggle.querySelectorAll(".mode-opt").forEach(opt => {
      opt.addEventListener("click", (e) => {
        e.stopPropagation();
        setMode(opt.dataset.mode);
        closeModeDropdown();
        addStatusMessage(`Switched to ${opt.dataset.mode === "observe" ? "observe" : "participate"} mode`);
      });
    });
  }
  document.addEventListener("click", () => closeModeDropdown());

  const avalonNumPlayers = document.getElementById("avalon-num-players");
  if (avalonNumPlayers) {
    avalonNumPlayers.addEventListener("change", function () {
      updateCounter();
      updateTableHeadPreview();
    });
  }

  if (DOM.avalonRerollRolesBtn) {
    DOM.avalonRerollRolesBtn.addEventListener("click", (e) => {
      e.preventDefault();
      if (state.selectedMode === "participate" && LobbyClient.connected && !LobbyClient.isHost) return;
      // state.avalonRoleOrder = avalonAssignRolesFor5();
      state.avalonRoleOrder = avalonAssignRoles();
      updateTableHeadPreview();
    });
  }

  if (DOM.randomSelectBtn) {
    DOM.randomSelectBtn.addEventListener("click", (e) => {
      e.preventDefault();

      if (state.selectedMode === "participate" && LobbyClient.connected && !LobbyClient.isHost) {
        addStatusMessage("Only Host can select bots.");
        return;
      }

      const game = state.selectedGame;
      const numPlayersEl = document.getElementById("avalon-num-players");
      const totalRequired = numPlayersEl ? parseInt(numPlayersEl.value, 10) : 5;

      const humanCount = LobbyClient.connected ? state.lobbyPlayers.length : (state.selectedMode === "participate" ? 1 : 0);
      const aiNeeded = totalRequired - humanCount;

      if (aiNeeded <= 0) {
        addStatusMessage("Lobby is full (Humans). No AI needed.");
        return;
      }

      state.selectedIds.clear();
      state.selectedIdsOrder = [];
      const allIds = Array.from({ length: CONFIG.portraitCount }, (_, i) => i + 1);
      const shuffled = allIds.slice();
      shuffleInPlace(shuffled);
      const selected = shuffled.slice(0, aiNeeded);
      selected.forEach(id => {
        state.selectedIds.add(id);
        state.selectedIdsOrder.push(id);
      });

      if (LobbyClient.connected && LobbyClient.isHost) {
        LobbyClient.send({
          type: "SYNC_AI",
          ai_ids: Array.from(state.selectedIds).map(id => parseInt(id, 10))
        });
      }

      renderPortraits();
      updateCounter();
      layoutTablePlayers();
      updateTableHeadPreview();

      addStatusMessage(`Randomly selected ${aiNeeded} AI agents.`);
    });
  }

  if (DOM.startBtn) {
    DOM.startBtn.addEventListener("click", async () => {
      const game = state.selectedGame;
      const mode = state.selectedMode;

      if (game === 'avalon' && mode === 'participate' && LobbyClient.connected) {
        if (!LobbyClient.isHost) return;

        const numPlayersEl = document.getElementById("avalon-num-players");
        const required = numPlayersEl ? parseInt(numPlayersEl.value, 10) : 5;
        const currentTotal = state.lobbyPlayers.length + state.selectedIds.size;

        if (currentTotal !== required) {
          addStatusMessage(`Need exactly ${required} players. Current: ${currentTotal}`);
          return;
        }

        addStatusMessage("Sending Start Request...");
        const payload = buildMultiplayerPayload(game);
        LobbyClient.send(payload);
        DOM.startBtn.disabled = true;
        return;
      }

      if (game === "avalon" && mode === "participate") {
        alert("Please Connect to Lobby to play Avalon Multiplayer.");
        return;
      }

      const payload = buildPayload(game, mode);
      const ids = state.selectedIdsOrder.filter(id => state.selectedIds.has(id));
      sessionStorage.setItem("gameConfig", JSON.stringify(payload));
      sessionStorage.setItem("selectedPortraits", JSON.stringify(ids));
      sessionStorage.setItem("gameLanguage", payload.language || "en");

      setTimeout(() => {
        const url = computeRedirectUrl(game, mode);
        window.location.href = url;
      }, CONFIG.travelDuration + 300);
    });
  }

  window.addEventListener("resize", () => layoutTablePlayers());
}

function initDOM() {
  DOM = {
    strip: document.getElementById("portraits-strip"),
    portraitsGrid: document.getElementById("portraits-grid"),
    tablePlayers: document.getElementById("table-players"),
    statusLog: document.getElementById("status-log"),
    counterEl: document.getElementById("counter"),
    modeLabelEl: document.getElementById("mode-label"),
    avalonFields: document.getElementById("avalon-fields"),
    diplomacyFields: document.getElementById("diplomacy-fields"),
    startBtn: document.getElementById("start-btn"),
    powerModelsSection: document.getElementById("power-models-section"),
    powerModelsGrid: document.getElementById("power-models-grid"),
    gameCards: Array.from(document.querySelectorAll(".game-card")),
    modeToggle: document.querySelector(".mode-toggle"),
    selectionHintPill: document.getElementById("selection-hint-pill"),
    selectionHint: document.getElementById("selection-hint"),
    avalonRerollRolesBtn: document.getElementById("avalon-reroll-roles"),
    diplomacyShufflePowersBtn: document.getElementById("diplomacy-shuffle-powers"),
    randomSelectBtn: document.getElementById("random-select-btn"),
    connectionUI: document.getElementById("connection-ui"),
    connectionStatus: document.getElementById("connection-status"),
    nicknameInput: document.getElementById("nickname-input"),
    connectBtn: document.getElementById("connect-btn"),
    // [NEW] Avatar Preview Element
    avatarPreview: document.getElementById("my-avatar-preview"),
  };
}

let lastConfigUpdateTime = localStorage.getItem(STORAGE_KEYS.CONFIG_UPDATE_TIME) || "0";

async function init() {
  initDOM();

  try {
    await loadWebConfig();
  } catch (e) {
    console.warn("Config load failed, continuing...");
  }

  // Safe init avatar selection
  if (typeof initAvatarSelection === 'function') {
    try { initAvatarSelection(); } catch (e) { console.error(e); }
  }

  renderPortraits();
  updateCounter();
  layoutTablePlayers();
  setMode("observe");
  updateConfigVisibility();
  updateTableHeadPreview();
  initEventListeners();

  let lastFocusTime = Date.now();
  window.addEventListener("focus", () => {
    const now = Date.now();
    if (now - lastFocusTime < 500) return;
    lastFocusTime = now;
    const currentUpdateTime = localStorage.getItem(STORAGE_KEYS.CONFIG_UPDATE_TIME) || "0";
    if (currentUpdateTime !== lastConfigUpdateTime) {
      lastConfigUpdateTime = currentUpdateTime;
      renderPortraits();
    }
  });
  window.addEventListener("storage", (e) => {
    if (e.key === STORAGE_KEYS.AGENT_CONFIGS) {
      renderPortraits();
    }
  });
  window.addEventListener('localStorageChange', () => {
    renderPortraits();
  });
  addStatusMessage("Welcome to Agent Arena!");
  addStatusMessage("Select Avalon -> Participate to join Lobby.");
  initStepHints();
  blinkStepHints();
}

function initStepHints() {
  const stepHints = document.querySelectorAll(".step-hint");
  stepHints.forEach(hint => {
    const target = hint.dataset.target;
    if (!target) return;
    hint.addEventListener("mouseenter", () => {
      let targetEl = null;
      if (target === "games") targetEl = document.getElementById("games");
      else if (target === "agents") targetEl = document.getElementById("portraits-strip");
      else if (target === "scene") targetEl = document.getElementById("scene");
      else if (target === "start-btn") targetEl = document.getElementById("start-btn");
      if (targetEl) targetEl.classList.add("highlight");
    });
    hint.addEventListener("mouseleave", () => {
      let targetEl = null;
      if (target === "games") targetEl = document.getElementById("games");
      else if (target === "agents") targetEl = document.getElementById("portraits-strip");
      else if (target === "scene") targetEl = document.getElementById("scene");
      else if (target === "start-btn") targetEl = document.getElementById("start-btn");
      if (targetEl) targetEl.classList.remove("highlight");
    });
  });
}

function blinkStepHints() {
  const stepHints = document.querySelectorAll(".step-hint");
  stepHints.forEach(hint => {
    hint.classList.add("initial-blink");
    hint.addEventListener("animationend", () => {
      hint.classList.remove("initial-blink");
    }, { once: true });
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}