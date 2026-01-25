// Participate mode JavaScript - Pixel Town Style

window.addEventListener("beforeunload", () => {
  const keysToKeep = [
    "gameConfig",
    "selectedPortraits",
    "gameLanguage",
    "my_player_id",
  ];
  Object.keys(sessionStorage).forEach((key) => {
    if (!keysToKeep.includes(key)) {
      sessionStorage.removeItem(key);
    }
  });
});

window.addEventListener("pageshow", (event) => {
  if (event.persisted) {
    window.location.reload();
  }
});

// 1. IDENTITY
const myPlayerIdRaw = sessionStorage.getItem("my_player_id");
const myPlayerId = myPlayerIdRaw ? parseInt(myPlayerIdRaw, 10) : null;

const gameLanguage = sessionStorage.getItem("gameLanguage") || "en";
document.body.classList.add(`lang-${gameLanguage}`);

// [NEW] Metadata from server
let gameMetadata = null;

// 2. WEBSOCKET SETUP
const wsClient = new WebSocketClient(null, { uid: myPlayerId });

// 3. DOM ELEMENTS
const messagesContainer = document.getElementById("messages-container");
const phaseDisplay = document.getElementById("phase-display");
const missionDisplay = document.getElementById("mission-display");
const roundDisplay = document.getElementById("round-display");
const statusDisplay = document.getElementById("status-display");
const userInputElement = document.getElementById("user-input");
const sendButton = document.getElementById("send-button");
const userInputRequest = document.getElementById("user-input-request");
const inputPrompt = document.getElementById("input-prompt");
const tablePlayers = document.getElementById("table-players");
const inputContainer = document.querySelector(".input-container");
// [NEW] 游戏结束控制区
const gameEndControls = document.getElementById("game-end-controls");
const returnLobbyBtn = document.getElementById("return-lobby-btn");

// ================= [START 新增代码] =================
// 模态框相关元素
const modalOverlay = document.getElementById("modal-overlay");
const hostConfirmModal = document.getElementById("host-confirm-modal");
const gameEndedModal = document.getElementById("game-ended-modal");
const endTitle = document.getElementById("end-title");
const endReason = document.getElementById("end-reason");

// 按钮相关元素
const hostResetBtn = document.getElementById("host-reset-btn");
const confirmResetYes = document.getElementById("confirm-reset-yes");
const confirmResetNo = document.getElementById("confirm-reset-no");
const globalReturnLobbyBtn = document.getElementById("global-return-lobby-btn");
// ================= [END 新增代码] =================

const gameSetup = document.getElementById("game-setup");
if (gameSetup && myPlayerId !== null) gameSetup.style.display = "none";

let messageCount = 0;
let numPlayers = 5;
let waitingForInput = false;

// 4. VISUAL HELPER FUNCTIONS - REWRITTEN FOR SERVER METADATA

function getPortraitSrc(playerId) {
  const validId = parseInt(playerId, 10);

  // Use Server Metadata if available
  if (gameMetadata && gameMetadata.players) {
    const p = gameMetadata.players.find((x) => x.id === validId);
    if (p) {
      // [FIX] Handle human avatars ("human" or "h1" etc)
      if (p.is_human) {
        if (
          p.portrait_id &&
          p.portrait_id !== -1 &&
          p.portrait_id !== "human"
        ) {
          return `/static/portraits/portrait_${p.portrait_id}.png`;
        }
        return `/static/portraits/portrait_human.png`;
      }
      // AI avatars (integers)
      if (p.portrait_id)
        return `/static/portraits/portrait_${p.portrait_id}.png`;
    }
  }

  // Fallback logic
  if (myPlayerId !== null && validId === myPlayerId) {
    return `/static/portraits/portrait_human.png`;
  }

  const pId = (validId % 15) + 1;
  return `/static/portraits/portrait_${pId}.png`;
}

function getModelName(playerId) {
  const validId = parseInt(playerId, 10);

  if (myPlayerId !== null && validId === myPlayerId) {
    return "You";
  }

  // Use Server Metadata
  if (gameMetadata && gameMetadata.players) {
    const p = gameMetadata.players.find((x) => x.id === validId);
    if (p && p.name) return p.name;
  }

  return `Player ${validId}`;
}

function polarPositions(count, radiusX, radiusY) {
  return Array.from({ length: count }).map((_, i) => {
    const angle = (Math.PI * 2 * i) / count - Math.PI / 2;
    return { x: radiusX * Math.cos(angle), y: radiusY * Math.sin(angle) };
  });
}

function setupTablePlayers(count) {
  numPlayers = count;
  tablePlayers.innerHTML = "";

  const rect = tablePlayers.getBoundingClientRect();
  const cx = rect.width / 2;
  const cy = rect.height / 2;

  const isMobile = window.innerWidth <= 768;

  // --- 移动端布局算法 (Grid/Line System) ---
  if (isMobile) {
    // 1. 决定行数和每行人数
    let row1Count, row2Count;

    if (count <= 5) {
      row1Count = count;
      row2Count = 0;
    } else {
      // 6-10人：拆分两行 (例: 7 -> 4+3)
      row1Count = Math.ceil(count / 2);
      row2Count = count - row1Count;
    }

    // [修改点 1] 参数调整：增加间距，微调头像大小以适应屏幕
    // 之前: itemWidth=58, gap=12
    // 现在: itemWidth=54, gap=18 (间距明显变大，头像稍微精致一点)
    const itemWidth = 54;
    const gap = 18;

    // 计算每一行及其起始 X 坐标 (为了居中)
    const getStartX = (c) => {
      const totalWidth = c * itemWidth + (c - 1) * gap;
      return (rect.width - totalWidth) / 2;
    };

    const row1StartX = getStartX(row1Count);
    const row2StartX = getStartX(row2Count);

    // 垂直位置
    const row1Y = row2Count === 0 ? rect.height * 0.5 : rect.height * 0.28;
    const row2Y = rect.height * 0.72;

    for (let i = 0; i < count; i++) {
      const seat = createSeatElement(i);

      let targetX, targetY;

      if (i < row1Count) {
        // 第一行
        targetX = row1StartX + i * (itemWidth + gap);
        targetY = row1Y;
      } else {
        // 第二行
        const rowIndex = i - row1Count;
        targetX = row2StartX + rowIndex * (itemWidth + gap);
        targetY = row2Y;
      }

      // [修改点 2] 修正 seat 居中偏移量 (54的一半是27)
      seat.style.left = `${targetX}px`;
      seat.style.top = `${targetY - 27}px`;

      // [修改点 3] 必须强制设置 scale(1)，否则可能会继承旧逻辑
      seat.style.transform = `scale(1)`;
      seat.style.setProperty("--base-rotation", `0deg`);

      // [修改点 4] 配合 CSS 修改，手动指定移动端大小
      seat.style.width = `${itemWidth}px`;
      seat.style.height = `${itemWidth}px`;

      tablePlayers.appendChild(seat);
    }

  }
  // --- 桌面端布局算法 (保持不变) ---
  else {
    let radiusX = Math.min(300, Math.max(160, rect.width * 0.45));
    let radiusY = Math.min(180, Math.max(100, rect.height * 0.4));
    const positions = polarPositions(count, radiusX, radiusY);

    for (let i = 0; i < count; i++) {
      const seat = createSeatElement(i);
      seat.style.left = `${cx + positions[i].x - 34}px`;
      seat.style.top = `${cy + positions[i].y - 34}px`;
      const baseRotation = (i % 2 ? 1 : -1) * 2;
      seat.style.setProperty("--base-rotation", `${baseRotation}deg`);
      seat.style.transform = `rotate(${baseRotation}deg)`;
      tablePlayers.appendChild(seat);
    }
  }
}

// 辅助函数：提取生成 DOM 的通用逻辑，减少重复代码
function createSeatElement(i) {
  const seat = document.createElement("div");
  seat.className = "seat";
  seat.dataset.playerId = String(i);

  const isMe = myPlayerId !== null && i === myPlayerId;
  if (isMe) seat.classList.add("is-me");

  const portraitSrc = getPortraitSrc(i);
  const modelName = getModelName(i);

  seat.innerHTML = `
        <div class="seat-label"></div>
        <span class="id-tag">${isMe ? "YOU" : "P" + i}</span>
        <img src="${portraitSrc}" alt="Player ${i}">
        <span class="name-tag">${modelName}</span>
        <div class="speech-bubble">💬</div>
    `;
  return seat;
}
function highlightSpeaker(playerId) {
  document.querySelectorAll(".seat").forEach((seat) => {
    const seatPlayerId = seat.dataset.playerId;
    const isSpeaking = seatPlayerId === String(playerId);

    const bubble = seat.querySelector(".speech-bubble");

    if (isSpeaking) {
      seat.classList.add("speaking");
      if (bubble) {
        bubble.style.animation = "none";
        bubble.offsetHeight;
        bubble.style.animation = "bubble-pop 2s ease-out forwards";
        bubble.style.opacity = "1";
      }
    } else {
      seat.classList.remove("speaking");
      if (bubble) bubble.style.opacity = "0";
    }
  });
}

function formatTime(timestamp) {
  if (!timestamp) return "";
  const date = new Date(timestamp);
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function addMessage(message) {
  messageCount++;
  if (messageCount === 1) messagesContainer.innerHTML = "";

  const messageDiv = document.createElement("div");
  messageDiv.className = "chat-message";

  let senderType = "system";
  let avatarHtml = '<div class="chat-avatar system">🎭</div>';
  let senderName = message.sender || "System";
  let playerId = null;

  if (message.sender === "Moderator") {
    senderType = "moderator";
    avatarHtml = '<div class="chat-avatar system">⚔</div>';
  } else if (message.sender && message.sender.startsWith("Player")) {
    const match = message.sender.match(/Player\s*(\d+)/);
    if (match) {
      playerId = parseInt(match[1], 10);
      senderType = playerId === myPlayerId ? "user" : "agent";

      const portraitSrc = getPortraitSrc(playerId);
      avatarHtml = `<div class="chat-avatar"><img src="${portraitSrc}" alt="${senderName}"></div>`;

      if (playerId === myPlayerId) messageDiv.classList.add("own");

      highlightSpeaker(playerId);
    }
  }

  messageDiv.innerHTML = `
        ${avatarHtml}
        <div class="chat-bubble">
            <div class="chat-header">
                <span class="chat-sender ${senderType}">${escapeHtml(senderName)}</span>
                <span class="chat-time">${formatTime(message.timestamp)}</span>
            </div>
            <div class="chat-content">${escapeHtml(message.content || "")}</div>
        </div>
    `;

  messagesContainer.appendChild(messageDiv);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function updateRoleLabels(roleData) {
  if (!roleData) return;

  let roles = [];
  if (Array.isArray(roleData)) {
    roles = roleData;
  } else if (roleData.role_names) {
    roles = roleData.role_names;
  }

  roles.forEach((roleInfo, index) => {
    let playerId = index;
    let roleName = "";

    if (typeof roleInfo === "string") {
      roleName = roleInfo;
    } else if (Array.isArray(roleInfo)) {
      if (typeof roleInfo[0] === "number") playerId = roleInfo[0];
      if (typeof roleInfo[1] === "string") roleName = roleInfo[1];
    }

    const seat = tablePlayers.querySelector(
      `.seat[data-player-id="${playerId}"]`,
    );
    if (seat && roleName) {
      const label = seat.querySelector(".seat-label");
      if (label) {
        label.textContent = roleName;
        seat.classList.add("has-label");
      }
    }
  });
}

// 5. GAME LOGIC & INPUT

function updateGameState(state) {
  if (phaseDisplay) {
    const phases = [
      "Team Selection",
      "Team Voting",
      "Quest Voting",
      "Assassination",
    ];
    const phaseName =
      state.phase !== null && state.phase !== undefined
        ? phases[state.phase] || "Unknown"
        : "-";
    phaseDisplay.textContent = `Phase: ${phaseName}`;
  }
  if (missionDisplay)
    missionDisplay.textContent = `Mission: ${state.mission_id ?? "-"}`;
  if (roundDisplay)
    roundDisplay.textContent = `Round: ${state.round_id ?? "-"}`;
  if (statusDisplay)
    statusDisplay.textContent = `Status: ${state.status ?? "Waiting"}`;

  // [CRITICAL] Ensure table exists
  if (
    state.num_players &&
    (state.num_players !== numPlayers || tablePlayers.children.length === 0)
  ) {
    setupTablePlayers(state.num_players);
  }

  // [CRITICAL] Update roles if available
  if (state.role_names) {
    updateRoleLabels(state.role_names);
  } else if (state.roles) {
    updateRoleLabels(state.roles);
  }
}

function showInputRequest(agentId, prompt) {
  console.log(
    `[System] Input Request for Agent ${agentId} (Me: ${myPlayerId})`,
  );

  // 1. 安全检查
  if (myPlayerId === null) return;

  // 2. [关键修复] 如果请求的目标 ID 不是我，直接忽略！
  // 绝对不要在这里调用 hideInputRequest()，否则会把刚刚打开的窗口关掉。
  if (parseInt(agentId) !== myPlayerId) {
    return;
  }

  // 3. 只有确认是发给我的，才执行下面的开启逻辑
  waitingForInput = true;
  inputPrompt.textContent = prompt;
  userInputRequest.style.display = "block";
  userInputElement.disabled = false;
  sendButton.disabled = false;
  userInputElement.focus();

  highlightSpeaker(myPlayerId);
}

function hideInputRequest() {
  waitingForInput = false;
  userInputRequest.style.display = "none";
  userInputElement.disabled = true;
  sendButton.disabled = true;
  userInputElement.value = "";
}

function sendUserInput() {
  const content = userInputElement.value.trim();
  if (!content) return;

  wsClient.sendUserInput(myPlayerId, content);
  hideInputRequest();
}

// 6. EVENT LISTENERS

sendButton.addEventListener("click", sendUserInput);
userInputElement.addEventListener("keypress", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendUserInput();
  }
});
// ... (原代码：userInputElement.addEventListener("keypress", ...))

// ================= [START 新增代码] =================

// --- 模态框辅助函数 ---
function showModal(modalType) {
  if (!modalOverlay) return; // 防御性检查
  modalOverlay.style.display = "flex";
  if (modalType === "host_confirm") {
    if (hostConfirmModal) hostConfirmModal.style.display = "block";
    if (gameEndedModal) gameEndedModal.style.display = "none";
  } else if (modalType === "game_ended") {
    if (hostConfirmModal) hostConfirmModal.style.display = "none";
    if (gameEndedModal) gameEndedModal.style.display = "block";
  }
}

function hideModal() {
  if (modalOverlay) modalOverlay.style.display = "none";
  if (hostConfirmModal) hostConfirmModal.style.display = "none";
  if (gameEndedModal) gameEndedModal.style.display = "none";
}

// --- 房主权限与事件绑定 ---

// 1. 只有房主 (ID 0) 显示桌面中间的重置按钮
if (myPlayerId === 0 && hostResetBtn) {
  hostResetBtn.style.display = "block";

  // 点击重置按钮 -> 显示确认框
  hostResetBtn.addEventListener("click", () => {
    showModal("host_confirm");
  });
}

// 2. 确认重置 (YES) -> 调用后端 API
if (confirmResetYes) {
  confirmResetYes.addEventListener("click", () => {
    // 调用我们刚刚写的 stop-game 接口
    fetch("/api/stop-game", { method: "POST" })
      .then(res => res.json())
      .then(data => {
        console.log("Host triggered stop:", data);
        //hideModal();
        // 注意：这里不需要手动跳转，等待后端广播 GAME_FORCE_STOPPED
      })
      .catch(err => console.error("API Error:", err));
  });
}

// 3. 取消重置 (NO) -> 关闭模态框
if (confirmResetNo) {
  confirmResetNo.addEventListener("click", hideModal);
}

// 4. 游戏结束模态框里的“返回大厅”按钮
if (globalReturnLobbyBtn) {
  globalReturnLobbyBtn.addEventListener("click", () => {
    window.location.href = "/"; // 强制回首页
  });
}
// ================= [END 新增代码] =================

// (原代码：if (returnLobbyBtn) { ... }) 
// 原有的 returnLobbyBtn 逻辑可以保留作为备用，或者你可以选择注释掉它，防止UI重复

// 7. WEBSOCKET HANDLERS

wsClient.onMessage("message", (message) => {
  addMessage(message);
});
// [NEW] 接收历史聊天记录
// 注意：如果 server.py 按照“第一步”修改为逐条发送，这个监听器实际上不会被触发，
// 但保留它可以作为双重保险。
wsClient.onMessage("chat_history", (data) => {
  // [FIX] 增加安全检查，防止 data.history 为 undefined 时报错
  if (data && data.history && Array.isArray(data.history)) {
    console.log("Restoring chat history via list...", data.history.length);
    data.history.forEach(msg => {
      addMessage(msg);
    });

    if (messagesContainer) {
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
  } else {
    console.warn("Received empty or invalid chat history packet");
  }
});

// ================= [START 新增代码] =================
// 监听房主强制结束信号
wsClient.onMessage("GAME_FORCE_STOPPED", (data) => {
  console.warn("Game Force Stopped:", data);
  if (endTitle) endTitle.textContent = "Room Dissolved";
  if (endReason) endReason.textContent = "The Host has terminated the session.";
  // [新增] 强制关闭所有玩家的输入请求窗口
  hideInputRequest();

  showModal("game_ended");

  // 隐藏房主按钮，防止重复点击
  if (hostResetBtn) hostResetBtn.style.display = "none";
});
// ================= [END 新增代码] =================

// ================= [START 修改代码：替换原有的 game_state 监听] =================
wsClient.onMessage("game_state", (state) => {
  updateGameState(state);

  // 1. 游戏进行中
  if (state.status === "running") {
    if (messagesContainer) messagesContainer.style.display = "flex";
    if (inputContainer) inputContainer.style.display = "flex";

    // 隐藏旧的结束控制区
    if (gameEndControls) gameEndControls.style.display = "none";
  }
  // 2. 游戏停止 (通常会被 GAME_FORCE_STOPPED 接管，这里做兜底)
  else if (state.status === "stopped") {
    hideInputRequest();
    if (messagesContainer) messagesContainer.innerHTML += '<div class="system-msg">Game stopped.</div>';
  }
  // 3. 游戏正常结束 -> 弹出模态框
  else if (state.status === "finished") {
    hideInputRequest();

    // 设置弹窗内容
    if (endTitle) endTitle.textContent = "Game Finished";

    let reasonText = "The game has ended.";
    if (state.good_wins === true) reasonText = "🏆 Good Wins!";
    else if (state.good_wins === false) reasonText = "😈 Evil Wins!";
    else if (state.result) reasonText = `Result: ${state.result}`;

    if (endReason) endReason.textContent = reasonText;

    // 弹出窗口
    showModal("game_ended");

    // 隐藏桌面的房主按钮
    if (hostResetBtn) hostResetBtn.style.display = "none";
  }
});
// ================= [END 修改代码] =================

// ... (原代码：wsClient.onMessage("user_input_request", ...))

// [NEW] Receive Game Metadata from Server
wsClient.onMessage("game_metadata", (data) => {
  console.log("Received Game Metadata:", data);
  if (data.metadata) {
    gameMetadata = data.metadata;
    // Re-render table with new data
    setupTablePlayers(gameMetadata.num_players || 5);
  }
});


wsClient.onMessage("user_input_request", (request) => {
  showInputRequest(request.agent_id, request.prompt);
});

wsClient.onMessage("error", (error) => {
  console.error("Error from server:", error);
  addMessage({
    sender: "System",
    content: `Error: ${error.message || "Unknown error"}`,
    timestamp: new Date().toISOString(),
  });
});

// 8. INITIALIZATION
wsClient.onConnect(() => {
  console.log(`Connected as Player ${myPlayerId}`);
});

wsClient.onDisconnect(() => {
  console.log("Disconnected");
  hideInputRequest();
});

if (myPlayerId !== null) {
  wsClient.connect();
} else {
  console.warn("No MyPlayerID found.");
}

window.addEventListener("resize", () => {
  if (numPlayers) setupTablePlayers(numPlayers);
});
