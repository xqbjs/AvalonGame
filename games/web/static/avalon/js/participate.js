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
  const radiusX = Math.min(300, Math.max(160, rect.width * 0.45));
  const radiusY = Math.min(180, Math.max(100, rect.height * 0.4));
  const positions = polarPositions(count, radiusX, radiusY);

  for (let i = 0; i < count; i++) {
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
    seat.style.left = `${cx + positions[i].x - 34}px`;
    seat.style.top = `${cy + positions[i].y - 34}px`;
    const baseRotation = (i % 2 ? 1 : -1) * 2;
    seat.style.setProperty("--base-rotation", `${baseRotation}deg`);
    seat.style.transform = `rotate(var(--base-rotation, 0deg))`;
    tablePlayers.appendChild(seat);
  }
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

// 7. WEBSOCKET HANDLERS

wsClient.onMessage("message", (message) => {
  addMessage(message);
});

// [NEW] Receive Game Metadata from Server
wsClient.onMessage("game_metadata", (data) => {
  console.log("Received Game Metadata:", data);
  if (data.metadata) {
    gameMetadata = data.metadata;
    // Re-render table with new data
    setupTablePlayers(gameMetadata.num_players || 5);
  }
});

wsClient.onMessage("game_state", (state) => {
  updateGameState(state);

  if (state.status === "running") {
    messagesContainer.style.display = "flex";
    inputContainer.style.display = "flex";
  } else if (state.status === "stopped") {
    hideInputRequest();
    messagesContainer.innerHTML = '<p class="system-msg">Game stopped.</p>';
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
