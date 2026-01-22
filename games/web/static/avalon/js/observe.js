// Observe mode JavaScript - Pixel Town Style

window.addEventListener('beforeunload', () => {
    // Observer usually doesn't need to keep session state, 
    // but we keep language preference
    const keysToKeep = ['gameLanguage'];
    Object.keys(sessionStorage).forEach(key => {
        if (!keysToKeep.includes(key)) {
            sessionStorage.removeItem(key);
        }
    });
});

window.addEventListener('pageshow', (event) => {
    if (event.persisted) {
        window.location.reload();
    }
});

const wsClient = new WebSocketClient();
const messagesContainer = document.getElementById('messages-container');
const phaseDisplay = document.getElementById('phase-display');
const missionDisplay = document.getElementById('mission-display');
const roundDisplay = document.getElementById('round-display');
const statusDisplay = document.getElementById('status-display');
const gameSetup = document.getElementById('game-setup');
const startGameBtn = document.getElementById('start-game-btn');
const numPlayersSelect = document.getElementById('num-players');
const languageSelect = document.getElementById('language');
const tablePlayers = document.getElementById('table-players');

let messageCount = 0;
let gameStarted = false;
let numPlayers = 5;

// State to store player metadata (received from server)
// Format: { 0: { name: "BotA", is_human: false, portrait_id: 1 }, ... }
let playerMetadata = {}; 

const gameLanguage = sessionStorage.getItem('gameLanguage') || 'en';
document.body.classList.add(`lang-${gameLanguage}`);

// Legacy: Try to load from session if available (Local Debugging)
let selectedPortraits = [];
try {
    const stored = sessionStorage.getItem('selectedPortraits');
    if (stored) selectedPortraits = JSON.parse(stored);
} catch (e) {}

let agentConfigs = {};
try {
    const gameConfigStr = sessionStorage.getItem('gameConfig');
    if (gameConfigStr) {
        const gameConfig = JSON.parse(gameConfigStr);
        if (gameConfig.agent_configs) {
            agentConfigs = gameConfig.agent_configs;
        }
    }
} catch (e) {}


// --- VISUAL HELPERS ---

function getPortraitSrc(playerId) {
    const pid = parseInt(playerId, 10);
    
    // 1. Priority: Metadata from Server (Multiplayer)
    if (playerMetadata[pid]) {
        const meta = playerMetadata[pid];
        if (meta.is_human) {
            return `/static/portraits/portrait_human.png`;
        }
        if (meta.portrait_id) {
            return `/static/portraits/portrait_${meta.portrait_id}.png`;
        }
    }

    // 2. Fallback: Local Session (Legacy / Single Player)
    if (selectedPortraits && selectedPortraits.length > 0) {
        // Simple mapping: If we have a list, use it directly or map by index
        // If pid exists in selectedPortraits (as a value), that's wrong. 
        // selectedPortraits is usually [id1, id2, id3...] matching Player 0, 1, 2...
        if (pid < selectedPortraits.length) {
             const pId = selectedPortraits[pid];
             // If pId is -1 (legacy placeholder for human), return human
             if (pId === -1) return `/static/portraits/portrait_human.png`;
             return `/static/portraits/portrait_${pId}.png`;
        }
    }
    
    // 3. Fallback: Generic Robot
    const id = (pid % 15) + 1;
    return `/static/portraits/portrait_${id}.png`;
}

function getModelName(playerId) {
    const pid = parseInt(playerId, 10);

    // 1. Priority: Metadata from Server
    if (playerMetadata[pid]) {
        return playerMetadata[pid].name || `Player ${pid}`;
    }

    // 2. Fallback: Local Config
    let portraitId = null;
    if (selectedPortraits && pid < selectedPortraits.length) {
        portraitId = selectedPortraits[pid];
    }
    
    if (portraitId && portraitId !== -1) {
        if (agentConfigs && agentConfigs[portraitId]) {
            return agentConfigs[portraitId].base_model;
        }
    } else if (portraitId === -1) {
        return "Human";
    }
    
    return `Player ${pid}`;
}

function polarPositions(count, radiusX, radiusY) {
    return Array.from({ length: count }).map((_, i) => {
        const angle = (Math.PI * 2 * i) / count - Math.PI / 2;
        return { x: radiusX * Math.cos(angle), y: radiusY * Math.sin(angle) };
    });
}

function setupTablePlayers(count) {
    numPlayers = count;
    tablePlayers.innerHTML = '';
    
    const rect = tablePlayers.getBoundingClientRect();
    const cx = rect.width / 2;
    const cy = rect.height / 2;
    const radiusX = Math.min(300, Math.max(160, rect.width * 0.45));
    const radiusY = Math.min(180, Math.max(100, rect.height * 0.40));
    const positions = polarPositions(count, radiusX, radiusY);
    
    for (let i = 0; i < count; i++) {
        const seat = document.createElement('div');
        seat.className = 'seat';
        seat.dataset.playerId = String(i);
        
        const modelName = getModelName(i);
        
        seat.innerHTML = `
            <div class="seat-label"></div>
            <span class="id-tag">P${i}</span>
            <img src="${getPortraitSrc(i)}" alt="Player ${i}">
            <span class="name-tag">${modelName}</span>
            <div class="speech-bubble">💬</div>
        `;
        seat.style.left = `${cx + positions[i].x - 34}px`;
        seat.style.top = `${cy + positions[i].y - 34}px`;
        const baseRotation = (i % 2 ? 1 : -1) * 2;
        seat.style.setProperty('--base-rotation', `${baseRotation}deg`);
        seat.style.transform = `rotate(var(--base-rotation, 0deg))`;
        tablePlayers.appendChild(seat);
    }
}

function highlightSpeaker(playerId) {
    document.querySelectorAll('.seat').forEach(seat => {
        const isSpeaking = seat.dataset.playerId === String(playerId);
        
        if (isSpeaking && !seat.classList.contains('speaking')) {
            const bubble = seat.querySelector('.speech-bubble');
            if (bubble) {
                bubble.style.animation = 'none';
                bubble.offsetHeight;
                bubble.style.animation = '';
            }
        }
        
        seat.classList.toggle('speaking', isSpeaking);
    });
}

function formatTime(timestamp) {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function addMessage(message) {
    messageCount++;
    if (messageCount === 1) messagesContainer.innerHTML = '';
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'chat-message';
    
    let senderType = 'system';
    let avatarHtml = '<div class="chat-avatar system">🎭</div>';
    let senderName = message.sender || 'System';
    let playerId = null;
    
    if (message.sender === 'Moderator') {
        senderType = 'moderator';
        avatarHtml = '<div class="chat-avatar system">⚔</div>';
    } else if (message.sender && message.sender.startsWith('Player')) {
        senderType = 'agent';
        const match = message.sender.match(/Player\s*(\d+)/);
        if (match) {
            playerId = parseInt(match[1], 10);
            const portraitSrc = getPortraitSrc(playerId);
            avatarHtml = `<div class="chat-avatar"><img src="${portraitSrc}" alt="${senderName}"></div>`;
            highlightSpeaker(playerId);
        } else {
            avatarHtml = '<div class="chat-avatar system">🎭</div>';
        }
    }
    
    messageDiv.innerHTML = `
        ${avatarHtml}
        <div class="chat-bubble">
            <div class="chat-header">
                <span class="chat-sender ${senderType}">${escapeHtml(senderName)}</span>
                <span class="chat-time">${formatTime(message.timestamp)}</span>
            </div>
            <div class="chat-content">${escapeHtml(message.content || '')}</div>
        </div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function updateGameState(state) {
    if (phaseDisplay) {
        const phases = ['Team Selection', 'Team Voting', 'Quest Voting', 'Assassination'];
        const phaseName = (state.phase !== null && state.phase !== undefined) ? (phases[state.phase] || 'Unknown') : '-';
        phaseDisplay.textContent = `Phase: ${phaseName}`;
    }
    if (missionDisplay) missionDisplay.textContent = `Mission: ${state.mission_id ?? '-'}`;
    if (roundDisplay) roundDisplay.textContent = `Round: ${state.round_id ?? '-'}`;
    if (statusDisplay) statusDisplay.textContent = `Status: ${state.status ?? 'Waiting'}`;
    
    // Auto-update table if player count is received from server
    if (state.num_players && state.num_players !== numPlayers) {
        setupTablePlayers(state.num_players);
    } else if (tablePlayers.children.length === 0 && state.num_players) {
        setupTablePlayers(state.num_players);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function updateRoleLabels(roles) {
    if (!roles || !Array.isArray(roles)) return;
    
    roles.forEach((roleInfo, playerId) => {
        // Support multiple role formats: [id, "RoleName", ...] or {role_name: "..."}
        let roleName = "";
        if (typeof roleInfo === 'string') roleName = roleInfo;
        else if (Array.isArray(roleInfo)) roleName = roleInfo[1]; // Common in avalon engine
        else if (roleInfo.role_name) roleName = roleInfo.role_name;
        
        if (!roleName) return;
        
        const seat = tablePlayers.querySelector(`.seat[data-player-id="${playerId}"]`);
        if (!seat) return;
        
        const label = seat.querySelector('.seat-label');
        if (!label) return;
        
        label.textContent = roleName;
        seat.classList.add('has-label');
    });
}

// --- WEBSOCKET HANDLERS ---

wsClient.onMessage('message', (message) => {
    addMessage(message);
});

// [NEW] Handle Metadata / Configuration broadcast
// Server should send this on connection or game start
wsClient.onMessage('game_config', (config) => {
    /* Expected config format:
       {
         players: [
           { id: 0, name: "Alice", is_human: true },
           { id: 1, name: "GPT-4", is_human: false, portrait_id: 5 },
           ...
         ]
       }
    */
    if (config && config.players) {
        config.players.forEach(p => {
            playerMetadata[p.id] = p;
        });
        // Re-render table with new metadata
        setupTablePlayers(config.players.length);
    }
});

wsClient.onMessage('game_state', (state) => {
    updateGameState(state);
    
    // If roles are revealed (Observer usually sees all)
    if (state.roles) {
        updateRoleLabels(state.roles);
    }
    
    if (state.status === 'running' && !gameStarted) {
        gameSetup.style.display = 'none';
        messagesContainer.style.display = 'flex';
        gameStarted = true;
    }
    if (state.status === 'stopped' || state.status === 'waiting') {
        gameStarted = false;
        // Don't clear message log completely, just show status
        // Only show setup if we are allowed to restart (Admin?)
        // For now, keep setup hidden to avoid confusion, unless empty
        if (messagesContainer.children.length === 0) {
             gameSetup.style.display = 'block'; 
        }
    }
});

wsClient.onMessage('error', (error) => {
    console.error('Error from server:', error);
    addMessage({
        sender: 'System',
        content: `Error: ${error.message || 'Unknown error'}`,
        timestamp: new Date().toISOString()
    });
});

// UI Event Listeners (Mainly for legacy/debug local start)
numPlayersSelect.addEventListener('change', () => {
    setupTablePlayers(parseInt(numPlayersSelect.value));
});

async function startGame() {
    const np = parseInt(numPlayersSelect.value);
    const language = languageSelect.value;
    
    try {
        startGameBtn.disabled = true;
        startGameBtn.textContent = 'Starting...';
        
        const response = await fetch('/api/start-game', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                game: 'avalon',
                num_players: np,
                language: language,
                mode: 'observe', // Explicitly start as observe mode
            }),
        });
        
        const result = await response.json();
        
        if (response.ok) {
            gameSetup.style.display = 'none';
            messagesContainer.style.display = 'flex';
            gameStarted = true;
        } else {
            alert(`Error: ${result.detail || 'Failed to start game'}`);
            startGameBtn.disabled = false;
            startGameBtn.textContent = 'Start Observing';
        }
    } catch (error) {
        console.error('Error starting game:', error);
        alert(`Error: ${error.message}`);
        startGameBtn.disabled = false;
        startGameBtn.textContent = 'Start Observing';
    }
}

startGameBtn.addEventListener('click', startGame);

// Init
wsClient.onConnect(() => {
    console.log('Connected to game server');
    // We don't automatically reset UI here to avoid flickering if reconnecting
});

wsClient.onDisconnect(() => {
    console.log('Disconnected from game server');
});

function initializeObserve() {
    // Initial setup (default 5 seats)
    setupTablePlayers(numPlayers);
    wsClient.connect();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeObserve);
} else {
    initializeObserve();
}

window.addEventListener('resize', () => {
    setupTablePlayers(numPlayers);
});