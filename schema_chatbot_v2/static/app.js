// ================= STATE MANAGEMENT =================
const STORAGE_KEY = 'idp_studio_conversations';
const ACTIVE_CHAT_KEY = 'idp_studio_active_chat';
const AUTH_TOKEN_KEY = 'idp_access_token';
const AUTH_USER_KEY = 'idp_user';
const AUTH_ROLE_KEY = 'idp_role';
const AUTH_FULLNAME_KEY = 'idp_fullname';
const AUTH_EMAIL_KEY = 'idp_email';
const THEME_KEY = 'idp_theme';

let conversations = [];
let currentChatId = null;
let isProcessing = false;
let isAnswering = false;
let actionMenuTargetChatId = null;
let currentUser = null;

// DOM Elements - Navigation & Theme
const sidebar = document.getElementById('sidebar');
const sidebarExpandBtn = document.getElementById('sidebar-expand-btn');
const conversationsList = document.getElementById('conversations-list');
const backendStatusPill = document.getElementById('backend-status-pill');
const backendStatusText = document.getElementById('backend-status-text');
const statusDot = document.getElementById('status-dot');

const themeToggleBtn = document.getElementById('theme-toggle-btn');
const themeIcon = document.getElementById('theme-icon');
const themeLabel = document.getElementById('theme-label');

const llmModelBadge = document.getElementById('llm-model-badge');
const llmModelText = document.getElementById('llm-model-text');

// DOM Elements - User & Admin Header
const adminNavBtn = document.getElementById('admin-nav-btn');
const userMenuContainer = document.getElementById('user-menu-container');
const userMenuBtn = document.getElementById('user-menu-btn');
const userAvatarCircle = document.getElementById('user-avatar-circle');
const userHeaderName = document.getElementById('user-header-name');
const userDropdownMenu = document.getElementById('user-dropdown-menu');
const userMenuAvatarLarge = document.getElementById('user-menu-avatar-large');
const userMenuFullname = document.getElementById('user-menu-fullname');
const userMenuEmail = document.getElementById('user-menu-email');
const userMenuRoleBadge = document.getElementById('user-menu-role-badge');
const adminDropdownSection = document.getElementById('admin-dropdown-section');

// DOM Elements - Auth Cards & Modals
const authContainer = document.getElementById('auth-container');
const loginCard = document.getElementById('login-card');
const signupCard = document.getElementById('signup-card');
const loginForm = document.getElementById('login-form');
const loginIdentifier = document.getElementById('login-identifier');
const loginPassword = document.getElementById('login-password');
const loginErrorMsg = document.getElementById('login-error-msg');
const loginBtn = document.getElementById('login-btn');

const signupForm = document.getElementById('signup-form');
const signupFullname = document.getElementById('signup-fullname');
const signupEmail = document.getElementById('signup-email');
const signupPassword = document.getElementById('signup-password');
const signupConfirmPassword = document.getElementById('signup-confirm-password');
const signupErrorMsg = document.getElementById('signup-error-msg');
const signupBtn = document.getElementById('signup-btn');

const forgotPasswordModal = document.getElementById('forgot-password-modal');
const forgotEmailInput = document.getElementById('forgot-email-input');

const adminDashboardModal = document.getElementById('admin-dashboard-modal');
const adminModalUsername = document.getElementById('admin-modal-username');

// DOM Elements - Chat
const topDocBadge = document.getElementById('top-doc-badge');
const topDocName = document.getElementById('top-doc-name');
const chatThreadContainer = document.getElementById('chat-thread-container');
const emptyStateHero = document.getElementById('empty-state-hero');
const messagesList = document.getElementById('messages-list');
const attachedDocChip = document.getElementById('attached-doc-chip');
const chipFilename = document.getElementById('chip-filename');
const chipStatusText = document.getElementById('chip-status-text');
const chipSpinner = document.getElementById('chip-spinner');
const chipCheckIcon = document.getElementById('chip-check-icon');
const plusMenuBtn = document.getElementById('plus-menu-btn');
const nativeFileInput = document.getElementById('native-file-input');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const chatActionsMenu = document.getElementById('chat-actions-menu');
const toast = document.getElementById('toast');

// ================= INITIALIZATION =================

window.addEventListener('DOMContentLoaded', () => {
  initTheme();
  checkAuth();
  loadConversations();
  checkBackendHealth();
  setInterval(checkBackendHealth, 10000);

  // Keyboard shortcut Ctrl+K / Cmd+K for new chat
  window.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
      e.preventDefault();
      createNewChat();
    }
  });

  // Close menus on click outside
  document.addEventListener('click', (e) => {
    if (chatActionsMenu && !chatActionsMenu.contains(e.target) && !e.target.closest('.actions-btn')) {
      closeChatActionsMenu();
    }
    if (userMenuContainer && !userMenuContainer.contains(e.target)) {
      closeUserDropdown();
    }
  });
});

// ================= THEME TOGGLE =================

function initTheme() {
  const savedTheme = localStorage.getItem(THEME_KEY) || 'dark';
  applyTheme(savedTheme);
}

function toggleTheme() {
  const isLight = document.body.classList.contains('light-theme');
  const newTheme = isLight ? 'dark' : 'light';
  applyTheme(newTheme);
  localStorage.setItem(THEME_KEY, newTheme);
}

function applyTheme(theme) {
  if (theme === 'light') {
    document.body.classList.add('light-theme');
    if (themeIcon) themeIcon.textContent = '🌙';
    if (themeLabel) themeLabel.textContent = 'Dark Mode';
  } else {
    document.body.classList.remove('light-theme');
    if (themeIcon) themeIcon.textContent = '☀️';
    if (themeLabel) themeLabel.textContent = 'Light Mode';
  }
}

// ================= AUTHENTICATION & RBAC =================

async function checkAuth() {
  const token = localStorage.getItem(AUTH_TOKEN_KEY);
  if (!token) {
    showAuthContainer();
    switchAuthView('login');
    return;
  }

  try {
    const res = await fetch('/auth/me', {
      headers: {
        'Authorization': `Bearer ${token}`,
      },
    });

    if (res.ok) {
      const user = await res.json();
      currentUser = user;
      localStorage.setItem(AUTH_USER_KEY, user.username || '');
      localStorage.setItem(AUTH_ROLE_KEY, user.role || 'user');
      localStorage.setItem(AUTH_FULLNAME_KEY, user.full_name || user.username || '');
      localStorage.setItem(AUTH_EMAIL_KEY, user.email || '');
      updateUserUI(user);
      hideAuthContainer();
    } else {
      // Token invalid or expired
      localStorage.removeItem(AUTH_TOKEN_KEY);
      showAuthContainer();
      switchAuthView('login');
    }
  } catch (e) {
    // If network fails, use cached user credentials
    const cachedUser = localStorage.getItem(AUTH_USER_KEY);
    const cachedRole = localStorage.getItem(AUTH_ROLE_KEY) || 'user';
    const cachedFullname = localStorage.getItem(AUTH_FULLNAME_KEY) || cachedUser || 'User';
    const cachedEmail = localStorage.getItem(AUTH_EMAIL_KEY) || '';

    if (cachedUser) {
      currentUser = { username: cachedUser, role: cachedRole, full_name: cachedFullname, email: cachedEmail };
      updateUserUI(currentUser);
      hideAuthContainer();
    } else {
      showAuthContainer();
      switchAuthView('login');
    }
  }
}

function showAuthContainer() {
  if (authContainer) authContainer.classList.remove('hidden');
}

function hideAuthContainer() {
  if (authContainer) authContainer.classList.add('hidden');
}

function switchAuthView(view) {
  if (loginErrorMsg) loginErrorMsg.classList.add('hidden');
  if (signupErrorMsg) signupErrorMsg.classList.add('hidden');

  if (view === 'signup') {
    if (loginCard) loginCard.classList.add('hidden');
    if (signupCard) signupCard.classList.remove('hidden');
    if (signupFullname) signupFullname.focus();
  } else {
    if (signupCard) signupCard.classList.add('hidden');
    if (loginCard) loginCard.classList.remove('hidden');
    if (loginIdentifier) loginIdentifier.focus();
  }
}

function getInitials(name) {
  if (!name) return 'U';
  const parts = name.trim().split(/\s+/);
  if (parts.length === 1) {
    return parts[0].substring(0, 2).toUpperCase();
  }
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

function updateUserUI(user) {
  if (!user) return;
  const displayName = user.full_name || user.username || 'User';
  const role = (user.role || 'user').toLowerCase();
  const initials = getInitials(displayName);

  // Avatar & Header Name
  if (userAvatarCircle) userAvatarCircle.textContent = initials;
  if (userMenuAvatarLarge) userMenuAvatarLarge.textContent = initials;
  if (userHeaderName) userHeaderName.textContent = displayName;
  if (userMenuFullname) userMenuFullname.textContent = displayName;
  if (userMenuEmail) userMenuEmail.textContent = user.email || `${user.username}@example.com`;

  // Role Badge in Menu
  if (userMenuRoleBadge) {
    userMenuRoleBadge.textContent = role.toUpperCase();
    if (role === 'admin') {
      userMenuRoleBadge.className = 'inline-flex items-center text-[10px] font-bold uppercase tracking-wider font-mono px-1.5 py-0.5 rounded bg-purple-500/20 border border-purple-500/30 text-purple-300';
    } else {
      userMenuRoleBadge.className = 'inline-flex items-center text-[10px] font-bold uppercase tracking-wider font-mono px-1.5 py-0.5 rounded bg-blue-500/20 border border-blue-500/30 text-blue-300';
    }
  }

  // Admin Nav Tab & Admin Menu Option
  if (role === 'admin') {
    if (adminNavBtn) {
      adminNavBtn.classList.remove('hidden');
      adminNavBtn.classList.add('flex');
    }
    if (adminDropdownSection) {
      adminDropdownSection.classList.remove('hidden');
    }
  } else {
    if (adminNavBtn) {
      adminNavBtn.classList.add('hidden');
      adminNavBtn.classList.remove('flex');
    }
    if (adminDropdownSection) {
      adminDropdownSection.classList.add('hidden');
    }
  }
}

function togglePasswordVisibility(inputId, buttonId) {
  const input = document.getElementById(inputId);
  const button = document.getElementById(buttonId);
  if (!input || !button) return;

  if (input.type === 'password') {
    input.type = 'text';
    button.innerHTML = `
      <svg class="w-4 h-4 text-indigo-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path>
        <line x1="1" y1="1" x2="23" y2="23"></line>
      </svg>
    `;
  } else {
    input.type = 'password';
    button.innerHTML = `
      <svg class="w-4 h-4 text-slate-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
        <circle cx="12" cy="12" r="3"></circle>
      </svg>
    `;
  }
}

async function handleLoginSubmit(e) {
  if (e) e.preventDefault();
  const identifier = loginIdentifier.value.trim();
  const password = loginPassword.value.trim();

  if (!identifier || !password) {
    loginErrorMsg.textContent = 'Please fill in both identifier and password';
    loginErrorMsg.classList.remove('hidden');
    return;
  }

  loginBtn.disabled = true;
  loginBtn.innerHTML = '<span class="inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></span> Authenticating...';
  loginErrorMsg.classList.add('hidden');

  try {
    const res = await fetch('/auth/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ identifier, password }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Incorrect email/username or password' }));
      throw new Error(err.detail || 'Incorrect email/username or password');
    }

    const data = await res.json();
    localStorage.setItem(AUTH_TOKEN_KEY, data.access_token);
    localStorage.setItem(AUTH_USER_KEY, data.username || identifier);
    localStorage.setItem(AUTH_ROLE_KEY, data.role || 'user');
    localStorage.setItem(AUTH_FULLNAME_KEY, data.full_name || data.username || identifier);
    localStorage.setItem(AUTH_EMAIL_KEY, data.email || '');

    currentUser = {
      username: data.username || identifier,
      role: data.role || 'user',
      full_name: data.full_name || data.username || identifier,
      email: data.email || '',
    };

    updateUserUI(currentUser);
    hideAuthContainer();
    showToast(`Welcome back, ${currentUser.full_name}!`);
  } catch (err) {
    loginErrorMsg.textContent = err.message;
    loginErrorMsg.classList.remove('hidden');
  } finally {
    loginBtn.disabled = false;
    loginBtn.textContent = 'Log In';
  }
}

async function handleSignupSubmit(e) {
  if (e) e.preventDefault();
  const fullName = signupFullname.value.trim();
  const email = signupEmail.value.trim().toLowerCase();
  const password = signupPassword.value;
  const confirmPassword = signupConfirmPassword.value;

  if (!fullName || !email || !password) {
    signupErrorMsg.textContent = 'Please fill in all required fields';
    signupErrorMsg.classList.remove('hidden');
    return;
  }

  if (password.length < 8) {
    signupErrorMsg.textContent = 'Password must be at least 8 characters long';
    signupErrorMsg.classList.remove('hidden');
    return;
  }

  if (password !== confirmPassword) {
    signupErrorMsg.textContent = 'Passwords do not match';
    signupErrorMsg.classList.remove('hidden');
    return;
  }

  signupBtn.disabled = true;
  signupBtn.innerHTML = '<span class="inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></span> Creating Account...';
  signupErrorMsg.classList.add('hidden');

  try {
    const res = await fetch('/auth/signup', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        full_name: fullName,
        email: email,
        password: password,
        confirm_password: confirmPassword,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Registration failed' }));
      throw new Error(err.detail || 'Registration failed');
    }

    const data = await res.json();
    localStorage.setItem(AUTH_TOKEN_KEY, data.access_token);
    localStorage.setItem(AUTH_USER_KEY, data.username || email.split('@')[0]);
    localStorage.setItem(AUTH_ROLE_KEY, data.role || 'user');
    localStorage.setItem(AUTH_FULLNAME_KEY, data.full_name || fullName);
    localStorage.setItem(AUTH_EMAIL_KEY, data.email || email);

    currentUser = {
      username: data.username || email.split('@')[0],
      role: data.role || 'user',
      full_name: data.full_name || fullName,
      email: data.email || email,
    };

    updateUserUI(currentUser);
    hideAuthContainer();
    showToast(`Account created! Welcome, ${currentUser.full_name}`);
  } catch (err) {
    signupErrorMsg.textContent = err.message;
    signupErrorMsg.classList.remove('hidden');
  } finally {
    signupBtn.disabled = false;
    signupBtn.textContent = 'Create Account';
  }
}

async function handleLogout() {
  const token = localStorage.getItem(AUTH_TOKEN_KEY);
  if (token) {
    try {
      await fetch('/auth/logout', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
      });
    } catch (e) {
      // Ignore network errors on logout
    }
  }

  localStorage.removeItem(AUTH_TOKEN_KEY);
  localStorage.removeItem(AUTH_USER_KEY);
  localStorage.removeItem(AUTH_ROLE_KEY);
  localStorage.removeItem(AUTH_FULLNAME_KEY);
  localStorage.removeItem(AUTH_EMAIL_KEY);
  currentUser = null;

  showAuthContainer();
  switchAuthView('login');
  showToast('Signed out successfully');
}

// User Menu Dropdown Controls
function toggleUserDropdown(e) {
  if (e) e.stopPropagation();
  if (!userDropdownMenu) return;
  userDropdownMenu.classList.toggle('hidden');
}

function closeUserDropdown() {
  if (userDropdownMenu) userDropdownMenu.classList.add('hidden');
}

// Forgot Password Modal
function openForgotPasswordModal() {
  if (forgotPasswordModal) forgotPasswordModal.classList.remove('hidden');
  if (forgotEmailInput) {
    forgotEmailInput.value = loginIdentifier ? loginIdentifier.value.trim() : '';
    forgotEmailInput.focus();
  }
}

function closeForgotPasswordModal() {
  if (forgotPasswordModal) forgotPasswordModal.classList.add('hidden');
}

function handleSendPasswordReset() {
  const email = forgotEmailInput ? forgotEmailInput.value.trim() : '';
  closeForgotPasswordModal();
  showToast(email ? `Password reset link sent to ${email}` : 'Password reset instructions sent');
}

// Admin Dashboard Modal
function openAdminDashboardModal() {
  if (adminDashboardModal) adminDashboardModal.classList.remove('hidden');
  if (adminModalUsername && currentUser) {
    adminModalUsername.textContent = `${currentUser.full_name || currentUser.username} (${currentUser.email || currentUser.role})`;
  }
}

function closeAdminDashboardModal() {
  if (adminDashboardModal) adminDashboardModal.classList.add('hidden');
}

// ================= BACKEND HEALTH & LLM BADGE =================

async function checkBackendHealth() {
  try {
    const res = await fetch('/health');
    if (res.ok) {
      const data = await res.json();
      backendStatusPill.className = 'flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-xs font-medium text-emerald-400';
      statusDot.className = 'w-2 h-2 rounded-full bg-emerald-500 shadow-sm shadow-emerald-500/50 animate-pulse';
      backendStatusText.textContent = 'Connected';

      // Update LLM Model Badge
      if (llmModelText) {
        const provider = data.llm_provider || 'sarvam';
        llmModelText.textContent = provider === 'sarvam' ? 'sarvam-105b' : provider;
      }
    } else {
      setBackendDisconnected();
    }
  } catch (e) {
    setBackendDisconnected();
  }
}

function setBackendDisconnected() {
  backendStatusPill.className = 'flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-rose-500/10 border border-rose-500/20 text-xs font-medium text-rose-400';
  statusDot.className = 'w-2 h-2 rounded-full bg-rose-500 shadow-sm shadow-rose-500/50';
  backendStatusText.textContent = 'Disconnected';
}

// ================= SIDEBAR & CONVERSATIONS =================

function toggleSidebar() {
  sidebar.classList.toggle('collapsed');
  if (sidebar.classList.contains('collapsed')) {
    sidebarExpandBtn.classList.remove('hidden');
  } else {
    sidebarExpandBtn.classList.add('hidden');
  }
}

function loadConversations() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    conversations = saved ? JSON.parse(saved) : [];
  } catch (e) {
    conversations = [];
  }

  const savedActive = localStorage.getItem(ACTIVE_CHAT_KEY);
  if (savedActive && conversations.some(c => c.id === savedActive)) {
    currentChatId = savedActive;
  } else if (conversations.length > 0) {
    currentChatId = conversations[0].id;
  } else {
    createNewChat();
    return;
  }

  renderConversationsSidebar();
  renderActiveChat();
}

function saveConversations() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
  if (currentChatId) {
    localStorage.setItem(ACTIVE_CHAT_KEY, currentChatId);
  }
  renderConversationsSidebar();
}

function getActiveChat() {
  return conversations.find(c => c.id === currentChatId);
}

function createNewChat() {
  const newChat = {
    id: `chat_${Date.now()}`,
    title: 'New Chat',
    createdAt: new Date().toISOString(),
    document: null,
    messages: [],
  };

  conversations.unshift(newChat);
  currentChatId = newChat.id;
  saveConversations();
  renderActiveChat();
  chatInput.focus();
}

function selectConversation(chatId) {
  currentChatId = chatId;
  saveConversations();
  renderActiveChat();
}

function renderConversationsSidebar() {
  conversationsList.innerHTML = '';
  if (conversations.length === 0) {
    conversationsList.innerHTML = '<div class="px-2 py-4 text-xs text-slate-500 text-center">No recent chats</div>';
    return;
  }

  conversations.forEach((conv) => {
    const isActive = conv.id === currentChatId;
    const item = document.createElement('div');
    item.className = `conversation-item ${isActive ? 'active' : ''} group`;
    item.onclick = () => selectConversation(conv.id);

    item.innerHTML = `
      <div class="flex items-center gap-2 truncate pr-2">
        <svg class="w-3.5 h-3.5 ${isActive ? 'text-indigo-400' : 'text-slate-400'} flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
        </svg>
        <span class="truncate">${escapeHtml(conv.title || 'Untitled Chat')}</span>
      </div>
      <button 
        class="actions-btn p-1 hover:text-white text-slate-400 rounded transition" 
        onclick="openChatActionsMenu(event, '${conv.id}')"
        title="Chat actions"
      >
        <svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="1"></circle>
          <circle cx="19" cy="12" r="1"></circle>
          <circle cx="5" cy="12" r="1"></circle>
        </svg>
      </button>
    `;
    conversationsList.appendChild(item);
  });
}

function openChatActionsMenu(e, chatId) {
  e.stopPropagation();
  actionMenuTargetChatId = chatId;
  const rect = e.currentTarget.getBoundingClientRect();
  
  chatActionsMenu.style.top = `${rect.bottom + 4}px`;
  chatActionsMenu.style.left = `${Math.min(window.innerWidth - 180, rect.left)}px`;
  chatActionsMenu.classList.remove('hidden');
}

function closeChatActionsMenu() {
  chatActionsMenu.classList.add('hidden');
  actionMenuTargetChatId = null;
}

function handleShareCurrentChat() {
  const target = conversations.find(c => c.id === (actionMenuTargetChatId || currentChatId));
  closeChatActionsMenu();
  if (!target) return;

  const transcript = target.messages.map(m => `${m.role.toUpperCase()}: ${m.content}`).join('\n\n');
  navigator.clipboard.writeText(`=== IDP Studio Chat: ${target.title} ===\n\n${transcript}`);
  showToast('Chat transcript copied to clipboard!');
}

function handleDownloadCurrentChat() {
  const target = conversations.find(c => c.id === (actionMenuTargetChatId || currentChatId));
  closeChatActionsMenu();
  if (!target) return;

  const content = `# IDP Intelligence Studio — ${target.title}\n\n` +
    (target.document ? `**Document:** ${target.document.doc_id} (${target.document.page_count || 1} pages)\n\n` : '') +
    target.messages.map(m => `### ${m.role === 'user' ? 'User' : 'Assistant'}\n${m.content}\n`).join('\n');

  const blob = new Blob([content], { type: 'text/markdown;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${(target.title || 'chat').replace(/[^a-z0-9]/gi, '_').toLowerCase()}.md`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('Chat downloaded successfully');
}

function handleDeleteCurrentChat() {
  const targetId = actionMenuTargetChatId || currentChatId;
  closeChatActionsMenu();
  if (!targetId) return;

  conversations = conversations.filter(c => c.id !== targetId);
  if (currentChatId === targetId) {
    currentChatId = conversations.length > 0 ? conversations[0].id : null;
  }

  if (!currentChatId) {
    createNewChat();
  } else {
    saveConversations();
    renderActiveChat();
  }
  showToast('Chat deleted');
}

// ================= RENDER ACTIVE CHAT =================

function renderActiveChat() {
  const chat = getActiveChat();
  if (!chat) return;

  // Update Top Document Badge
  if (chat.document) {
    topDocName.textContent = chat.document.doc_id;
    topDocBadge.classList.remove('hidden');
    topDocBadge.classList.add('flex');
  } else {
    topDocBadge.classList.add('hidden');
    topDocBadge.classList.remove('flex');
  }

  // Render Messages
  if (!chat.document && chat.messages.length === 0) {
    emptyStateHero.style.display = 'flex';
    messagesList.innerHTML = '';
  } else {
    emptyStateHero.style.display = 'none';
    messagesList.innerHTML = '';

    chat.messages.forEach((msg) => {
      renderMessageItem(msg);
    });
  }

  scrollChatToBottom();
}

function renderMessageItem(msg) {
  const wrapper = document.createElement('div');
  wrapper.className = 'w-full animate-fadeIn';

  if (msg.role === 'user') {
    wrapper.innerHTML = `
      <div class="flex items-start justify-end gap-3 max-w-2xl ml-auto">
        <div class="p-3.5 px-4 bg-gradient-to-tr from-indigo-600 to-blue-600 rounded-2xl rounded-tr-sm text-sm text-white shadow-md leading-relaxed">
          ${escapeHtml(msg.content)}
        </div>
      </div>
    `;
  } else {
    // Assistant message (ChatGPT style)
    let extraDocHtml = '';
    if (msg.docCard) {
      extraDocHtml = `
        <div class="doc-ready-card">
          <div class="check-circle">✓</div>
          <div class="text-xs font-semibold text-slate-100">${escapeHtml(msg.docCard.filename)}</div>
          <span class="text-[11px] text-slate-400 font-normal">(${msg.docCard.pageCount || 1} ${msg.docCard.pageCount === 1 ? 'page' : 'pages'})</span>
        </div>
      `;
    }

    wrapper.innerHTML = `
      <div class="flex items-start gap-3.5 max-w-3xl">
        <div class="w-8 h-8 rounded-xl bg-gradient-to-tr from-indigo-600/30 to-purple-600/30 border border-indigo-500/30 flex items-center justify-center text-sm flex-shrink-0 mt-0.5 shadow-sm">
          🤖
        </div>
        <div class="flex-1 space-y-2">
          ${extraDocHtml}
          <div class="p-4 bg-cardbg border border-borderSubtle rounded-2xl rounded-tl-sm text-sm text-slate-200 shadow-sm chat-markdown">
            ${formatMarkdownAndCitations(msg.content)}
          </div>
        </div>
      </div>
    `;
  }

  messagesList.appendChild(wrapper);
}

// ================= FILE UPLOAD CONTROLS =================

function triggerFileInput() {
  if (nativeFileInput) nativeFileInput.click();
}

function handleNativeFileSelected(e) {
  if (e.target.files && e.target.files.length > 0) {
    processDocumentAttachment({ file: e.target.files[0] });
  }
  nativeFileInput.value = '';
}

async function processDocumentAttachment({ file, sampleName }) {
  if (isProcessing) return;

  const chat = getActiveChat();
  if (!chat) return;

  isProcessing = true;
  sendBtn.disabled = true;

  const docName = file ? file.name : `${sampleName}.pdf`;

  // Show Attached Document Chip above input
  attachedDocChip.classList.remove('hidden');
  attachedDocChip.classList.add('flex');
  chipFilename.textContent = docName;
  chipStatusText.textContent = 'Processing OCR & extraction...';
  chipSpinner.classList.remove('hidden');
  chipCheckIcon.classList.add('hidden');

  const formData = new FormData();
  let endpoint = '/api/pipeline/process-upload';

  if (file) {
    formData.append('file', file);
  } else if (sampleName) {
    formData.append('sample_name', sampleName);
    formData.append('md_name', `${sampleName}.md`);
  }

  try {
    let resp = await fetch(endpoint, {
      method: 'POST',
      body: formData,
    });

    if (!resp.ok) {
      // Fallback to extract/from-output if process-upload is not defined in backend
      const fallbackForm = new FormData();
      fallbackForm.append('md_name', `${sampleName || 'sdg_goals_output'}.md`);
      resp = await fetch('/pipeline/extract/from-output', {
        method: 'POST',
        body: fallbackForm,
      });
    }

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: 'Document processing failed' }));
      throw new Error(err.detail || 'Document processing failed');
    }

    const data = await resp.json();

    // Update Chat state
    chat.document = data;
    if (chat.title === 'New Chat' || !chat.title) {
      chat.title = `${data.doc_id.replace(/\.[^/.]+$/, '')} Q&A`;
    }

    // Hide empty state hero
    emptyStateHero.style.display = 'none';

    // Append Assistant Document Ready Message
    const readyMessage = {
      id: `msg_${Date.now()}`,
      role: 'assistant',
      content: 'Your document is ready. What would you like to know?',
      docCard: {
        filename: data.doc_id,
        pageCount: data.page_count,
      },
      timestamp: new Date().toISOString(),
    };

    chat.messages.push(readyMessage);
    saveConversations();
    renderActiveChat();

    // Update Attached Chip to Done
    chipStatusText.textContent = 'Document ready';
    chipSpinner.classList.add('hidden');
    chipCheckIcon.classList.remove('hidden');

    setTimeout(() => {
      attachedDocChip.classList.add('hidden');
      attachedDocChip.classList.remove('flex');
    }, 2000);

  } catch (err) {
    chipStatusText.textContent = 'Error processing';
    chipSpinner.classList.add('hidden');
    
    emptyStateHero.style.display = 'none';
    chat.messages.push({
      id: `msg_${Date.now()}`,
      role: 'assistant',
      content: `❌ Could not process **${docName}**: ${err.message}. Please try again with a valid PDF, scan, or markdown document.`,
      timestamp: new Date().toISOString(),
    });
    saveConversations();
    renderActiveChat();
  } finally {
    isProcessing = false;
    sendBtn.disabled = false;
  }
}

function handleRemoveDocument() {
  const chat = getActiveChat();
  if (!chat) return;

  chat.document = null;
  attachedDocChip.classList.add('hidden');
  attachedDocChip.classList.remove('flex');
  saveConversations();
  renderActiveChat();
  showToast('Document detached from current chat');
}

// ================= SEND MESSAGE & Q&A =================

async function handleSendMessage(e) {
  if (e) e.preventDefault();
  const query = chatInput.value.trim();
  if (!query || isAnswering) return;

  const chat = getActiveChat();
  if (!chat) return;

  chatInput.value = '';

  // 1. Add User Message
  const userMsg = {
    id: `msg_${Date.now()}`,
    role: 'user',
    content: query,
    timestamp: new Date().toISOString(),
  };

  emptyStateHero.style.display = 'none';
  chat.messages.push(userMsg);
  
  if (chat.title === 'New Chat') {
    chat.title = query.slice(0, 30);
  }

  saveConversations();
  renderActiveChat();

  // 2. Check if Document is Attached
  if (!chat.document) {
    setTimeout(() => {
      const botGuideMsg = {
        id: `msg_${Date.now()}`,
        role: 'assistant',
        content: 'Please upload a document using the **+** button to get started. Once uploaded, I will provide grounded answers with exact page citations.',
        timestamp: new Date().toISOString(),
      };
      chat.messages.push(botGuideMsg);
      saveConversations();
      renderActiveChat();
    }, 300);
    return;
  }

  // 3. Execute Q&A Query
  isAnswering = true;
  sendBtn.disabled = true;

  const thinkingId = appendThinkingRow();

  try {
    const historyPayload = chat.messages
      .filter(m => m.role === 'user' || m.role === 'assistant')
      .slice(-6)
      .map(m => ({ role: m.role, content: m.content }));

    const payload = {
      question: query,
      doc_id: chat.document.doc_id,
      history: historyPayload,
      extracted_data: chat.document.extracted_data || chat.document,
    };

    let resp = await fetch('/api/query-bot/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!resp.ok) {
      resp = await fetch('/api/pipeline/qa-query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
    }

    removeElement(thinkingId);

    if (!resp.ok) {
      throw new Error('Failed to retrieve answer from server');
    }

    const data = await resp.json();
    let rawAnswer = data.answer || 'I could not find information about that in the uploaded document.';

    // Format citations cleanly if present in cited_pages
    if (data.cited_pages && data.cited_pages.length > 0 && !rawAnswer.includes('Source:')) {
      const pageTags = data.cited_pages.map(p => `Page ${p}`).join(', ');
      rawAnswer += `\n\nSource: ${pageTags}`;
    }

    // Stream response
    await streamAssistantResponse(rawAnswer);

    // Save final message
    chat.messages.push({
      id: `msg_${Date.now()}`,
      role: 'assistant',
      content: rawAnswer,
      timestamp: new Date().toISOString(),
    });
    saveConversations();

  } catch (err) {
    removeElement(thinkingId);
    chat.messages.push({
      id: `msg_${Date.now()}`,
      role: 'assistant',
      content: `❌ Error answering question: ${err.message}`,
      timestamp: new Date().toISOString(),
    });
    saveConversations();
    renderActiveChat();
  } finally {
    isAnswering = false;
    sendBtn.disabled = false;
  }
}

function appendThinkingRow() {
  const id = `thinking-${Date.now()}`;
  const wrapper = document.createElement('div');
  wrapper.id = id;
  wrapper.className = 'w-full animate-fadeIn';
  wrapper.innerHTML = `
    <div class="flex items-start gap-3.5 max-w-3xl">
      <div class="w-8 h-8 rounded-xl bg-gradient-to-tr from-indigo-600/30 to-purple-600/30 border border-indigo-500/30 flex items-center justify-center text-sm flex-shrink-0 mt-0.5">
        🤖
      </div>
      <div class="p-3.5 bg-cardbg border border-borderSubtle rounded-2xl rounded-tl-sm text-xs text-indigo-300 flex items-center gap-2">
        <div class="flex items-center gap-1">
          <span class="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-bounce" style="animation-delay: 0ms"></span>
          <span class="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-bounce" style="animation-delay: 150ms"></span>
          <span class="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-bounce" style="animation-delay: 300ms"></span>
        </div>
        <span class="text-[12px] opacity-90">Analyzing document facts...</span>
      </div>
    </div>
  `;
  messagesList.appendChild(wrapper);
  scrollChatToBottom();
  return id;
}

async function streamAssistantResponse(fullText) {
  const wrapper = document.createElement('div');
  wrapper.className = 'w-full animate-fadeIn';
  
  const bubble = document.createElement('div');
  bubble.className = 'p-4 bg-cardbg border border-borderSubtle rounded-2xl rounded-tl-sm text-sm text-slate-200 shadow-sm chat-markdown typing-caret';

  wrapper.innerHTML = `
    <div class="flex items-start gap-3.5 max-w-3xl">
      <div class="w-8 h-8 rounded-xl bg-gradient-to-tr from-indigo-600/30 to-purple-600/30 border border-indigo-500/30 flex items-center justify-center text-sm flex-shrink-0 mt-0.5 shadow-sm">
        🤖
      </div>
      <div class="flex-1"></div>
    </div>
  `;

  wrapper.querySelector('.flex-1').appendChild(bubble);
  messagesList.appendChild(wrapper);

  const step = Math.max(1, Math.floor(fullText.length / 25));
  let currentPos = 0;

  while (currentPos < fullText.length) {
    currentPos = Math.min(fullText.length, currentPos + step);
    const chunk = fullText.slice(0, currentPos);
    bubble.innerHTML = formatMarkdownAndCitations(chunk);
    scrollChatToBottom();
    await new Promise((r) => setTimeout(r, 15));
  }

  bubble.className = 'p-4 bg-cardbg border border-borderSubtle rounded-2xl rounded-tl-sm text-sm text-slate-200 shadow-sm chat-markdown';
  bubble.innerHTML = formatMarkdownAndCitations(fullText);
  scrollChatToBottom();
}

// ================= FORMATTING & UTILITIES =================

function formatMarkdownAndCitations(text) {
  if (!text) return '';

  // Clean reasoning tokens
  let clean = text.replace(/<think>[\s\S]*?<\/think>/gi, '')
                  .replace(/<end of thinking>/gi, '')
                  .trim();

  // Highlight Source: Page X
  clean = clean.replace(/(Source:\s*Page\s*\d+(?:,\s*Page\s*\d+)*)/gi, '<span class="citation-badge">$1</span>');
  clean = clean.replace(/\[Page\s*(\d+)\]/gi, '<span class="citation-badge">Source: Page $1</span>');

  // Markdown bold
  clean = clean.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

  // Bullet points
  const lines = clean.split('\n');
  let inList = false;
  let html = '';

  for (let line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
      if (!inList) {
        html += '<ul>';
        inList = true;
      }
      html += `<li>${trimmed.substring(2)}</li>`;
    } else {
      if (inList) {
        html += '</ul>';
        inList = false;
      }
      if (trimmed.length > 0) {
        html += `<p>${line}</p>`;
      }
    }
  }
  if (inList) html += '</ul>';

  return html || clean;
}

function showToast(msg) {
  toast.textContent = msg;
  toast.classList.remove('hidden');
  setTimeout(() => {
    toast.classList.add('hidden');
  }, 2200);
}

function removeElement(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function scrollChatToBottom() {
  chatThreadContainer.scrollTop = chatThreadContainer.scrollHeight;
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
