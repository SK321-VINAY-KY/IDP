(function () {
    'use strict';

    const API_BASE = window.location.origin;
    const TOKEN_KEY = 'idp_auth_token';
    const ROLE_KEY = 'idp_auth_role';

    let state = {
        token: localStorage.getItem(TOKEN_KEY) || null,
        role: localStorage.getItem(ROLE_KEY) || null,
        username: null,
        sessionId: null,
        docs: [],
        userDocs: [],
        schemas: [],
        currentSchema: null,
        currentErrors: [],
        currentState: 'idle',
        completed: false,
        jobs: [],
        userJobs: [],
        selectedJobId: null,
        pipelineAvailable: false,
        lastConfirmedSchemaId: null,
    };

    const $ = (id) => document.getElementById(id);
    const el = (tag, cls, html) => {
        const d = document.createElement(tag);
        if (cls) d.className = cls;
        if (html != null) d.innerHTML = html;
        return d;
    };

    function fmtSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / 1024 / 1024).toFixed(2) + ' MB';
    }

    function escapeHtml(str) {
        if (str === null || str === undefined) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    async function api(path, opts = {}) {
        const url = path.startsWith('http') ? path : API_BASE + path;
        const headers = opts.headers ? { ...opts.headers } : {};

        if (opts.json) {
            headers['Content-Type'] = 'application/json';
        }
        if (state.token) {
            headers['Authorization'] = 'Bearer ' + state.token;
        }

        const resp = await fetch(url, {
            ...opts,
            headers,
            body: opts.json ? JSON.stringify(opts.json) : opts.body,
        });

        if (resp.status === 401) {
            // Token expired or invalid
            logout();
            throw new Error('Session expired or unauthorized. Please log in again.');
        }

        const ct = resp.headers.get('content-type') || '';
        const body = ct.includes('application/json') ? await resp.json() : await resp.text();
        if (!resp.ok) {
            const msg = typeof body === 'object' ? (body.detail || resp.statusText) : String(body || resp.statusText);
            throw new Error(msg);
        }
        return body;
    }

    // ======================= Auth & Role Management =======================

    function showLoginModal() {
        const overlay = $('loginOverlay');
        if (overlay) overlay.classList.remove('hidden');
        const userHeader = $('userHeaderSection');
        if (userHeader) userHeader.classList.add('hidden');
    }

    function hideLoginModal() {
        const overlay = $('loginOverlay');
        if (overlay) overlay.classList.add('hidden');
    }

    function logout() {
        state.token = null;
        state.role = null;
        state.username = null;
        state.sessionId = null;
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem(ROLE_KEY);
        showLoginModal();
    }

    async function handleLoginSubmit(e) {
        e.preventDefault();
        const username = $('loginUsername').value.trim();
        const password = $('loginPassword').value;
        const errorBox = $('loginError');
        const submitBtn = $('loginSubmitBtn');

        if (!username || !password) return;

        submitBtn.disabled = true;
        submitBtn.textContent = 'Signing in...';
        hideStatus('loginError');

        try {
            const fd = new URLSearchParams();
            fd.append('username', username);
            fd.append('password', password);

            const resp = await fetch(API_BASE + '/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: fd.toString(),
            });

            const data = await resp.json();
            if (!resp.ok) {
                throw new Error(data.detail || 'Login failed');
            }

            state.token = data.access_token;
            state.role = data.role;
            localStorage.setItem(TOKEN_KEY, state.token);
            localStorage.setItem(ROLE_KEY, state.role);

            hideLoginModal();
            await initAuthenticatedSession();
        } catch (err) {
            showStatus('loginError', err.message, 'error');
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Sign In';
        }
    }

    async function initAuthenticatedSession() {
        try {
            const me = await api('/auth/me');
            state.username = me.username;
            state.role = me.role;

            const userHeader = $('userHeaderSection');
            if (userHeader) userHeader.classList.remove('hidden');
            const userBadge = $('userBadge');
            if (userBadge) {
                userBadge.textContent = `${state.username} (${state.role})`;
                userBadge.className = 'badge ' + (state.role === 'admin' ? 'badge-primary' : 'badge-info');
            }

            applyRoleVisibility();

            if (state.role === 'admin') {
                loadDocuments();
                loadSchemas();
                loadPipelineStatus();
                loadJobs();
            } else {
                loadUserDocuments();
                loadUserJobs();
            }

            if (!state.sessionId) {
                newSession(false);
            }
        } catch (err) {
            console.warn('initAuthenticatedSession failed', err);
            showLoginModal();
        }
    }

    function applyRoleVisibility() {
        const isAdmin = state.role === 'admin';

        document.querySelectorAll('.admin-only').forEach(el => {
            if (isAdmin) el.classList.remove('hidden');
            else el.classList.add('hidden');
        });

        document.querySelectorAll('.user-only').forEach(el => {
            if (!isAdmin) el.classList.remove('hidden');
            else el.classList.add('hidden');
        });
    }

    // ======================= Theme Toggle =======================
    const THEME_KEY = 'idp_console_theme';

    function initTheme() {
        const saved = localStorage.getItem(THEME_KEY) || 'dark';
        applyTheme(saved);

        const btn = $('themeToggleBtn');
        if (btn) {
            btn.addEventListener('click', () => {
                const current = document.documentElement.getAttribute('data-theme') || 'dark';
                const next = current === 'dark' ? 'light' : 'dark';
                applyTheme(next);
            });
        }
    }

    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem(THEME_KEY, theme);
        const icon = $('themeIcon');
        const label = $('themeLabel');
        if (icon) icon.textContent = theme === 'dark' ? '☀️' : '🌙';
        if (label) label.textContent = theme === 'dark' ? 'Light Mode' : 'Dark Mode';
    }

    // ======================= Tabs =======================

    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            const targetPanel = $('tab-' + btn.dataset.tab);
            if (targetPanel) targetPanel.classList.add('active');

            if (btn.dataset.tab === 'chatbot') {
                if (state.role === 'admin') {
                    loadSchemas();
                    loadPipelineStatus();
                    loadJobs();
                    if (state.docs.length === 0) loadDocuments();
                    else renderDocChecklist();
                } else {
                    loadUserJobs();
                }
            } else if (btn.dataset.tab === 'documents') {
                loadDocuments();
            } else if (btn.dataset.tab === 'user-documents') {
                loadUserDocuments();
            } else if (btn.dataset.tab === 'logs') {
                loadUserActivityLogs();
            } else if (btn.dataset.tab === 'users') {
                loadUsersList();
            } else if (btn.dataset.tab === 'querybot') {
                loadAllExtractedData();
            }
        });
    });

    // ======================= Health / Boot =======================

    async function checkHealth() {
        try {
            const data = await api('/health');
            $('healthBadge').textContent = 'online';
            $('healthBadge').className = 'badge badge-success';
            $('llmBadge').textContent = 'LLM: ' + (data.llm_provider || 'unknown');
        } catch (e) {
            $('healthBadge').textContent = 'offline';
            $('healthBadge').className = 'badge badge-danger';
            $('llmBadge').textContent = '--';
        }
    }

    // ======================= Admin Documents Tab =======================

    async function loadDocuments() {
        if (state.role !== 'admin') return;
        try {
            const data = await api('/documents');
            state.docs = data.documents || [];
            const outs = data.outputs || [];
            if ($('docCount')) $('docCount').textContent = state.docs.length;
            if ($('outCount')) $('outCount').textContent = outs.length;
            renderDocList(state.docs, outs);
            renderDocChecklist();
        } catch (e) {
            if ($('docList')) $('docList').innerHTML = `<p class="muted">Failed to load: ${e.message}</p>`;
        }
    }

    function renderDocList(docs, outs) {
        const docList = $('docList');
        if (!docList) return;
        if (!docs.length) {
            docList.innerHTML = '<p class="muted">No documents yet. Upload PDFs above.</p>';
        } else {
            docList.innerHTML = '';
            docs.forEach(d => {
                const row = el('div', 'doc-item');
                const info = el('div', 'doc-item-info');
                info.innerHTML = `
                    <span class="doc-icon">📄</span>
                    <div>
                        <div class="doc-name">${escapeHtml(d.name)}</div>
                        <div class="doc-meta">${fmtSize(d.size)} • ${new Date(d.modified).toLocaleString()}</div>
                    </div>
                `;
                const badge = d.has_output
                    ? '<span class="badge doc-badge badge-success">processed</span>'
                    : '<span class="badge doc-badge badge-mute">pending</span>';
                const actions = el('div');
                actions.innerHTML = badge;
                row.appendChild(info);
                row.appendChild(actions);
                docList.appendChild(row);
            });
        }

        const outList = $('outList');
        if (!outList) return;
        if (!outs.length) {
            outList.innerHTML = '<p class="muted">No processed outputs yet. Run the pipeline.</p>';
        } else {
            outList.innerHTML = '';
            outs.forEach(o => {
                const row = el('div', 'doc-item');
                const info = el('div', 'doc-item-info');
                const schema = o.schema_ref ? ` • schema: <code>${escapeHtml(o.schema_ref.schema_id || '--')}</code>` : '';
                info.innerHTML = `
                    <span class="doc-icon">📝</span>
                    <div>
                        <div class="doc-name">${escapeHtml(o.name)}</div>
                        <div class="doc-meta">${fmtSize(o.size)} • ${new Date(o.modified).toLocaleString()}${schema}</div>
                    </div>
                `;
                row.appendChild(info);
                outList.appendChild(row);
            });
        }
    }

    // ======================= User Documents Tab =======================

    async function loadUserDocuments() {
        try {
            const data = await api('/me/documents');
            state.userDocs = data.documents || [];
            if ($('userDocCount')) $('userDocCount').textContent = state.userDocs.length;
            renderUserDocList(state.userDocs);
        } catch (e) {
            if ($('userDocList')) $('userDocList').innerHTML = `<p class="muted">Failed to load: ${e.message}</p>`;
        }
    }

    function renderUserDocList(docs) {
        const host = $('userDocList');
        if (!host) return;
        if (!docs.length) {
            host.innerHTML = '<p class="muted">No private documents yet. Upload PDFs above.</p>';
        } else {
            host.innerHTML = '';
            docs.forEach(d => {
                const row = el('div', 'doc-item');
                const info = el('div', 'doc-item-info');
                info.innerHTML = `
                    <span class="doc-icon">📄</span>
                    <div>
                        <div class="doc-name">${escapeHtml(d.name)}</div>
                        <div class="doc-meta">${fmtSize(d.size)} • ${new Date(d.modified).toLocaleString()}</div>
                    </div>
                `;
                row.appendChild(info);
                host.appendChild(row);
            });
        }
    }

    // ======================= Upload Handlers =======================

    (function setupUploads() {
        // Admin upload
        const zone = $('uploadZone');
        const input = $('fileInput');
        const browse = $('browseBtn');

        if (zone && input && browse) {
            function handleFiles(files) {
                const pdfs = Array.from(files).filter(f => f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf'));
                if (!pdfs.length) return;
                uploadAdminFiles(pdfs);
            }

            zone.addEventListener('click', (e) => {
                if (e.target.tagName !== 'A') { input.click(); e.preventDefault(); }
            });
            browse.addEventListener('click', (e) => { input.click(); e.preventDefault(); });
            input.addEventListener('change', (e) => handleFiles(e.target.files));
            zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
            zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
            zone.addEventListener('drop', (e) => {
                e.preventDefault();
                zone.classList.remove('dragover');
                handleFiles(e.dataTransfer.files);
            });

            async function uploadAdminFiles(files) {
                const fd = new FormData();
                files.forEach(f => fd.append('files', f, f.name));
                showStatus('uploadStatus', `Uploading ${files.length} file(s)...`);
                try {
                    const res = await api('/documents/upload', { method: 'POST', body: fd });
                    showStatus('uploadStatus', `Saved ${res.count} file(s): ${res.saved.join(', ')}`, 'success');
                    loadDocuments();
                } catch (e) {
                    showStatus('uploadStatus', 'Upload failed: ' + e.message, 'error');
                }
                setTimeout(() => hideStatus('uploadStatus'), 4000);
            }
        }

        // User private upload
        const uZone = $('userUploadZone');
        const uInput = $('userFileInput');
        const uBrowse = $('userBrowseBtn');

        if (uZone && uInput && uBrowse) {
            function handleUserFiles(files) {
                const pdfs = Array.from(files).filter(f => f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf'));
                if (!pdfs.length) return;
                uploadUserFiles(pdfs);
            }

            uZone.addEventListener('click', (e) => {
                if (e.target.tagName !== 'A') { uInput.click(); e.preventDefault(); }
            });
            uBrowse.addEventListener('click', (e) => { uInput.click(); e.preventDefault(); });
            uInput.addEventListener('change', (e) => handleUserFiles(e.target.files));
            uZone.addEventListener('dragover', (e) => { e.preventDefault(); uZone.classList.add('dragover'); });
            uZone.addEventListener('dragleave', () => uZone.classList.remove('dragover'));
            uZone.addEventListener('drop', (e) => {
                e.preventDefault();
                uZone.classList.remove('dragover');
                handleUserFiles(e.dataTransfer.files);
            });

            async function uploadUserFiles(files) {
                const fd = new FormData();
                files.forEach(f => fd.append('files', f, f.name));
                showStatus('userUploadStatus', `Uploading ${files.length} private file(s)...`);
                try {
                    const res = await api('/me/documents', { method: 'POST', body: fd });
                    showStatus('userUploadStatus', `Saved ${res.count} file(s): ${res.saved.join(', ')}`, 'success');
                    loadUserDocuments();
                } catch (e) {
                    showStatus('userUploadStatus', 'Upload failed: ' + e.message, 'error');
                }
                setTimeout(() => hideStatus('userUploadStatus'), 4000);
            }
        }
    })();

    if ($('refreshDocsBtn')) $('refreshDocsBtn').addEventListener('click', loadDocuments);
    if ($('refreshUserDocsBtn')) $('refreshUserDocsBtn').addEventListener('click', loadUserDocuments);

    function showStatus(id, html, kind) {
        const s = $(id);
        if (!s) return;
        s.className = 'status-box' + (kind ? ' ' + kind : '');
        s.innerHTML = html;
        s.classList.remove('hidden');
    }
    function hideStatus(id) { const s = $(id); if (s) s.classList.add('hidden'); }

    // ======================= Chatbot Tab =======================

    function addChatMsg(who, text) {
        const log = $('chatLog');
        if (!log) return;
        const wrap = el('div', 'chat-msg ' + who);
        const b = el('div', 'msg-bubble');
        b.textContent = text;
        wrap.appendChild(b);
        log.appendChild(wrap);
        log.scrollTop = log.scrollHeight;
    }

    function setStateBadge(s, s2) {
        const b = $('stateBadge');
        if (!b) return;
        const map = {
            START: 'badge-info', REVIEW: 'badge-warn', COMPLETED: 'badge-success',
        };
        b.className = 'badge ' + (map[s] || 'badge-mute');
        b.textContent = s.toLowerCase() + (s2 ? ' • ' + s2 : '');
    }

    let schemaSyncTimer = null;

    function collectSchemaFromInputs() {
        if (!state.currentSchema) {
            state.currentSchema = { document_type: '', fields: [] };
        }
        const docTypeInput = $('editDocType');
        if (docTypeInput) {
            state.currentSchema.document_type = docTypeInput.value.trim().toLowerCase().replace(/\s+/g, '_');
        }

        const rows = document.querySelectorAll('#schemaPanel table.schema-table tbody tr');
        const fields = [];
        rows.forEach(tr => {
            const nameInput = tr.querySelector('.field-name-input');
            const typeSelect = tr.querySelector('.field-type-select');
            const reqBtn = tr.querySelector('.req-toggle');
            const descInput = tr.querySelector('.field-desc-input');

            const name = nameInput ? nameInput.value.trim().toLowerCase().replace(/[\s-]+/g, '_') : '';
            if (!name) return;

            let rawType = typeSelect ? typeSelect.value : 'string';
            let itemType = null;
            if (rawType.startsWith('array[')) {
                itemType = rawType.substring(6, rawType.length - 1);
                rawType = 'array';
            }

            fields.push({
                name: name,
                type: rawType,
                item_type: itemType,
                required: reqBtn ? reqBtn.classList.contains('is-req') : true,
                description: descInput ? descInput.value.trim() : ''
            });
        });

        state.currentSchema.fields = fields;
        const jsonView = $('schemaJsonView');
        if (jsonView) {
            jsonView.textContent = JSON.stringify(state.currentSchema, null, 2);
        }
        return state.currentSchema;
    }

    function validateAndRefreshUI() {
        const schema = state.currentSchema;
        const errs = $('schemaErrors');
        if (!schema) {
            if (errs) errs.classList.add('hidden');
            if ($('confirmBtn')) $('confirmBtn').disabled = true;
            return;
        }

        const errors = [];
        if (!schema.document_type) errors.push('document_type is not set');
        if (!schema.fields || !schema.fields.length) errors.push('schema has no fields');
        const seen = new Set();
        (schema.fields || []).forEach(f => {
            if (!f.name) errors.push('field name cannot be blank');
            else if (seen.has(f.name)) errors.push(`duplicate field name: ${f.name}`);
            seen.add(f.name);
            if (f.type === 'array' && !f.item_type) errors.push(`field '${f.name}' is an array but has no item_type`);
        });

        state.currentErrors = errors;
        if (errors.length) {
            if (errs) {
                errs.innerHTML = '<strong>⚠️ Schema issues:</strong><ul>' + errors.map(e => '<li>' + escapeHtml(e) + '</li>').join('') + '</ul>';
                errs.classList.remove('hidden');
            }
        } else {
            if (errs) errs.classList.add('hidden');
        }

        const canConfirm = (state.currentState === 'REVIEW' || state.currentState === 'START') && !errors.length && (schema.fields || []).length > 0 && !!schema.document_type;
        if ($('confirmBtn')) $('confirmBtn').disabled = state.completed || !canConfirm;
    }

    async function syncSchemaToServer() {
        if (!state.sessionId || state.completed) return;
        const syncStatus = $('schemaSyncStatus');
        if (syncStatus) syncStatus.textContent = '⏳ Saving...';
        try {
            const data = await api('/session/' + state.sessionId + '/schema', {
                method: 'POST',
                json: {
                    document_type: state.currentSchema.document_type,
                    fields: state.currentSchema.fields
                }
            });
            state.currentState = data.state;
            state.currentErrors = data.errors || [];
            validateAndRefreshUI();
            if (syncStatus) syncStatus.textContent = '✓ Saved';
            setTimeout(() => { if (syncStatus) syncStatus.textContent = '✓ Interactive Editor'; }, 1500);
        } catch (e) {
            if (syncStatus) syncStatus.textContent = '⚠️ Sync error';
        }
    }

    function scheduleSchemaSync() {
        collectSchemaFromInputs();
        validateAndRefreshUI();
        clearTimeout(schemaSyncTimer);
        schemaSyncTimer = setTimeout(syncSchemaToServer, 500);
    }

    function addNewSchemaField() {
        if (!state.currentSchema) {
            state.currentSchema = { document_type: 'document', fields: [] };
        }
        const count = (state.currentSchema.fields || []).length + 1;
        state.currentSchema.fields.push({
            name: 'field_' + count,
            type: 'string',
            required: true,
            description: ''
        });
        renderSchemaPanel();
        scheduleSchemaSync();
        setTimeout(() => {
            const inputs = document.querySelectorAll('.field-name-input');
            if (inputs.length) {
                inputs[inputs.length - 1].focus({ preventScroll: true });
                inputs[inputs.length - 1].select();
            }
        }, 50);
    }

    function renderSchemaPanel() {
        const schema = state.currentSchema;
        const panel = $('schemaPanel');
        const errs = $('schemaErrors');
        const addBtn = $('addSchemaFieldBtn');

        if (!panel) return;

        if (!schema) {
            panel.innerHTML = '<p class="muted">No schema yet. Start a session, upload samples, or click "+ Add Field".</p>';
            if (errs) errs.classList.add('hidden');
            if ($('copySchemaBtn')) $('copySchemaBtn').disabled = true;
            if ($('printSchemaBtn')) $('printSchemaBtn').disabled = true;
            if ($('downloadSchemaPdfHeaderBtn')) $('downloadSchemaPdfHeaderBtn').disabled = true;
            if ($('downloadSchemaJsonHeaderBtn')) $('downloadSchemaJsonHeaderBtn').disabled = true;
            if ($('confirmBtn')) $('confirmBtn').disabled = true;
            if (addBtn) addBtn.disabled = !state.sessionId;
            return;
        }

        if ($('copySchemaBtn')) $('copySchemaBtn').disabled = false;
        if ($('printSchemaBtn')) $('printSchemaBtn').disabled = false;
        const canDownloadSchema = !!(state.completed || state.lastConfirmedSchemaId);
        if ($('downloadSchemaPdfHeaderBtn')) $('downloadSchemaPdfHeaderBtn').disabled = !canDownloadSchema;
        if ($('downloadSchemaJsonHeaderBtn')) $('downloadSchemaJsonHeaderBtn').disabled = !canDownloadSchema;
        if (addBtn) addBtn.disabled = state.completed;

        const docType = schema.document_type || '';
        const fields = schema.fields || [];

        const typeOptions = [
            'string', 'number', 'integer', 'boolean', 'date',
            'array[string]', 'array[object]', 'object'
        ];

        let html = `
            <div class="doc-type-edit-row">
                <label for="editDocType">Document Type:</label>
                <input id="editDocType" class="doc-type-input" type="text" value="${escapeHtml(docType)}" placeholder="e.g. resume, invoice, insurance_claim" ${state.completed ? 'disabled' : ''}>
            </div>
        `;

        if (!fields.length) {
            html += `
                <div style="padding: 16px; text-align: center;">
                    <p class="muted small">No fields defined yet.</p>
                    <button type="button" class="btn btn-outline btn-sm" id="addFieldInlineBtn" ${state.completed ? 'disabled' : ''}>➕ Add First Field</button>
                </div>
            `;
        } else {
            html += `
                <table class="schema-table">
                    <thead>
                        <tr>
                            <th style="width: 28%;">Field Name</th>
                            <th style="width: 26%;">Type</th>
                            <th style="width: 14%; text-align: center;">Req</th>
                            <th style="width: 26%;">Description</th>
                            <th style="width: 6%; text-align: center;"></th>
                        </tr>
                    </thead>
                    <tbody>
            `;

            fields.forEach((f, idx) => {
                const curType = f.type === 'array' ? (f.item_type ? `array[${f.item_type}]` : 'array[string]') : (f.type || 'string');
                const isReq = !!f.required;
                const opts = typeOptions.map(t => `<option value="${t}" ${t === curType ? 'selected' : ''}>${t}</option>`).join('');

                html += `
                    <tr data-idx="${idx}">
                        <td>
                            <input type="text" class="schema-input field-name-input" data-field="name" value="${escapeHtml(f.name || '')}" placeholder="field_name" ${state.completed ? 'disabled' : ''}>
                        </td>
                        <td>
                            <select class="schema-select field-type-select" data-field="type" ${state.completed ? 'disabled' : ''}>
                                ${opts}
                            </select>
                        </td>
                        <td style="text-align: center;">
                            <button type="button" class="req-toggle ${isReq ? 'is-req' : 'is-opt'}" data-idx="${idx}" ${state.completed ? 'disabled' : ''}>
                                ${isReq ? 'YES' : 'NO'}
                            </button>
                        </td>
                        <td>
                            <input type="text" class="schema-input field-desc-input" data-field="description" value="${escapeHtml(f.description || '')}" placeholder="Description..." ${state.completed ? 'disabled' : ''}>
                        </td>
                        <td style="text-align: center;">
                            <button type="button" class="btn-del-field" data-idx="${idx}" title="Delete field" ${state.completed ? 'disabled' : ''}>✕</button>
                        </td>
                    </tr>
                `;
            });

            html += `
                    </tbody>
                </table>
                <div class="schema-bottom-actions">
                    <button type="button" class="btn btn-ghost btn-sm" id="addFieldInlineBtn" ${state.completed ? 'disabled' : ''}>➕ Add Field</button>
                    <span id="schemaSyncStatus" class="sync-badge">✓ Interactive Editor</span>
                </div>
            `;
        }

        html += `<details style="margin-top:12px"><summary class="muted small" style="cursor:pointer">View Full JSON Schema</summary><pre id="schemaJsonView">${escapeHtml(JSON.stringify(schema, null, 2))}</pre></details>`;
        panel.innerHTML = html;

        validateAndRefreshUI();
    }

    function handleChatResponse(data) {
        state.sessionId = data.session_id;
        state.currentState = data.state;
        state.currentSchema = data.schema;
        state.currentErrors = data.errors || [];
        state.completed = !!data.completed;
        if (data.completed && data.schema_id) {
            state.lastConfirmedSchemaId = data.schema_id;
        }
        if (data.message) addChatMsg('bot', data.message);
        setStateBadge(data.state, data.completed ? 'confirmed' : '');
        renderSchemaPanel();
        if ($('chatInput')) $('chatInput').disabled = false;
        if ($('sendBtn')) $('sendBtn').disabled = false;
        if (data.completed) {
            showStatus('confirmStatus', `🎉 Schema confirmed! schema_id = <code>${escapeHtml(data.schema_id)}</code>. Saved to schema_registry/.`, 'success');
            if ($('postConfirmBox')) {
                $('postConfirmBox').classList.remove('hidden');
                $('quickRunPipelineBtn').disabled = false;
            }
            if (state.role === 'admin') {
                setTimeout(() => {
                    loadSchemas();
                    loadPipelineStatus();
                }, 300);
            }
        } else {
            if ($('postConfirmBox')) $('postConfirmBox').classList.add('hidden');
        }
    }

    async function newSession(shouldFocus = true) {
        try {
            state.lastConfirmedSchemaId = null;
            if ($('postConfirmBox')) $('postConfirmBox').classList.add('hidden');
            hideStatus('confirmStatus');
            const data = await api('/chat', { method: 'POST', json: { session_id: null, message: null } });
            state.sessionId = data.session_id;
            if ($('chatLog')) $('chatLog').innerHTML = '';
            handleChatResponse(data);
            if (shouldFocus && $('chatInput')) {
                $('chatInput').focus({ preventScroll: true });
            }
        } catch (e) {
            addChatMsg('bot', 'Failed to start session: ' + e.message);
        }
    }

    if ($('newSessionBtn')) $('newSessionBtn').addEventListener('click', () => newSession(true));

    async function sendChat() {
        const input = $('chatInput');
        const msg = input ? input.value.trim() : '';
        if (!msg || !state.sessionId) return;
        input.value = '';
        addChatMsg('user', msg);
        input.disabled = true;
        if ($('sendBtn')) $('sendBtn').disabled = true;
        try {
            const data = await api('/chat', {
                method: 'POST',
                json: { session_id: state.sessionId, message: msg }
            });
            handleChatResponse(data);
        } catch (e) {
            addChatMsg('bot', 'Error: ' + e.message);
            if (input) input.disabled = false;
            if ($('sendBtn')) $('sendBtn').disabled = false;
        }
    }

    if ($('sendBtn')) $('sendBtn').addEventListener('click', sendChat);
    if ($('chatInput')) {
        $('chatInput').addEventListener('keydown', (e) => {
            if (e.key === 'Enter') sendChat();
        });
    }

    if ($('printSchemaBtn')) {
        $('printSchemaBtn').addEventListener('click', async () => {
            if (!state.sessionId) return;
            try {
                const data = await api('/session/' + state.sessionId);
                state.currentSchema = data.schema;
                state.currentState = data.state;
                state.completed = data.completed;
                renderSchemaPanel();
            } catch (e) {
                addChatMsg('bot', 'Error re-print schema: ' + e.message);
            }
        });
    }

    if ($('copySchemaBtn')) {
        $('copySchemaBtn').addEventListener('click', async () => {
            if (!state.currentSchema) return;
            try {
                await navigator.clipboard.writeText(JSON.stringify(state.currentSchema, null, 2));
                const old = $('copySchemaBtn').textContent;
                $('copySchemaBtn').textContent = '✓ Copied!';
                setTimeout(() => { $('copySchemaBtn').textContent = old; }, 1500);
            } catch (e) {
                addChatMsg('bot', 'Copy failed: ' + e.message);
            }
        });
    }

    if ($('confirmBtn')) {
        $('confirmBtn').addEventListener('click', async () => {
            if (!state.sessionId || state.completed) return;
            if ($('chatInput')) $('chatInput').value = '/confirm';
            sendChat();
        });
    }

    if ($('downloadSchemaPdfHeaderBtn')) {
        $('downloadSchemaPdfHeaderBtn').addEventListener('click', () => downloadSchemaPdf());
    }
    if ($('downloadSchemaJsonHeaderBtn')) {
        $('downloadSchemaJsonHeaderBtn').addEventListener('click', () => downloadSchemaJson());
    }
    if ($('downloadSchemaPdfBtn')) {
        $('downloadSchemaPdfBtn').addEventListener('click', () => downloadSchemaPdf());
    }
    if ($('downloadSchemaJsonBtn')) {
        $('downloadSchemaJsonBtn').addEventListener('click', () => downloadSchemaJson());
    }

    if ($('addSchemaFieldBtn')) {
        $('addSchemaFieldBtn').addEventListener('click', addNewSchemaField);
    }

    if ($('schemaPanel')) {
        $('schemaPanel').addEventListener('input', (e) => {
            if (e.target.matches('#editDocType, .field-name-input, .field-desc-input')) {
                scheduleSchemaSync();
            }
        });

        $('schemaPanel').addEventListener('change', (e) => {
            if (e.target.matches('.field-type-select')) {
                scheduleSchemaSync();
            }
        });

        $('schemaPanel').addEventListener('click', (e) => {
            const toggleBtn = e.target.closest('.req-toggle');
            if (toggleBtn) {
                const isReq = toggleBtn.classList.contains('is-req');
                toggleBtn.classList.toggle('is-req', !isReq);
                toggleBtn.classList.toggle('is-opt', isReq);
                toggleBtn.textContent = !isReq ? 'YES' : 'NO';
                scheduleSchemaSync();
                return;
            }

            const delBtn = e.target.closest('.btn-del-field');
            if (delBtn) {
                const idx = parseInt(delBtn.getAttribute('data-idx'), 10);
                if (!isNaN(idx) && state.currentSchema && state.currentSchema.fields) {
                    state.currentSchema.fields.splice(idx, 1);
                    renderSchemaPanel();
                    scheduleSchemaSync();
                }
                return;
            }

            if (e.target.matches('#addFieldInlineBtn')) {
                addNewSchemaField();
            }
        });
    }

    // Sample inference upload
    (function setupInferUpload() {
        const input = $('inferInput');
        const browse = $('inferBrowseBtn');
        const sel = $('inferSelected');
        const run = $('inferBtn');
        let files = [];

        if (!input || !browse || !run) return;

        browse.addEventListener('click', () => input.click());
        input.addEventListener('change', (e) => {
            files = Array.from(e.target.files).filter(f =>
                f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf'));
            sel.textContent = files.length ? `${files.length} file(s): ${files.map(f => f.name).join(', ')}` : 'no files';
            run.disabled = !(files.length >= 2 && files.length <= 5);
        });

        run.addEventListener('click', async () => {
            if (!(files.length >= 2 && files.length <= 5)) return;
            const fd = new FormData();
            files.forEach(f => fd.append('files', f, f.name));
            if (state.sessionId) fd.append('session_id', state.sessionId);

            showStatus('inferStatus', 'Running Sarvam Doc AI + schema inference... expect 60-180s for 2 PDFs. Please be patient.', 'warn');
            run.disabled = true;
            const t0 = Date.now();
            try {
                const data = await api('/schema/infer', { method: 'POST', body: fd, timeout: 0 });
                const secs = (Date.now() - t0) / 1000;
                if ($('chatLog') && !$('chatLog').children.length) {
                    $('chatLog').innerHTML = '';
                }
                addChatMsg('bot', `[Inference returned in ${secs.toFixed(1)}s]`);
                handleChatResponse(data);
                showStatus('inferStatus', `Inference complete (${secs.toFixed(1)}s).`, 'success');
                setTimeout(() => hideStatus('inferStatus'), 5000);
            } catch (e) {
                showStatus('inferStatus', 'Inference failed: ' + e.message, 'error');
            } finally {
                run.disabled = !(files.length >= 2 && files.length <= 5);
            }
        });
    })();

    // ======================= Admin Pipeline Section =======================

    function renderDocChecklist() {
        const host = $('docChecklist');
        const allBtn = $('selAllBtn');
        const noneBtn = $('selNoneBtn');
        if (!host) return;
        if (!state.docs.length) {
            host.innerHTML = '<p class="muted small">Load documents from the Documents tab first.</p>';
            if (allBtn) allBtn.disabled = true;
            if (noneBtn) noneBtn.disabled = true;
            return;
        }
        if (allBtn) allBtn.disabled = false;
        if (noneBtn) noneBtn.disabled = false;
        host.innerHTML = '';
        state.docs.forEach(d => {
            const row = el('label', 'check-item');
            row.innerHTML = `
                <input type="checkbox" class="doc-check" value="${escapeHtml(d.name)}" checked>
                <span class="cname">${escapeHtml(d.name)}</span>
                <span class="cmeta">${fmtSize(d.size)}</span>
                <span class="cmeta">${d.has_output ? 're-run' : 'pending'}</span>
            `;
            host.appendChild(row);
        });
        updateRunBtn();
    }

    if ($('selAllBtn')) {
        $('selAllBtn').addEventListener('click', () => {
            document.querySelectorAll('.doc-check').forEach(c => c.checked = true);
            updateRunBtn();
        });
    }
    if ($('selNoneBtn')) {
        $('selNoneBtn').addEventListener('click', () => {
            document.querySelectorAll('.doc-check').forEach(c => c.checked = false);
            updateRunBtn();
        });
    }

    function updateRunBtn() {
        const runBtn = $('runPipelineBtn');
        if (!runBtn) return;
        const anyChecked = document.querySelectorAll('.doc-check:checked').length > 0;
        const sel = $('schemaSelect');
        runBtn.disabled = !(anyChecked && state.pipelineAvailable && sel && !!sel.value);
    }

    if ($('docChecklist')) $('docChecklist').addEventListener('change', updateRunBtn);

    async function loadSchemas() {
        if (state.role !== 'admin') return;
        try {
            const data = await api('/schemas');
            state.schemas = data.schemas || [];
            const sel = $('schemaSelect');
            if (!sel) return;
            if (!state.schemas.length) {
                sel.innerHTML = '<option value="">-- no confirmed schemas --</option>';
                sel.disabled = true;
            } else {
                sel.innerHTML = '<option value="">-- select a schema --</option>' +
                    state.schemas.map(s => `<option value="${escapeHtml(s.schema_id)}">
                        ${escapeHtml(s.document_type || 'untitled')} • ${s.field_count} fields • ${escapeHtml(s.schema_id.slice(0, 12))}
                    </option>`).join('');
                sel.disabled = false;
            }
            sel.removeEventListener('change', updateRunBtn);
            sel.addEventListener('change', updateRunBtn);
            updateRunBtn();
        } catch (e) {
            console.warn('loadSchemas failed', e);
        }
    }

    if ($('refreshSchemasBtn')) $('refreshSchemasBtn').addEventListener('click', loadSchemas);

    async function loadPipelineStatus() {
        if (state.role !== 'admin') return;
        try {
            const data = await api('/pipeline/status');
            state.pipelineAvailable = !!data.available;
            const badge = $('pipelineAvail');
            if (badge) {
                if (data.available) {
                    badge.textContent = '✓ available • ' + (data.routing_mode || '');
                    badge.className = 'badge badge-success';
                } else {
                    badge.textContent = '✕ unavailable';
                    badge.className = 'badge badge-danger';
                }
            }
            updateRunBtn();
        } catch (e) {
            console.warn('pipeline status failed', e);
        }
    }

    function jobStatusBadgeClass(s) {
        return {
            queued: 'badge-warn',
            running: 'badge-info',
            paused: 'badge-warn',
            killed: 'badge-danger',
            completed: 'badge-success'
        }[s] || 'badge-mute';
    }

    async function pauseJob(jobId, e) {
        if (e) e.stopPropagation();
        try {
            await api(`/pipeline/jobs/${jobId}/pause`, { method: 'POST' });
            loadJobs();
            if (state.selectedJobId === jobId) loadJobDetail(jobId);
        } catch (err) {
            alert('Failed to pause job: ' + err.message);
        }
    }

    async function resumeJob(jobId, e) {
        if (e) e.stopPropagation();
        try {
            await api(`/pipeline/jobs/${jobId}/resume`, { method: 'POST' });
            loadJobs();
            if (state.selectedJobId === jobId) loadJobDetail(jobId);
        } catch (err) {
            alert('Failed to resume job: ' + err.message);
        }
    }

    async function killJob(jobId, e) {
        if (e) e.stopPropagation();
        if (!confirm(`Are you sure you want to kill / cancel job ${jobId}?`)) return;
        try {
            await api(`/pipeline/jobs/${jobId}/kill`, { method: 'POST' });
            loadJobs();
            if (state.selectedJobId === jobId) loadJobDetail(jobId);
        } catch (err) {
            alert('Failed to kill job: ' + err.message);
        }
    }

    async function loadJobs() {
        if (state.role !== 'admin') return;
        try {
            const data = await api('/pipeline/status');
            const jobs = Object.values(data.jobs || {});
            state.jobs = jobs;
            const host = $('jobList');
            if (!host) return;
            if (!jobs.length) {
                host.innerHTML = '<p class="muted">No jobs yet.</p>';
                return;
            }
            host.innerHTML = '';
            jobs.sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''));
            jobs.forEach(j => {
                const done = j.completed || 0;
                const total = j.total || 0;
                const pct = total ? Math.round(100 * done / total) : 0;
                const row = el('div', 'job-item');

                let actionButtons = '';
                if (j.status === 'running' || j.status === 'queued') {
                    actionButtons = `
                        <button class="btn btn-ghost btn-sm" title="Pause job" data-action="pause" data-job="${escapeHtml(j.job_id)}">⏸</button>
                        <button class="btn btn-ghost btn-sm" title="Kill job" style="color:#fca5a5" data-action="kill" data-job="${escapeHtml(j.job_id)}">✕</button>
                    `;
                } else if (j.status === 'paused') {
                    actionButtons = `
                        <button class="btn btn-ghost btn-sm" title="Resume job" style="color:#6ee7b7" data-action="resume" data-job="${escapeHtml(j.job_id)}">▶</button>
                        <button class="btn btn-ghost btn-sm" title="Kill job" style="color:#fca5a5" data-action="kill" data-job="${escapeHtml(j.job_id)}">✕</button>
                    `;
                }

                row.innerHTML = `
                    <div class="job-item-left">
                        <div class="job-id">${escapeHtml(j.job_id)}</div>
                        <div class="job-meta">${escapeHtml(j.created_at || '')} • schema: ${escapeHtml(j.schema_id || '--')} • ${done}/${total} docs</div>
                    </div>
                    <div class="job-item-right">
                        <div class="progress-bar-wrap"><div class="progress-bar-fill" style="width:${pct}%"></div></div>
                        <span class="badge ${jobStatusBadgeClass(j.status)}">${escapeHtml(j.status)}</span>
                        <div class="job-item-actions">${actionButtons}</div>
                    </div>
                `;
                row.addEventListener('click', (e) => {
                    const btn = e.target.closest('button[data-action]');
                    if (btn) {
                        e.stopPropagation();
                        const act = btn.dataset.action;
                        const jid = btn.dataset.job;
                        if (act === 'pause') pauseJob(jid, e);
                        else if (act === 'resume') resumeJob(jid, e);
                        else if (act === 'kill') killJob(jid, e);
                        return;
                    }
                    loadJobDetail(j.job_id, true);
                });
                host.appendChild(row);
            });
            if (state.selectedJobId) loadJobDetail(state.selectedJobId, false);
        } catch (e) {
            console.warn('loadJobs failed', e);
        }
    }

    async function pauseUserJob(jobId, e) {
        if (e) e.stopPropagation();
        try {
            await api(`/me/pipeline/jobs/${jobId}/pause`, { method: 'POST' });
            loadUserJobs();
            loadJobDetail(jobId);
        } catch (err) {
            alert('Pause failed: ' + err.message);
        }
    }

    async function resumeUserJob(jobId, e) {
        if (e) e.stopPropagation();
        try {
            await api(`/me/pipeline/jobs/${jobId}/resume`, { method: 'POST' });
            loadUserJobs();
            loadJobDetail(jobId);
        } catch (err) {
            alert('Resume failed: ' + err.message);
        }
    }

    async function killUserJob(jobId, e) {
        if (e) e.stopPropagation();
        if (!confirm(`Cancel job ${jobId}?`)) return;
        try {
            await api(`/me/pipeline/jobs/${jobId}/kill`, { method: 'POST' });
            loadUserJobs();
            loadJobDetail(jobId);
        } catch (err) {
            alert('Kill failed: ' + err.message);
        }
    }

    async function loadJobDetail(jobId, shouldScroll = false) {
        if (!jobId || jobId === 'undefined') return;
        state.selectedJobId = jobId;
        try {
            const apiPath = state.role === 'user' ? ('/me/pipeline/jobs/' + jobId) : ('/pipeline/jobs/' + jobId);
            const j = await api(apiPath);
            window.__currentJobDetail = j;

            const isUser = state.role === 'user';
            const card = isUser ? ($('userJobDetailCard') || $('jobDetailCard')) : $('jobDetailCard');
            if (card) {
                card.classList.remove('hidden');
                if (shouldScroll) {
                    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            }
            const idSpan = isUser ? ($('userDetailJobId') || $('detailJobId')) : $('detailJobId');
            if (idSpan) idSpan.textContent = jobId;
            const st = isUser ? ($('userDetailStatus') || $('detailStatus')) : $('detailStatus');
            if (st) {
                st.textContent = j.status;
                st.className = 'badge ' + jobStatusBadgeClass(j.status);
            }

            const pauseBtn = isUser ? ($('userJobPauseBtn') || $('jobPauseBtn')) : $('jobPauseBtn');
            const resumeBtn = isUser ? ($('userJobResumeBtn') || $('jobResumeBtn')) : $('jobResumeBtn');
            const killBtn = isUser ? ($('userJobKillBtn') || $('jobKillBtn')) : $('jobKillBtn');

            if (pauseBtn && resumeBtn && killBtn) {
                if (j.status === 'running' || j.status === 'queued') {
                    pauseBtn.classList.remove('hidden');
                    resumeBtn.classList.add('hidden');
                    killBtn.classList.remove('hidden');
                } else if (j.status === 'paused') {
                    pauseBtn.classList.add('hidden');
                    resumeBtn.classList.remove('hidden');
                    killBtn.classList.remove('hidden');
                } else {
                    pauseBtn.classList.add('hidden');
                    resumeBtn.classList.add('hidden');
                    killBtn.classList.add('hidden');
                }

                pauseBtn.onclick = (e) => isUser ? pauseUserJob(jobId, e) : pauseJob(jobId, e);
                resumeBtn.onclick = (e) => isUser ? resumeUserJob(jobId, e) : resumeJob(jobId, e);
                killBtn.onclick = (e) => isUser ? killUserJob(jobId, e) : killJob(jobId, e);
            }

            const host = isUser ? ($('userJobDetail') || $('jobDetail')) : $('jobDetail');
            if (!host) return;

            const newJsonStr = JSON.stringify(j);
            // Skip full DOM rebuild if this job's data is identical to what's already rendered
            if (window.__renderedJobId === j.job_id && window.__renderedJobJson === newJsonStr && host.children.length > 0) {
                return;
            }

            // Capture current scroll positions before replacing innerHTML
            const fullJsonEl = document.getElementById('fullJsonView');
            const preScrollTop = fullJsonEl ? fullJsonEl.scrollTop : 0;
            const preScrollLeft = fullJsonEl ? fullJsonEl.scrollLeft : 0;
            const hostScrollTop = host.scrollTop;

            const wall = j.wall_time_s ? `${j.wall_time_s.toFixed(1)}s` : '--';
            const sucs = j.successes || [];
            const fails = j.failures || [];

            host.innerHTML = `
                <div class="job-summary-grid">
                    <div class="summary-box"><div class="lbl">Status</div><div class="val">${escapeHtml(j.status)}</div></div>
                    <div class="summary-box"><div class="lbl">Docs</div><div class="val">${sucs.length + fails.length} / ${(j.targets || []).length || 0}</div></div>
                    <div class="summary-box"><div class="lbl">Success</div><div class="val" style="color:#6ee7b7">${sucs.length}</div></div>
                    <div class="summary-box"><div class="lbl">Failed</div><div class="val" style="color:#fca5a5">${fails.length}</div></div>
                    <div class="summary-box"><div class="lbl">Wall time</div><div class="val">${wall}</div></div>
                </div>
                ${sucs.length ? `
                <div class="job-section">
                    <h3>✓ Successful (${sucs.length})</h3>
                    ${sucs.map((s, idx) => `
                        <div class="result-row ok" style="flex-direction: column; align-items: stretch; gap: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span class="result-name">${escapeHtml(s.pdf)}</span>
                                    <div class="result-meta">pages: ${s.pages} • conf: ${(s.avg_conf || 0).toFixed(3)} • Layer 1+2: ${s.elapsed_s}s ${s.extract_elapsed_s ? `• Layer 3 (Sarvam 105B): ${s.extract_elapsed_s}s` : ''}</div>
                                </div>
                                <div style="display: flex; gap: 6px; align-items: center;">
                                    ${s.db_run_id ? `<span class="badge badge-success">PostgreSQL: Run #${s.db_run_id}</span>` : ''}
                                    <span class="result-meta"><code>${escapeHtml(s.md)}</code></span>
                                </div>
                            </div>
                            ${s.extracted_data ? `
                            <div class="extracted-json-box">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                    <strong class="extracted-json-title">⚡ Layer 3 Extracted JSON (${escapeHtml(s.extracted_json || 'record.json')})</strong>
                                    <button class="btn btn-sm btn-ghost" onclick="navigator.clipboard.writeText(JSON.stringify(${escapeHtml(JSON.stringify(s.extracted_data))}, null, 2)); this.textContent='✓ Copied!'; setTimeout(()=>this.textContent='Copy JSON', 1500)">Copy JSON</button>
                                </div>
                                <pre class="extracted-json-pre">${escapeHtml(JSON.stringify(s.extracted_data, null, 2))}</pre>
                            </div>
                            ` : ''}
                        </div>
                    `).join('')}
                </div>
                ` : ''}
                ${fails.length ? `
                <div class="job-section">
                    <h3>✕ Failed (${fails.length})</h3>
                    ${fails.map(f => `
                        <div class="result-row fail">
                            <div>
                                <span class="result-name">${escapeHtml(f.pdf)}</span>
                                <div class="result-error">${escapeHtml(f.error_type)}: ${escapeHtml(f.error)}</div>
                            </div>
                            <div class="result-meta">${f.elapsed_s}s</div>
                        </div>
                    `).join('')}
                </div>
                ` : ''}

                <div class="job-section">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; flex-wrap:wrap; gap:8px;">
                        <h3 style="margin:0;">Full JSON</h3>
                        <div style="display:flex; gap:8px; align-items:center;">
                            <button class="btn btn-sm btn-ghost" onclick="navigator.clipboard.writeText(JSON.stringify(window.__currentJobDetail, null, 2)); this.textContent='✓ Copied!'; setTimeout(()=>this.textContent='📋 Copy JSON', 1500)">📋 Copy JSON</button>
                            <button class="btn btn-sm btn-outline" onclick="downloadJobJson()">📥 Download JSON</button>
                            <button class="btn btn-sm btn-primary" onclick="downloadJobPdf('${j.job_id}')">📄 Download PDF</button>
                        </div>
                    </div>
                    <pre id="fullJsonView" class="job-full-json-pre">${escapeHtml(JSON.stringify(j, null, 2))}</pre>
                </div>
            `;

            // Restore scroll positions after rebuilding
            const restoredPre = document.getElementById('fullJsonView');
            if (restoredPre && (preScrollTop || preScrollLeft)) {
                restoredPre.scrollTop = preScrollTop;
                restoredPre.scrollLeft = preScrollLeft;
            }
            if (host && hostScrollTop) {
                host.scrollTop = hostScrollTop;
            }

            window.__renderedJobId = j.job_id;
            window.__renderedJobJson = newJsonStr;
        } catch (e) {
            console.warn('job detail failed', e);
        }
    }

    if ($('refreshJobsBtn')) $('refreshJobsBtn').addEventListener('click', () => { loadJobs(); });

    if ($('runPipelineBtn')) {
        $('runPipelineBtn').addEventListener('click', async () => {
            const schema_id = $('schemaSelect').value;
            if (!schema_id) return;
            const selected = Array.from(document.querySelectorAll('.doc-check:checked')).map(c => c.value);
            if (!selected.length) return;

            const btn = $('runPipelineBtn');
            btn.disabled = true;
            const oldText = btn.textContent;
            btn.textContent = 'Starting...';
            try {
                const fd = new FormData();
                fd.append('schema_id', schema_id);
                fd.append('documents', JSON.stringify(selected));
                const res = await api('/pipeline/run', { method: 'POST', body: fd });
                btn.textContent = 'Running (job ' + res.job_id + ')...';
                state.selectedJobId = res.job_id;
                loadJobs();

                let polls = 0;
                const interval = setInterval(async () => {
                    polls++;
                    try {
                        const j = await api('/pipeline/jobs/' + res.job_id);
                        loadJobs();
                        if (j.status === 'completed' || j.status === 'killed' || polls > 300) {
                            clearInterval(interval);
                            btn.textContent = oldText;
                            btn.disabled = false;
                            updateRunBtn();
                        }
                    } catch (e) {
                        clearInterval(interval);
                        btn.textContent = oldText;
                        btn.disabled = false;
                    }
                }, 2000);
            } catch (e) {
                btn.textContent = oldText;
                btn.disabled = false;
                alert('Pipeline start failed: ' + e.message);
            }
        });
    }

    // ======================= User Pipeline & Jobs =======================

    async function runUserPipeline() {
        const btn = $('userAutoRunPipelineBtn') || $('quickRunPipelineBtn');
        if (btn) btn.disabled = true;
        showStatus('userAutoRunStatus', 'Starting extraction pipeline for your workspace...', 'info');

        try {
            const res = await api('/me/pipeline/run', { method: 'POST' });
            showStatus('userAutoRunStatus', `✓ Job ${res.job_id} queued for ${res.targets} document(s).`, 'success');
            loadUserJobs();
        } catch (e) {
            showStatus('userAutoRunStatus', 'Failed to run pipeline: ' + e.message, 'error');
        } finally {
            if (btn) btn.disabled = false;
        }
    }

    if ($('userAutoRunPipelineBtn')) $('userAutoRunPipelineBtn').addEventListener('click', runUserPipeline);
    if ($('quickRunPipelineBtn')) {
        $('quickRunPipelineBtn').addEventListener('click', () => {
            if (state.role === 'user') {
                runUserPipeline();
            } else {
                // Admin quick run: scroll down and pre-select
                const pipeSection = $('adminPipelineSection');
                if (pipeSection) pipeSection.scrollIntoView({ behavior: 'smooth' });
            }
        });
    }

    async function loadUserJobs() {
        if (state.role !== 'user') return;
        try {
            const data = await api('/me/pipeline/status');
            const jobs = data.jobs || [];
            state.userJobs = jobs;
            const host = $('userJobList');
            if (!host) return;

            if (!jobs.length) {
                host.innerHTML = '<p class="muted">No jobs yet. Click "Run Pipeline" above once you confirm a schema.</p>';
                return;
            }

            host.innerHTML = '';
            jobs.sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''));
            jobs.forEach(j => {
                const total = j.total || 0;
                const done = (j.succeeded || 0) + (j.failed || 0);
                const pct = total ? Math.round(100 * done / total) : 0;
                const row = el('div', 'user-job-row');

                let actionButtons = '';
                if (j.status === 'running' || j.status === 'queued') {
                    actionButtons = `
                        <button class="btn btn-ghost btn-sm" title="Pause job" data-action="pause" data-job="${escapeHtml(j.job_id)}">⏸ Pause</button>
                        <button class="btn btn-ghost btn-sm" title="Kill job" style="color:#fca5a5" data-action="kill" data-job="${escapeHtml(j.job_id)}">✕ Kill</button>
                    `;
                } else if (j.status === 'paused') {
                    actionButtons = `
                        <button class="btn btn-ghost btn-sm" title="Resume job" style="color:#6ee7b7" data-action="resume" data-job="${escapeHtml(j.job_id)}">▶ Resume</button>
                        <button class="btn btn-ghost btn-sm" title="Kill job" style="color:#fca5a5" data-action="kill" data-job="${escapeHtml(j.job_id)}">✕ Kill</button>
                    `;
                } else if (j.status === 'completed' || j.succeeded > 0) {
                    actionButtons = `
                        <button class="btn btn-outline btn-sm" title="Download JSON report" data-action="download-json" data-job="${escapeHtml(j.job_id)}">📥 JSON</button>
                        <button class="btn btn-primary btn-sm" title="Download PDF report" data-action="download-pdf" data-job="${escapeHtml(j.job_id)}">📄 PDF</button>
                    `;
                }

                const currentDoc = j.currently_processing ? ` • processing: <code>${escapeHtml(j.currently_processing)}</code>` : '';

                row.innerHTML = `
                    <div class="user-job-row-left">
                        <div class="job-id">${escapeHtml(j.job_id)} <span class="badge ${jobStatusBadgeClass(j.status)}">${escapeHtml(j.status)}</span></div>
                        <div class="job-meta">
                            Success: <strong style="color:#6ee7b7">${j.succeeded}</strong> •
                            Failed: <strong style="color:#fca5a5">${j.failed}</strong> •
                            Remaining: ${j.remaining} / ${total}${currentDoc}
                        </div>
                    </div>
                    <div class="user-job-row-right">
                        <div class="progress-bar-wrap"><div class="progress-bar-fill" style="width:${pct}%"></div></div>
                        <div class="job-item-actions">${actionButtons}</div>
                    </div>
                `;

                row.addEventListener('click', async (e) => {
                    const btn = e.target.closest('button[data-action]');
                    if (btn) {
                        e.stopPropagation();
                        const act = btn.dataset.action;
                        const jid = btn.dataset.job;
                        try {
                            if (act === 'download-json') {
                                const jobDetail = await api(`/me/pipeline/jobs/${jid}`);
                                const jsonStr = JSON.stringify(jobDetail, null, 2);
                                const blob = new Blob([jsonStr], { type: 'application/json' });
                                const url = URL.createObjectURL(blob);
                                const a = document.createElement('a');
                                a.href = url;
                                a.download = `${jid}.json`;
                                document.body.appendChild(a);
                                a.click();
                                document.body.removeChild(a);
                                URL.revokeObjectURL(url);
                                return;
                            }
                            if (act === 'download-pdf') {
                                downloadJobPdf(jid);
                                return;
                            }
                            if (act === 'pause') await api(`/me/pipeline/jobs/${jid}/pause`, { method: 'POST' });
                            else if (act === 'resume') await api(`/me/pipeline/jobs/${jid}/resume`, { method: 'POST' });
                            else if (act === 'kill') {
                                if (!confirm(`Cancel job ${jid}?`)) return;
                                await api(`/me/pipeline/jobs/${jid}/kill`, { method: 'POST' });
                            }
                            loadUserJobs();
                        } catch (err) {
                            alert(`Action ${act} failed: ${err.message}`);
                        }
                        return;
                    }
                    loadJobDetail(j.job_id, true);
                });

                host.appendChild(row);
            });
            if (state.selectedJobId && state.role === 'user') loadJobDetail(state.selectedJobId, false);
        } catch (e) {
            console.warn('loadUserJobs failed', e);
        }
    }

    if ($('refreshUserJobsBtn')) $('refreshUserJobsBtn').addEventListener('click', loadUserJobs);

    // ======================= Admin Logs Viewer =======================

    async function loadUserActivityLogs() {
        if (state.role !== 'admin') return;
        const host = $('userLogsList');
        if (!host) return;
        const filterUser = $('logFilterUsername') ? $('logFilterUsername').value.trim() : '';

        try {
            const path = '/admin/logs/users' + (filterUser ? `?username=${encodeURIComponent(filterUser)}` : '');
            const data = await api(path);
            const logs = data.logs || [];

            if (!logs.length) {
                host.innerHTML = '<p class="muted">No user activity recorded yet.</p>';
                return;
            }

            let html = `
                <table class="log-table">
                    <thead>
                        <tr>
                            <th style="width: 22%;">Timestamp</th>
                            <th style="width: 18%;">User</th>
                            <th style="width: 22%;">Action</th>
                            <th style="width: 38%;">Detail</th>
                        </tr>
                    </thead>
                    <tbody>
            `;

            logs.slice().reverse().forEach(l => {
                html += `
                    <tr>
                        <td class="muted small">${escapeHtml(l.timestamp)}</td>
                        <td><strong>${escapeHtml(l.username)}</strong></td>
                        <td><span class="badge badge-info">${escapeHtml(l.action)}</span></td>
                        <td><code style="font-size:11px;">${escapeHtml(JSON.stringify(l.detail))}</code></td>
                    </tr>
                `;
            });

            html += '</tbody></table>';
            host.innerHTML = html;
        } catch (e) {
            host.innerHTML = `<p class="muted">Error loading user logs: ${e.message}</p>`;
        }
    }

    async function loadSystemLogs() {
        if (state.role !== 'admin') return;
        const host = $('systemLogsList');
        if (!host) return;

        try {
            const data = await api('/admin/logs/system?limit=300');
            const lines = data.lines || [];
            if (!lines.length) {
                host.textContent = 'No system logs recorded yet.';
            } else {
                host.textContent = lines.join('\n');
                host.scrollTop = host.scrollHeight;
            }
        } catch (e) {
            host.textContent = 'Error loading system logs: ' + e.message;
        }
    }

    if ($('logSubTabUsersBtn') && $('logSubTabSystemBtn')) {
        $('logSubTabUsersBtn').addEventListener('click', () => {
            $('logSubTabUsersBtn').className = 'btn btn-primary btn-sm';
            $('logSubTabSystemBtn').className = 'btn btn-outline btn-sm';
            $('userLogsPanel').classList.remove('hidden');
            $('systemLogsPanel').classList.add('hidden');
            loadUserActivityLogs();
        });

        $('logSubTabSystemBtn').addEventListener('click', () => {
            $('logSubTabSystemBtn').className = 'btn btn-primary btn-sm';
            $('logSubTabUsersBtn').className = 'btn btn-outline btn-sm';
            $('systemLogsPanel').classList.remove('hidden');
            $('userLogsPanel').classList.add('hidden');
            loadSystemLogs();
        });
    }

    if ($('applyLogFilterBtn')) $('applyLogFilterBtn').addEventListener('click', loadUserActivityLogs);
    if ($('clearLogFilterBtn')) {
        $('clearLogFilterBtn').addEventListener('click', () => {
            if ($('logFilterUsername')) $('logFilterUsername').value = '';
            loadUserActivityLogs();
        });
    }
    if ($('refreshLogsBtn')) {
        $('refreshLogsBtn').addEventListener('click', () => {
            if ($('systemLogsPanel') && !$('systemLogsPanel').classList.contains('hidden')) {
                loadSystemLogs();
            } else {
                loadUserActivityLogs();
            }
        });
    }

    // ======================= Admin User Management =======================

    async function loadUsersList() {
        if (state.role !== 'admin') return;
        const host = $('usersListTable');
        if (!host) return;

        try {
            const users = await api('/admin/users');
            if (!users.length) {
                host.innerHTML = '<p class="muted">No users found.</p>';
                return;
            }

            let html = `
                <table class="log-table">
                    <thead>
                        <tr>
                            <th>User ID</th>
                            <th>Username</th>
                            <th>Role</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            users.forEach(u => {
                html += `
                    <tr>
                        <td class="muted small">${escapeHtml(u.user_id)}</td>
                        <td><strong>${escapeHtml(u.username)}</strong></td>
                        <td><span class="badge ${u.role === 'admin' ? 'badge-primary' : 'badge-info'}">${escapeHtml(u.role)}</span></td>
                    </tr>
                `;
            });
            html += '</tbody></table>';
            host.innerHTML = html;
        } catch (e) {
            host.innerHTML = `<p class="muted">Error loading users: ${e.message}</p>`;
        }
    }

    if ($('refreshUsersListBtn')) $('refreshUsersListBtn').addEventListener('click', loadUsersList);

    if ($('createUserForm')) {
        $('createUserForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = $('newUsername').value.trim();
            const password = $('newPassword').value;
            const role = $('newRole').value;
            const btn = $('createUserBtn');

            if (!username || !password) return;

            btn.disabled = true;
            showStatus('createUserStatus', 'Creating account...', 'info');

            try {
                const res = await api('/admin/users', {
                    method: 'POST',
                    json: { username, password, role },
                });
                showStatus('createUserStatus', `✓ User '${res.username}' created successfully with role '${res.role}'!`, 'success');
                $('newUsername').value = '';
                $('newPassword').value = '';
                loadUsersList();
            } catch (err) {
                showStatus('createUserStatus', 'Failed to create user: ' + err.message, 'error');
            } finally {
                btn.disabled = false;
            }
        });
    }

    // ======================= Query Bot Tab =======================

    // Holds all extracted data merged from every completed job
    let qbAllExtractedData = null;
    // Conversation history for multi-turn memory
    let qbChatHistory = [];

    async function loadAllExtractedData() {
        try {
            const data = await api('/pipeline/status');
            const jobs = Object.values(data.jobs || {});
            const completed = jobs.filter(j => j.status === 'completed');

            if (!completed.length) {
                qbAllExtractedData = null;
                return;
            }

            const details = await Promise.all(
                completed.map(j => api('/pipeline/jobs/' + j.job_id).catch(() => null))
            );

            const merged = {};
            details.forEach(j => {
                if (!j) return;
                const sucs = j.successes || [];
                sucs.forEach(s => {
                    if (s.extracted_data) {
                        const key = s.extracted_json || s.pdf || s.doc_id || 'document';
                        merged[key] = s.extracted_data;
                    }
                });
                if (!sucs.length && j.extracted_data) {
                    merged[j.job_id] = j.extracted_data;
                }
            });

            qbAllExtractedData = Object.keys(merged).length ? merged : null;
        } catch (e) {
            qbAllExtractedData = null;
            console.warn('loadAllExtractedData error', e);
        }
    }

    async function askTabQueryBot() {
        const input = $('tab_qb_input');
        const btn = $('tab_qb_btn');
        const msgs = $('tab_qb_msgs');
        if (!input || !btn || !msgs) return;

        const question = input.value.trim();
        if (!question) { input.focus(); return; }

        if (!qbAllExtractedData) {
            const errDiv = document.createElement('div');
            errDiv.className = 'query-bot-msg bot error';
            errDiv.textContent = 'No extracted data available yet. Run a pipeline job first.';
            msgs.appendChild(errDiv);
            msgs.scrollTop = msgs.scrollHeight;
            return;
        }

        const userDiv = document.createElement('div');
        userDiv.className = 'query-bot-msg user';
        userDiv.textContent = question;
        msgs.appendChild(userDiv);

        const botDiv = document.createElement('div');
        botDiv.className = 'query-bot-msg bot loading';
        botDiv.textContent = 'Thinking…';
        msgs.appendChild(botDiv);
        msgs.scrollTop = msgs.scrollHeight;

        input.value = '';
        input.disabled = true;
        btn.disabled = true;
        btn.textContent = 'Asking…';

        try {
            const res = await api('/api/query-bot/ask', {
                method: 'POST',
                json: {
                    extracted_data: qbAllExtractedData,
                    question,
                    history: qbChatHistory,
                    doc_id: 'all extracted documents'
                }
            });
            const answer = res.answer || 'No answer returned.';
            botDiv.className = 'query-bot-msg bot';
            botDiv.textContent = answer;
            // Append to history only on success
            qbChatHistory.push({ role: 'user', content: question });
            qbChatHistory.push({ role: 'assistant', content: answer });
        } catch (err) {
            botDiv.className = 'query-bot-msg bot error';
            botDiv.textContent = 'Error: ' + (err.message || String(err));
        } finally {
            input.disabled = false;
            btn.disabled = false;
            btn.textContent = 'Ask Bot';
            input.focus();
            msgs.scrollTop = msgs.scrollHeight;
        }
    }

    if ($('tab_qb_btn')) $('tab_qb_btn').addEventListener('click', askTabQueryBot);
    if ($('tab_qb_input')) {
        $('tab_qb_input').addEventListener('keydown', (e) => {
            if (e.key === 'Enter') { e.preventDefault(); askTabQueryBot(); }
        });
    }
    if ($('qbClearBtn')) {
        $('qbClearBtn').addEventListener('click', () => {
            const msgs = $('tab_qb_msgs');
            if (msgs) {
                msgs.innerHTML = '<div class="query-bot-msg bot">Hi! Ask me anything about the extracted document data.</div>';
            }
            qbChatHistory = [];
        });
    }

    // ======================= Downloads (Jobs & Schemas) =======================
    window.downloadJobJson = function() {
        if (!window.__currentJobDetail) return;
        const jsonStr = JSON.stringify(window.__currentJobDetail, null, 2);
        const blob = new Blob([jsonStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${window.__currentJobDetail.job_id || 'pipeline_job'}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    window.downloadJobPdf = async function(jobId) {
        if (!jobId && window.__currentJobDetail) jobId = window.__currentJobDetail.job_id;
        if (!jobId) return;
        try {
            const apiPath = state.role === 'user' ? (`${API_BASE}/me/pipeline/jobs/${jobId}/pdf`) : (`${API_BASE}/pipeline/jobs/${jobId}/pdf`);
            const resp = await fetch(apiPath, {
                headers: state.token ? { 'Authorization': 'Bearer ' + state.token } : {}
            });
            if (!resp.ok) {
                const errText = await resp.text();
                throw new Error(errText || 'Failed to download PDF');
            }
            const blob = await resp.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${jobId}_report.pdf`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (e) {
            alert('PDF download failed: ' + e.message);
        }
    };

    window.downloadSchemaPdf = async function(schemaId) {
        schemaId = schemaId || state.lastConfirmedSchemaId;
        if (!schemaId) {
            alert('No confirmed schema available to download.');
            return;
        }
        try {
            const resp = await fetch(`${API_BASE}/schema/${schemaId}/pdf`, {
                headers: state.token ? { 'Authorization': 'Bearer ' + state.token } : {}
            });
            if (!resp.ok) {
                const errText = await resp.text();
                throw new Error(errText || 'Failed to download schema PDF');
            }
            const blob = await resp.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${schemaId}.pdf`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (e) {
            alert('Schema PDF download failed: ' + e.message);
        }
    };

    window.downloadSchemaJson = async function(schemaId) {
        schemaId = schemaId || state.lastConfirmedSchemaId;
        if (!schemaId) {
            alert('No confirmed schema available to download.');
            return;
        }
        try {
            const resp = await fetch(`${API_BASE}/schema/${schemaId}/json`, {
                headers: state.token ? { 'Authorization': 'Bearer ' + state.token } : {}
            });
            if (!resp.ok) {
                const errText = await resp.text();
                throw new Error(errText || 'Failed to download schema JSON');
            }
            const blob = await resp.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${schemaId}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (e) {
            alert('Schema JSON download failed: ' + e.message);
        }
    };

    // ======================= Init & Bootstrap =======================

    if ('scrollRestoration' in history) {
        history.scrollRestoration = 'manual';
    }
    window.scrollTo(0, 0);

    initTheme();
    checkHealth();

    if ($('loginForm')) $('loginForm').addEventListener('submit', handleLoginSubmit);
    if ($('logoutBtn')) $('logoutBtn').addEventListener('click', logout);

    if (state.token) {
        initAuthenticatedSession();
    } else {
        showLoginModal();
    }

    setInterval(checkHealth, 15000);
    setInterval(() => {
        if (state.token) {
            if (state.role === 'admin') loadJobs();
            else if (state.role === 'user') loadUserJobs();
        }
    }, 4000);

})();


