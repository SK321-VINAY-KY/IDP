(function () {
    'use strict';

    const API_BASE = window.location.origin;

    let state = {
        currentUser: { username: 'user1', role: 'normal', full_name: 'Normal User 1' },
        token: 'token_user1_secret',
        sessionId: null,
        docs: [],
        schemas: [],
        currentSchema: null,
        currentErrors: [],
        currentState: 'idle',
        completed: false,
        jobs: [],
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

    async function api(path, opts = {}) {
        const url = path.startsWith('http') ? path : API_BASE + path;
        const headers = opts.headers ? { ...opts.headers } : {};
        if (opts.json) {
            headers['Content-Type'] = 'application/json';
        }
        if (state.token) {
            headers['Authorization'] = 'Bearer ' + state.token;
        }
        if (state.currentUser && state.currentUser.username) {
            headers['X-User-Id'] = state.currentUser.username;
        }

        const resp = await fetch(url, {
            ...opts,
            headers,
            body: opts.json ? JSON.stringify(opts.json) : opts.body,
        });
        const ct = resp.headers.get('content-type') || '';
        const body = ct.includes('application/json') ? await resp.json() : await resp.text();
        if (!resp.ok) {
            const msg = typeof body === 'object' ? (body.detail || resp.statusText) : String(body || resp.statusText);
            throw new Error(msg);
        }
        return body;
    }

    // ======================= Auth & User Management =======================
    const AUTH_USER_KEY = 'idp_auth_user';
    const AUTH_TOKEN_KEY = 'idp_auth_token';

    async function initAuth() {
        const savedUser = localStorage.getItem(AUTH_USER_KEY) || 'user1';
        const userSelect = $('userSwitchSelect');
        if (userSelect) {
            userSelect.value = savedUser;
            userSelect.addEventListener('change', async (e) => {
                await switchUser(e.target.value);
            });
        }
        await switchUser(savedUser);
    }

    async function switchUser(username) {
        try {
            state.selectedJobId = null;
            const detailCard = $('jobDetailCard');
            if (detailCard) detailCard.classList.add('hidden');
            const detailHost = $('jobDetail');
            if (detailHost) detailHost.innerHTML = '';

            const res = await api('/auth/login', { json: { username: username } });
            state.currentUser = res.user;
            state.token = res.token;
            localStorage.setItem(AUTH_USER_KEY, res.user.username);
            localStorage.setItem(AUTH_TOKEN_KEY, res.token);
            renderUserProfile();
            await loadPipelineStatus();
            await loadJobs();
            if (res.user.role === 'admin') {
                loadAdminOverview();
                loadAdminUsers();
                loadAdminAudit();
                loadAdminLogs();
            }
        } catch (e) {
            console.error('Failed to switch user:', e);
        }
    }

    function renderUserProfile() {
        const u = state.currentUser;
        if (!u) return;
        const avatar = $('userAvatar');
        const role = $('userRoleBadge');
        const adminTabBtn = $('adminTabBtn');
        const jobListHeading = $('jobListHeading');
        const userSelect = $('userSwitchSelect');

        if (userSelect && userSelect.value !== u.username) {
            userSelect.value = u.username;
        }
        if (avatar) avatar.textContent = u.role === 'admin' ? '👑' : '👤';
        if (role) {
            role.textContent = u.role === 'admin' ? 'Administrator' : 'Normal User';
            role.className = 'badge user-role-pill ' + (u.role === 'admin' ? 'badge-primary' : 'badge-mute');
        }

        if (adminTabBtn) {
            if (u.role === 'admin') {
                adminTabBtn.classList.remove('hidden');
                adminTabBtn.style.display = 'inline-flex';
            } else {
                adminTabBtn.classList.add('hidden');
                adminTabBtn.style.display = 'none';
                const activeTab = document.querySelector('.tab-btn.active');
                if (activeTab && activeTab.dataset.tab === 'admin') {
                    document.querySelector('.tab-btn[data-tab="chatbot"]').click();
                }
            }
        }

        if (jobListHeading) {
            jobListHeading.textContent = u.role === 'admin' ? 'All Pipeline Jobs (Admin View)' : 'My Pipeline Jobs';
        }
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
                loadSchemas();
                loadPipelineStatus();
                loadJobs();
                if (state.docs.length === 0) loadDocuments();
                else renderDocChecklist();
            } else if (btn.dataset.tab === 'documents') {
                loadDocuments();
            } else if (btn.dataset.tab === 'admin') {
                loadAdminOverview();
                loadAdminUsers();
                loadAdminAudit();
                loadAdminLogs();
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

    // ======================= Documents Tab =======================

    async function loadDocuments() {
        try {
            const data = await api('/documents');
            state.docs = data.documents || [];
            const outs = data.outputs || [];
            $('docCount').textContent = state.docs.length;
            $('outCount').textContent = outs.length;
            renderDocList(state.docs, outs);
            renderDocChecklist();
        } catch (e) {
            $('docList').innerHTML = `<p class="muted">Failed to load: ${e.message}</p>`;
        }
    }

    function renderDocList(docs, outs) {
        if (!docs.length) {
            $('docList').innerHTML = '<p class="muted">No documents yet. Upload PDFs above.</p>';
        } else {
            $('docList').innerHTML = '';
            docs.forEach(d => {
                const row = el('div', 'doc-item');
                const info = el('div', 'doc-item-info');
                info.innerHTML = `
                    <span class="doc-icon"></span>
                    <div>
                        <div class="doc-name">${d.name}</div>
                        <div class="doc-meta">${fmtSize(d.size)}  ${new Date(d.modified).toLocaleString()}</div>
                    </div>
                `;
                const badge = d.has_output
                    ? '<span class="badge doc-badge badge-success">processed</span>'
                    : '<span class="badge doc-badge badge-mute">pending</span>';
                const actions = el('div');
                actions.innerHTML = badge;
                row.appendChild(info);
                row.appendChild(actions);
                $('docList').appendChild(row);
            });
        }

        if (!outs.length) {
            $('outList').innerHTML = '<p class="muted">No processed outputs yet. Run the pipeline.</p>';
        } else {
            $('outList').innerHTML = '';
            outs.forEach(o => {
                const row = el('div', 'doc-item');
                const info = el('div', 'doc-item-info');
                const schema = o.schema_ref ? `  schema: <code>${o.schema_ref.schema_id || '--'}</code>` : '';
                info.innerHTML = `
                    <span class="doc-icon"></span>
                    <div>
                        <div class="doc-name">${o.name}</div>
                        <div class="doc-meta">${fmtSize(o.size)}  ${new Date(o.modified).toLocaleString()}${schema}</div>
                    </div>
                `;
                row.appendChild(info);
                $('outList').appendChild(row);
            });
        }
    }

    // Upload zone
    (function setupUpload() {
        const zone = $('uploadZone');
        const input = $('fileInput');
        const browse = $('browseBtn');

        function handleFiles(files) {
            const pdfs = Array.from(files).filter(f => f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf'));
            if (!pdfs.length) return;
            uploadFiles(pdfs);
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

        async function uploadFiles(files) {
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
    })();

    $('refreshDocsBtn').addEventListener('click', loadDocuments);

    function showStatus(id, html, kind) {
        const s = $(id);
        s.className = 'status-box' + (kind ? ' ' + kind : '');
        s.innerHTML = html;
        s.classList.remove('hidden');
    }
    function hideStatus(id) { $(id).classList.add('hidden'); }

    // ======================= Chatbot Tab =======================

    function addChatMsg(who, text) {
        const log = $('chatLog');
        const wrap = el('div', 'chat-msg ' + who);
        const b = el('div', 'msg-bubble');
        b.textContent = text;
        wrap.appendChild(b);
        log.appendChild(wrap);
        log.scrollTop = log.scrollHeight;
    }

    function setStateBadge(s, s2) {
        const b = $('stateBadge');
        const map = {
            START: 'badge-info', REVIEW: 'badge-warn', COMPLETED: 'badge-success',
        };
        b.className = 'badge ' + (map[s] || 'badge-mute');
        b.textContent = s.toLowerCase() + (s2 ? '  ' + s2 : '');
    }

    let schemaSyncTimer = null;

    function escapeHtml(str) {
        if (str === null || str === undefined) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    function syncSchemaFromTable() {
        if (!state.sessionId) return;
        const dtInput = $('schemaDocTypeInput');
        const docType = dtInput ? dtInput.value.trim() : (state.currentSchema?.document_type || 'document');

        const rows = document.querySelectorAll('#schemaTableBody tr');
        const fields = [];
        rows.forEach(tr => {
            const nameInput = tr.querySelector('.field-name-input');
            const typeSelect = tr.querySelector('.field-type-select');
            const reqBtn = tr.querySelector('.field-req-btn');
            const descInput = tr.querySelector('.field-desc-input');

            if (nameInput && nameInput.value.trim()) {
                fields.push({
                    name: nameInput.value.trim(),
                    type: typeSelect ? typeSelect.value : 'string',
                    required: reqBtn ? reqBtn.getAttribute('data-required') === 'true' : true,
                    description: descInput ? descInput.value.trim() : '',
                });
            }
        });

        if (!state.currentSchema) state.currentSchema = {};
        state.currentSchema.document_type = docType;
        state.currentSchema.fields = fields;

        clearTimeout(schemaSyncTimer);
        schemaSyncTimer = setTimeout(async () => {
            try {
                const res = await api(`/session/${state.sessionId}/schema`, {
                    method: 'POST',
                    json: { document_type: docType, fields: fields }
                });
                state.currentErrors = res.errors || [];
                renderErrors(state.currentErrors);
            } catch (err) {
                console.warn('Manual schema sync failed:', err);
            }
        }, 500);
    }

    function renderSchemaPanel(schema) {
        state.currentSchema = schema;
        const panel = $('schemaPanel');
        const hasSchema = schema && (schema.document_type || (schema.fields && schema.fields.length));

        $('addSchemaFieldBtn').disabled = !state.sessionId || state.completed;
        $('copySchemaBtn').disabled = !hasSchema;
        $('printSchemaBtn').disabled = !state.sessionId;
        $('confirmBtn').disabled = !hasSchema || state.completed;

        if (!hasSchema) {
            panel.innerHTML = '<p class="muted">No schema yet. Chat with the bot or upload samples to generate one.</p>';
            return;
        }

        const docType = schema.document_type || 'document';
        const fields = schema.fields || [];
        const types = ['string', 'number', 'integer', 'boolean', 'date', 'array[string]', 'array[object]', 'object'];

        let html = `
            <div class="schema-editor-toolbar">
                <label class="schema-doctype-label">
                    <span>Document Type:</span>
                    <input type="text" id="schemaDocTypeInput" class="schema-doctype-input" value="${escapeHtml(docType)}" placeholder="e.g. invoice, resume, receipt" ${state.completed ? 'disabled' : ''}>
                </label>
            </div>
            <div class="schema-table-wrap">
                <table class="schema-table">
                    <thead>
                        <tr>
                            <th style="width: 25%;">Field Name</th>
                            <th style="width: 20%;">Type</th>
                            <th style="width: 14%; text-align: center;">Required</th>
                            <th style="width: 33%;">Description</th>
                            <th style="width: 8%; text-align: center;"></th>
                        </tr>
                    </thead>
                    <tbody id="schemaTableBody">
        `;

        if (fields.length === 0) {
            html += `<tr><td colspan="5" class="muted" style="text-align:center; padding: 14px;">No fields defined. Click <strong>+ Add Field</strong> above.</td></tr>`;
        } else {
            fields.forEach((f, idx) => {
                const isReq = f.required !== false;
                const fieldType = f.type || 'string';
                html += `
                    <tr data-index="${idx}">
                        <td>
                            <input type="text" class="field-input field-name-input" value="${escapeHtml(f.name)}" placeholder="field_name" ${state.completed ? 'disabled' : ''}>
                        </td>
                        <td>
                            <select class="field-select field-type-select" ${state.completed ? 'disabled' : ''}>
                                ${types.map(t => `<option value="${t}" ${t === fieldType ? 'selected' : ''}>${t}</option>`).join('')}
                            </select>
                        </td>
                        <td style="text-align: center;">
                            <button type="button" class="btn-req-toggle field-req-btn ${isReq ? 'is-req' : 'is-opt'}" data-required="${isReq}" ${state.completed ? 'disabled' : ''}>
                                ${isReq ? 'YES' : 'NO'}
                            </button>
                        </td>
                        <td>
                            <input type="text" class="field-input field-desc-input" value="${escapeHtml(f.description || '')}" placeholder="Optional description..." ${state.completed ? 'disabled' : ''}>
                        </td>
                        <td style="text-align: center;">
                            <button type="button" class="btn-row-del" title="Delete field" ${state.completed ? 'disabled' : ''}>✕</button>
                        </td>
                    </tr>
                `;
            });
        }

        html += `
                    </tbody>
                </table>
            </div>
        `;

        panel.innerHTML = html;

        const dtInput = $('schemaDocTypeInput');
        if (dtInput) {
            dtInput.addEventListener('input', syncSchemaFromTable);
        }

        panel.querySelectorAll('.field-name-input, .field-desc-input').forEach(inp => {
            inp.addEventListener('input', syncSchemaFromTable);
        });

        panel.querySelectorAll('.field-type-select').forEach(sel => {
            sel.addEventListener('change', syncSchemaFromTable);
        });

        panel.querySelectorAll('.field-req-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                if (state.completed) return;
                const cur = btn.getAttribute('data-required') === 'true';
                const next = !cur;
                btn.setAttribute('data-required', next ? 'true' : 'false');
                btn.textContent = next ? 'YES' : 'NO';
                btn.className = `btn-req-toggle field-req-btn ${next ? 'is-req' : 'is-opt'}`;
                syncSchemaFromTable();
            });
        });

        panel.querySelectorAll('.btn-row-del').forEach(btn => {
            btn.addEventListener('click', (e) => {
                if (state.completed) return;
                const tr = e.target.closest('tr');
                if (tr) {
                    tr.remove();
                    syncSchemaFromTable();
                }
            });
        });
    }

    $('addSchemaFieldBtn').addEventListener('click', () => {
        if (!state.sessionId || state.completed) return;
        if (!state.currentSchema) state.currentSchema = { document_type: 'document', fields: [] };
        if (!state.currentSchema.fields) state.currentSchema.fields = [];
        state.currentSchema.fields.push({
            name: 'new_field_' + (state.currentSchema.fields.length + 1),
            type: 'string',
            required: true,
            description: ''
        });
        renderSchemaPanel(state.currentSchema);
        syncSchemaFromTable();
        const inputs = document.querySelectorAll('.field-name-input');
        if (inputs.length) {
            const last = inputs[inputs.length - 1];
            last.focus();
            last.select();
        }
    });

    function renderErrors(errors) {
        const host = $('schemaErrors');
        if (!errors || !errors.length) {
            host.classList.add('hidden');
            host.innerHTML = '';
            return;
        }
        host.innerHTML = '<strong>Validation:</strong><ul>' + errors.map(e => `<li>${e}</li>`).join('') + '</ul>';
        host.classList.remove('hidden');
    }

    async function newSession() {
        $('chatLog').innerHTML = '';
        hideStatus('confirmStatus');
        hideStatus('inferStatus');
        $('postConfirmBox').classList.add('hidden');
        $('quickRunPipelineBtn').disabled = true;
        state.completed = false;
        state.currentSchema = null;
        state.currentErrors = [];
        renderSchemaPanel(null);
        renderErrors([]);
        try {
            const res = await api('/chat', { method: 'POST', json: { session_id: null } });
            state.sessionId = res.session_id;
            state.currentState = res.state;
            setStateBadge(res.state);
            addChatMsg('bot', res.message);
            $('chatInput').disabled = false;
            $('sendBtn').disabled = false;
            $('chatInput').focus();
        } catch (e) {
            addChatMsg('bot', 'Failed to start session: ' + e.message);
        }
    }

    $('newSessionBtn').addEventListener('click', newSession);

    async function sendMsg() {
        const input = $('chatInput');
        const text = input.value.trim();
        if (!text || !state.sessionId) return;
        input.value = '';
        addChatMsg('user', text);
        input.disabled = true;
        $('sendBtn').disabled = true;
        try {
            const res = await api('/chat', { method: 'POST', json: { session_id: state.sessionId, message: text } });
            state.currentState = res.state;
            state.completed = res.completed;
            setStateBadge(res.state);
            addChatMsg('bot', res.message);
            if (res.schema) renderSchemaPanel(res.schema);
            renderErrors(res.errors || []);
            if (res.completed) {
                input.disabled = true;
                $('sendBtn').disabled = true;
                $('confirmBtn').disabled = true;
                state.lastConfirmedSchemaId = res.schema_id;
                showStatus('confirmStatus', `Schema confirmed! Saved as <code>${res.schema_id}.json</code> in <code>schema_registry/</code>.`, 'success');
                $('postConfirmBox').classList.remove('hidden');
                $('quickRunPipelineBtn').disabled = false;
                loadSchemas();
            } else {
                input.disabled = false;
                $('sendBtn').disabled = false;
                input.focus();
            }
        } catch (e) {
            addChatMsg('bot', 'Error: ' + e.message);
            input.disabled = false;
            $('sendBtn').disabled = false;
        }
    }

    $('sendBtn').addEventListener('click', sendMsg);
    $('chatInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { e.preventDefault(); sendMsg(); }
    });

    $('confirmBtn').addEventListener('click', () => {
        $('chatInput').value = '/confirm';
        sendMsg();
    });

    $('copySchemaBtn').addEventListener('click', () => {
        if (!state.currentSchema) return;
        navigator.clipboard.writeText(JSON.stringify(state.currentSchema, null, 2))
            .then(() => alert('Schema copied to clipboard'))
            .catch(() => alert('Copy failed'));
    });

    $('printSchemaBtn').addEventListener('click', () => {
        $('chatInput').value = '/schema';
        sendMsg();
    });

    // Inference upload
    (function setupInferUpload() {
        const input = $('inferInput');
        const btn = $('inferBrowseBtn');
        const label = $('inferSelected');
        const runBtn = $('inferBtn');
        let selectedFiles = [];

        btn.addEventListener('click', () => input.click());
        input.addEventListener('change', () => {
            selectedFiles = Array.from(input.files).filter(f => f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf'));
            if (!selectedFiles.length) {
                label.textContent = 'no files';
                runBtn.disabled = true;
                return;
            }
            label.textContent = `${selectedFiles.length} file(s): ` + selectedFiles.map(f => f.name).join(', ');
            runBtn.disabled = !(selectedFiles.length >= 2 && selectedFiles.length <= 5);
            if (selectedFiles.length < 2 || selectedFiles.length > 5) {
                label.textContent += ' (pick 2-5 files)';
            }
        });

        runBtn.addEventListener('click', async () => {
            if (!selectedFiles.length) return;
            runBtn.disabled = true;
            btn.disabled = true;
            showStatus('inferStatus', `Inferring schema from ${selectedFiles.length} sample document(s)... <br><small class="muted">Contacting Sarvam Doc AI OCR and LLM. This typically takes 60-180 seconds. Please wait...</small>`);
            const fd = new FormData();
            selectedFiles.forEach(f => fd.append('files', f, f.name));
            if (state.sessionId) fd.append('session_id', state.sessionId);
            try {
                const res = await api('/schema/infer', { method: 'POST', body: fd });
                state.sessionId = res.session_id;
                state.currentState = res.state;
                state.completed = res.completed;
                setStateBadge(res.state);
                addChatMsg('bot', res.message);
                if (res.schema) renderSchemaPanel(res.schema);
                renderErrors(res.errors || []);
                showStatus('inferStatus', 'Inference complete! Proposed schema loaded.', 'success');
                $('chatInput').disabled = false;
                $('sendBtn').disabled = false;
                setTimeout(() => hideStatus('inferStatus'), 4000);
            } catch (e) {
                showStatus('inferStatus', 'Inference failed: ' + e.message, 'error');
            } finally {
                runBtn.disabled = false;
                btn.disabled = false;
            }
        });
    })();

    // ======================= Pipeline Continuation =======================

    async function quickRunPipeline() {
        const schemaId = state.lastConfirmedSchemaId;
        if (!schemaId) {
            alert('No confirmed schema yet. Confirm a schema in the chat first.');
            return;
        }
        if (!state.docs.length) {
            await loadDocuments();
        }
        await loadSchemas();
        await loadPipelineStatus();
        document.querySelectorAll('.doc-check').forEach(c => c.checked = true);
        const sel = $('schemaSelect');
        if (!sel) return;
        let opt = Array.from(sel.options).find(o => o.value === schemaId);
        if (!opt) {
            const fallbackOpt = el('option', '', `${schemaId} (just confirmed)`);
            fallbackOpt.value = schemaId;
            sel.appendChild(fallbackOpt);
        }
        sel.value = schemaId;
        updateRunBtn();
        const pipeSection = $('pipelineSection');
        if (pipeSection) {
            pipeSection.scrollIntoView({ behavior: 'smooth' });
        }
        setTimeout(() => {
            if (!$('runPipelineBtn').disabled) {
                $('runPipelineBtn').click();
            } else {
                alert('Still waiting on document/schema data. Press the button again in a moment.');
            }
        }, 300);
    }

    $('quickRunPipelineBtn').addEventListener('click', quickRunPipeline);

    async function loadSchemas() {
        try {
            const data = await api('/schemas');
            state.schemas = data.schemas || [];
            const sel = $('schemaSelect');
            sel.innerHTML = '';
            if (!state.schemas.length) {
                sel.innerHTML = '<option value="">-- No schemas in registry yet. Confirm one in chat. --</option>';
                sel.disabled = true;
            } else {
                sel.innerHTML = '<option value="">-- Select a registered schema --</option>';
                state.schemas.forEach(s => {
                    const opt = el('option', '', `${s.schema_id} (${s.document_type || 'unnamed'} - ${s.field_count || 0} fields)`);
                    opt.value = s.schema_id;
                    sel.appendChild(opt);
                });
                sel.disabled = false;
            }
            updateRunBtn();
        } catch (e) {
            console.warn('loadSchemas failed', e);
        }
    }

    $('refreshSchemasBtn').addEventListener('click', loadSchemas);
    $('schemaSelect').addEventListener('change', updateRunBtn);

    function renderDocChecklist() {
        const host = $('docChecklist');
        if (!state.docs.length) {
            host.innerHTML = '<p class="muted small">No documents in dataset/. Upload some in the Documents tab.</p>';
            $('selAllBtn').disabled = true;
            $('selNoneBtn').disabled = true;
            updateRunBtn();
            return;
        }
        host.innerHTML = '';
        state.docs.forEach((d, idx) => {
            const row = el('label', 'check-item');
            row.innerHTML = `
                <input type="checkbox" class="doc-check" value="${d.name}" checked>
                <span><strong>${d.name}</strong> <span class="muted small">(${fmtSize(d.size)})</span></span>
            `;
            host.appendChild(row);
        });
        $('selAllBtn').disabled = false;
        $('selNoneBtn').disabled = false;
        host.querySelectorAll('.doc-check').forEach(c => c.addEventListener('change', updateRunBtn));
        updateRunBtn();
    }

    $('selAllBtn').addEventListener('click', () => {
        document.querySelectorAll('.doc-check').forEach(c => c.checked = true);
        updateRunBtn();
    });
    $('selNoneBtn').addEventListener('click', () => {
        document.querySelectorAll('.doc-check').forEach(c => c.checked = false);
        updateRunBtn();
    });

    function updateRunBtn() {
        const sel = $('schemaSelect');
        const schemaOk = sel && sel.value && sel.value !== '';
        const checked = document.querySelectorAll('.doc-check:checked').length;
        const btn = $('runPipelineBtn');
        btn.disabled = !(schemaOk && checked > 0 && state.pipelineAvailable);
        if (!state.pipelineAvailable) {
            btn.title = 'Extraction pipeline modules are not importable in this environment';
        } else if (!schemaOk) {
            btn.title = 'Select a schema first';
        } else if (!checked) {
            btn.title = 'Select at least one document';
        } else {
            btn.title = `Run extraction on ${checked} document(s)`;
        }
    }

    async function loadPipelineStatus() {
        try {
            const data = await api('/pipeline/status');
            state.pipelineAvailable = data.available;
            const badge = $('pipelineAvail');
            if (data.available) {
                badge.className = 'badge badge-success';
                badge.textContent = `pipeline online (mode: ${data.routing_mode || 'single_engine'})`;
            } else {
                badge.className = 'badge badge-danger';
                badge.textContent = 'pipeline offline';
            }

            // Update mini KPI summary bar
            const mySummary = data.my_summary || {};
            if ($('kpiCompleted')) $('kpiCompleted').textContent = mySummary.completed || 0;
            if ($('kpiRemaining')) $('kpiRemaining').textContent = mySummary.remaining || 0;
            if ($('kpiRunning')) $('kpiRunning').textContent = mySummary.running || 0;
            if ($('kpiErrors')) $('kpiErrors').textContent = mySummary.errors || 0;
            if ($('userJobsBadge')) $('userJobsBadge').textContent = `${mySummary.total || 0} Total`;

            updateRunBtn();
        } catch (e) {
            console.warn('pipeline status failed', e);
        }
    }

    $('refreshJobsBtn').addEventListener('click', () => { loadJobs(); });

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
        try {
            const data = await api('/pipeline/status');
            const jobs = Object.values(data.jobs || {});
            state.jobs = jobs;

            // Always synchronize mini KPI strip with current user's summary
            const mySummary = data.my_summary || {};
            if ($('kpiCompleted')) $('kpiCompleted').textContent = mySummary.completed || 0;
            if ($('kpiRemaining')) $('kpiRemaining').textContent = mySummary.remaining || 0;
            if ($('kpiRunning')) $('kpiRunning').textContent = mySummary.running || 0;
            if ($('kpiErrors')) $('kpiErrors').textContent = mySummary.errors || 0;
            if ($('userJobsBadge')) $('userJobsBadge').textContent = `${mySummary.total || 0} Total`;

            const host = $('jobList');
            if (!jobs.length) {
                host.innerHTML = '<p class="muted">No jobs yet.</p>';
                $('jobDetailCard').classList.add('hidden');
                state.selectedJobId = null;
                return;
            }
            host.innerHTML = '';
            jobs.sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''));
            jobs.forEach(j => {
                const done = j.completed || 0;
                const total = j.total || 0;
                const pct = total ? Math.round(100 * done / total) : 0;
                const row = el('div', 'job-item');

                const ownerBadge = state.currentUser && state.currentUser.role === 'admin'
                    ? `<span class="badge badge-mute small" style="margin-left:6px;">👤 ${j.owner || 'user1'}</span>`
                    : '';

                let actionButtons = '';
                if (j.status === 'running' || j.status === 'queued') {
                    actionButtons = `
                        <button class="btn btn-ghost btn-sm" title="Pause job" data-action="pause" data-job="${j.job_id}">⏸</button>
                        <button class="btn btn-ghost btn-sm" title="Kill job" style="color:#fca5a5" data-action="kill" data-job="${j.job_id}">✕</button>
                    `;
                } else if (j.status === 'paused') {
                    actionButtons = `
                        <button class="btn btn-ghost btn-sm" title="Resume job" style="color:#6ee7b7" data-action="resume" data-job="${j.job_id}">▶</button>
                        <button class="btn btn-ghost btn-sm" title="Kill job" style="color:#fca5a5" data-action="kill" data-job="${j.job_id}">✕</button>
                    `;
                }

                row.innerHTML = `
                    <div class="job-item-left">
                        <div class="job-id">${j.job_id} ${ownerBadge}</div>
                        <div class="job-meta">${j.created_at}  schema: ${j.schema_id || '--'}  ${done}/${total} docs</div>
                    </div>
                    <div class="job-item-right">
                        <div class="progress-bar-wrap"><div class="progress-bar-fill" style="width:${pct}%"></div></div>
                        <span class="badge ${jobStatusBadgeClass(j.status)}">${j.status}</span>
                        <div class="job-item-actions">${actionButtons}</div>
                        <span class="muted small">--&gt;</span>
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
                    loadJobDetail(j.job_id);
                });
                host.appendChild(row);
            });

            if (state.selectedJobId) {
                const stillExists = jobs.some(j => j.job_id === state.selectedJobId);
                if (stillExists) {
                    loadJobDetail(state.selectedJobId);
                } else {
                    state.selectedJobId = null;
                    $('jobDetailCard').classList.add('hidden');
                }
            }
        } catch (e) {
            console.warn('loadJobs failed', e);
        }
    }

    async function loadJobDetail(jobId) {
        if (!jobId || jobId === 'undefined') return;
        state.selectedJobId = jobId;
        try {
            const j = await api('/pipeline/jobs/' + jobId);
            $('jobDetailCard').classList.remove('hidden');
            $('detailJobId').textContent = jobId;
            const st = $('detailStatus');
            st.textContent = j.status;
            st.className = 'badge ' + jobStatusBadgeClass(j.status);

            const pauseBtn = $('jobPauseBtn');
            const resumeBtn = $('jobResumeBtn');
            const killBtn = $('jobKillBtn');

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

            pauseBtn.onclick = (e) => pauseJob(jobId, e);
            resumeBtn.onclick = (e) => resumeJob(jobId, e);
            killBtn.onclick = (e) => killJob(jobId, e);

            const host = $('jobDetail');

            const wall = j.wall_time_s ? `${j.wall_time_s.toFixed(1)}s` : '--';
            const sucs = j.successes || [];
            const fails = j.failures || [];

            host.innerHTML = `
                <div class="job-summary-grid">
                    <div class="summary-box"><div class="lbl">Status</div><div class="val">${j.status}</div></div>
                    <div class="summary-box"><div class="lbl">Owner</div><div class="val" style="font-size:13px;">👤 ${j.owner || 'user1'}</div></div>
                    <div class="summary-box"><div class="lbl">Docs</div><div class="val">${sucs.length + fails.length} / ${(j.targets || []).length || 0}</div></div>
                    <div class="summary-box"><div class="lbl">Success</div><div class="val" style="color:#6ee7b7">${sucs.length}</div></div>
                    <div class="summary-box"><div class="lbl">Failed</div><div class="val" style="color:#fca5a5">${fails.length}</div></div>
                    <div class="summary-box"><div class="lbl">Wall time</div><div class="val">${wall}</div></div>
                </div>
                ${sucs.length ? `
                <div class="job-section">
                    <h3> Successful (${sucs.length})</h3>
                    ${sucs.map(s => `
                        <div class="result-row ok">
                            <div>
                                <span class="result-name">${s.pdf}</span>
                                <div class="result-meta">pages: ${s.pages}  chars: ${s.chars}  avg conf: ${(s.avg_conf || 0).toFixed(3)}  ${s.elapsed_s}s</div>
                            </div>
                            <div class="result-meta">--&gt; <code>${s.md}</code></div>
                        </div>
                    `).join('')}
                </div>
                ` : ''}
                ${fails.length ? `
                <div class="job-section">
                    <h3> Failed (${fails.length})</h3>
                    ${fails.map(f => `
                        <div class="result-row fail">
                            <div>
                                <span class="result-name">${f.pdf}</span>
                                <div class="result-error">${f.error_type}: ${f.error}</div>
                            </div>
                            <div class="result-meta">${f.elapsed_s}s</div>
                        </div>
                    `).join('')}
                </div>
                ` : ''}
                <div class="job-section">
                    <h3>Full JSON</h3>
                    <pre>${JSON.stringify(j, null, 2)}</pre>
                </div>
            `;
        } catch (e) {
            console.warn('job detail failed', e);
            $('jobDetailCard').classList.add('hidden');
            state.selectedJobId = null;
        }
    }

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
            let consecutiveErrors = 0;
            const interval = setInterval(async () => {
                polls++;
                try {
                    const j = await api('/pipeline/jobs/' + res.job_id);
                    consecutiveErrors = 0;
                    loadJobs();
                    if (j.status === 'completed' || j.status === 'killed' || polls > 300) {
                        clearInterval(interval);
                        btn.textContent = oldText;
                        btn.disabled = false;
                        updateRunBtn();
                    } else if (j.status === 'paused') {
                        btn.textContent = 'Paused (job ' + res.job_id + ')...';
                    } else if (j.status === 'running') {
                        btn.textContent = 'Running (job ' + res.job_id + ')...';
                    }
                } catch (e) {
                    consecutiveErrors++;
                    if (consecutiveErrors > 10) {
                        clearInterval(interval);
                        btn.textContent = oldText;
                        btn.disabled = false;
                        updateRunBtn();
                    }
                }
            }, 2000);
        } catch (e) {
            btn.textContent = oldText;
            btn.disabled = false;
            alert('Pipeline start failed: ' + e.message);
        }
    });

    // ======================= Admin Tab =======================

    async function loadAdminOverview() {
        try {
            const data = await api('/admin/overview');
            $('adminKpiUsers').textContent = data.total_users;
            $('adminKpiDocs').textContent = data.total_documents_uploaded;
            $('adminKpiSchemas').textContent = data.total_schemas_created;
            $('adminKpiSuccessRate').textContent = `${data.success_rate}%`;
            $('adminKpiJobs').textContent = `${data.total_jobs_executed} total (${data.total_jobs_succeeded} ok, ${data.total_jobs_failed} failed)`;
        } catch (e) {
            console.warn('Admin overview failed:', e);
        }
    }

    async function loadAdminUsers() {
        try {
            const users = await api('/admin/users');
            const tbody = document.querySelector('#adminUsersTable tbody');
            if (!tbody) return;

            if (!users.length) {
                tbody.innerHTML = '<tr><td colspan="7" class="text-center muted">No users found.</td></tr>';
                return;
            }

            tbody.innerHTML = '';
            const userFilter = $('auditFilterUser');
            if (userFilter) {
                userFilter.innerHTML = '<option value="">All Users</option>';
            }

            users.forEach(u => {
                if (userFilter) {
                    const opt = el('option', '', `${u.username} (${u.role})`);
                    opt.value = u.username;
                    userFilter.appendChild(opt);
                }

                const tr = el('tr');
                const roleBadge = u.role === 'admin'
                    ? '<span class="badge badge-primary">Admin</span>'
                    : '<span class="badge badge-mute">Normal</span>';

                tr.innerHTML = `
                    <td><strong>${u.username}</strong><br><small class="muted">${u.full_name}</small></td>
                    <td>${roleBadge}</td>
                    <td><strong>${u.documents_uploaded}</strong></td>
                    <td><strong>${u.schemas_created}</strong></td>
                    <td>${u.jobs_executed} <small class="muted">(${u.jobs_succeeded}✓ / ${u.jobs_failed}✕)</small></td>
                    <td><small class="muted">${u.last_active ? new Date(u.last_active).toLocaleString() : '--'}</small></td>
                    <td><button class="btn btn-outline btn-sm inspect-user-btn" data-user="${u.username}">Inspect</button></td>
                `;
                tbody.appendChild(tr);
            });

            tbody.querySelectorAll('.inspect-user-btn').forEach(b => {
                b.addEventListener('click', () => inspectAdminUser(b.dataset.user));
            });
        } catch (e) {
            console.warn('Admin users failed:', e);
        }
    }

    async function inspectAdminUser(username) {
        try {
            const data = await api(`/admin/user/${username}`);
            const card = $('adminUserDetailCard');
            const host = $('adminUserDetailContent');
            $('adminDetailUsername').textContent = `${username} (${data.user.full_name})`;
            card.classList.remove('hidden');

            const s = data.stats || {};
            const schemas = data.schemas || [];
            const recent = data.recent_activity || [];

            host.innerHTML = `
                <div class="job-summary-grid">
                    <div class="summary-box"><div class="lbl">Role</div><div class="val">${data.user.role}</div></div>
                    <div class="summary-box"><div class="lbl">Docs Uploaded</div><div class="val">${s.documents_uploaded || 0}</div></div>
                    <div class="summary-box"><div class="lbl">Schemas Created</div><div class="val">${s.schemas_created || 0}</div></div>
                    <div class="summary-box"><div class="lbl">Jobs Executed</div><div class="val">${s.jobs_executed || 0}</div></div>
                </div>

                <div class="mt-2">
                    <h3>User Schemas (${schemas.length})</h3>
                    ${schemas.length ? `
                        <div class="table-wrap mt-1">
                            <table class="admin-table">
                                <thead><tr><th>Schema ID</th><th>Document Type</th><th>Fields</th><th>Confirmed At</th></tr></thead>
                                <tbody>
                                    ${schemas.map(sc => `
                                        <tr>
                                            <td><code>${sc.schema_id}</code></td>
                                            <td><strong>${sc.document_type}</strong></td>
                                            <td>${sc.field_count} fields</td>
                                            <td><small class="muted">${sc.confirmed_at || '--'}</small></td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    ` : '<p class="muted small">No schemas confirmed by this user yet.</p>'}
                </div>

                <div class="mt-2">
                    <h3>Recent User Activity Timeline (${recent.length})</h3>
                    ${recent.length ? `
                        <div class="table-wrap mt-1">
                            <table class="admin-table">
                                <thead><tr><th>Time</th><th>Action</th><th>Details</th></tr></thead>
                                <tbody>
                                    ${recent.map(r => `
                                        <tr>
                                            <td><small class="muted">${new Date(r.timestamp).toLocaleTimeString()}</small></td>
                                            <td><span class="badge-action ${getActionBadgeClass(r.action)}">${r.action}</span></td>
                                            <td><small>${escapeHtml(JSON.stringify(r.details))}</small></td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    ` : '<p class="muted small">No recent activity recorded for this user.</p>'}
                </div>
            `;
            card.scrollIntoView({ behavior: 'smooth' });
        } catch (e) {
            alert('Failed to load user details: ' + e.message);
        }
    }

    $('closeAdminUserDetailBtn').addEventListener('click', () => {
        $('adminUserDetailCard').classList.add('hidden');
    });

    function getActionBadgeClass(action) {
        if (action.includes('LOGIN') || action.includes('REGISTER')) return 'badge-action-login';
        if (action.includes('UPLOAD')) return 'badge-action-upload';
        if (action.includes('SCHEMA')) return 'badge-action-schema';
        if (action.includes('JOB')) return 'badge-action-job';
        return 'badge-action-error';
    }

    async function loadAdminAudit() {
        const user = $('auditFilterUser') ? $('auditFilterUser').value : '';
        const action = $('auditFilterAction') ? $('auditFilterAction').value : '';
        try {
            let url = '/admin/activity?limit=100';
            if (user) url += `&username=${encodeURIComponent(user)}`;
            if (action) url += `&action=${encodeURIComponent(action)}`;

            const activities = await api(url);
            const host = $('adminAuditList');
            if (!host) return;

            if (!activities.length) {
                host.innerHTML = '<tr><td colspan="5" class="text-center muted">No audit records found matching filters.</td></tr>';
                return;
            }

            host.innerHTML = '';
            activities.forEach(a => {
                const tr = el('tr');
                const badgeClass = getActionBadgeClass(a.action);
                const statusBadge = a.status === 'success'
                    ? '<span class="badge badge-success small">OK</span>'
                    : `<span class="badge badge-danger small">${a.status}</span>`;

                tr.innerHTML = `
                    <td><small class="muted">${new Date(a.timestamp).toLocaleString()}</small></td>
                    <td><strong>${a.username}</strong> <small class="muted">(${a.role})</small></td>
                    <td><span class="badge-action ${badgeClass}">${a.action}</span></td>
                    <td><code>${escapeHtml(JSON.stringify(a.details))}</code></td>
                    <td style="text-align: center;">${statusBadge}</td>
                `;
                host.appendChild(tr);
            });
        } catch (e) {
            console.warn('Admin audit feed failed:', e);
        }
    }

    async function loadAdminLogs() {
        try {
            const data = await api('/admin/logs?limit=150');
            const pre = $('systemLogsContent');
            if (!pre) return;
            if (!data.logs || !data.logs.length) {
                pre.textContent = 'No system logs recorded yet.';
            } else {
                pre.textContent = data.logs.join('\n');
                pre.scrollTop = pre.scrollHeight;
            }
        } catch (e) {
            console.warn('Admin logs failed:', e);
        }
    }

    $('refreshAdminBtn').addEventListener('click', () => {
        loadAdminOverview();
        loadAdminUsers();
    });
    $('refreshAuditBtn').addEventListener('click', loadAdminAudit);
    $('refreshSystemLogsBtn').addEventListener('click', loadAdminLogs);

    if ($('auditFilterUser')) $('auditFilterUser').addEventListener('change', loadAdminAudit);
    if ($('auditFilterAction')) $('auditFilterAction').addEventListener('change', loadAdminAudit);

    // ======================= Init =======================

    initTheme();
    initAuth();
    checkHealth();
    loadDocuments();
    loadSchemas();
    loadPipelineStatus();
    loadJobs();
    setInterval(checkHealth, 15000);

    // Live background updater: refreshes job status & logs automatically
    setInterval(async () => {
        const activeTab = document.querySelector('.tab-btn.active');
        if (activeTab && activeTab.dataset.tab === 'chatbot') {
            await loadJobs();
        } else if (activeTab && activeTab.dataset.tab === 'admin' && state.currentUser && state.currentUser.role === 'admin') {
            await loadAdminOverview();
            await loadAdminAudit();
        }
    }, 3000);

    setTimeout(() => {
        if (!state.sessionId) newSession();
    }, 400);

})();
