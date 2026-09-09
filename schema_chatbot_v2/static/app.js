(function () {
    'use strict';

    const API_BASE = window.location.origin;
    const TOKEN_KEY = 'idp_auth_token';
    const ROLE_KEY = 'idp_auth_role';

    let state = {
        token: sessionStorage.getItem(TOKEN_KEY) || null,
        role: sessionStorage.getItem(ROLE_KEY) || null,
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
        sessionStorage.removeItem(TOKEN_KEY);
        sessionStorage.removeItem(ROLE_KEY);
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
            sessionStorage.setItem(TOKEN_KEY, state.token);
            sessionStorage.setItem(ROLE_KEY, state.role);
            localStorage.removeItem(TOKEN_KEY);
            localStorage.removeItem(ROLE_KEY);

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
                loadAllExtractedData();
            } else {
                loadUserDocuments();
                loadUserJobs();
                loadAllExtractedData();
            }

            initSchemaBuilder();
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

    // ======================= Interactive Schema Builder =======================

    const SCHEMA_PRESETS = {
        medical: {
            document_type: 'medical_discharge_summary',
            fields: [
                { name: 'patient_name', type: 'string', required: true, description: 'Full legal name of the patient' },
                { name: 'patient_id', type: 'string', required: false, description: 'Hospital patient identifier or MRN' },
                { name: 'admission_date', type: 'date', required: true, description: 'Date patient was admitted to hospital' },
                { name: 'discharge_date', type: 'date', required: false, description: 'Date patient was discharged from hospital' },
                { name: 'primary_diagnosis', type: 'string', required: true, description: 'Primary diagnosis or clinical condition diagnosed' },
                { name: 'procedure_performed', type: 'string', required: false, description: 'Surgical or medical procedure performed' },
                { name: 'approved_claim_amount', type: 'number', required: false, description: 'Total approved insurance claim settlement amount' },
            ]
        },
        invoice: {
            document_type: 'invoice',
            fields: [
                { name: 'invoice_number', type: 'string', required: true, description: 'Unique invoice reference or billing number' },
                { name: 'invoice_date', type: 'date', required: true, description: 'Date the invoice was issued' },
                { name: 'vendor_name', type: 'string', required: true, description: 'Name of the issuing vendor or supplier' },
                { name: 'total_amount', type: 'number', required: true, description: 'Total invoice amount payable including all taxes' },
                { name: 'tax_amount', type: 'number', required: false, description: 'Total tax or VAT amount charged' },
                { name: 'line_items', type: 'array', item_type: 'object', required: false, description: 'List of individual itemized charges or services' },
            ]
        },
        resume: {
            document_type: 'resume',
            fields: [
                { name: 'candidate_name', type: 'string', required: true, description: 'Full legal name of the job candidate' },
                { name: 'email_address', type: 'string', required: true, description: 'Primary contact email address' },
                { name: 'phone_number', type: 'string', required: false, description: 'Contact phone or mobile number' },
                { name: 'skills', type: 'array', item_type: 'string', required: false, description: 'List of technical and professional skills' },
                { name: 'years_of_experience', type: 'number', required: false, description: 'Total years of relevant professional experience' },
            ]
        },
        blank: {
            document_type: '',
            fields: [
                { name: 'field_1', type: 'string', required: true, description: '' }
            ]
        }
    };

    function collectSchemaFromInputs() {
        if (!state.currentSchema) {
            state.currentSchema = { document_type: '', fields: [] };
        }
        const docTypeInput = $('editDocType');
        if (docTypeInput) {
            state.currentSchema.document_type = docTypeInput.value.trim().toLowerCase().replace(/[\s-]+/g, '_');
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
        const docType = schema.document_type || ($('editDocType') ? $('editDocType').value.trim() : '');
        if (!docType) {
            errors.push('Document Type is required and cannot be empty.');
        }

        const fields = schema.fields || [];
        if (!fields.length) {
            errors.push('Schema must contain at least one field.');
        } else {
            const seen = new Set();
            fields.forEach((f, idx) => {
                if (!f.name || !f.name.trim()) {
                    errors.push(`Row ${idx + 1}: Field name cannot be blank.`);
                } else {
                    const norm = f.name.trim().toLowerCase().replace(/[\s-]+/g, '_');
                    if (seen.has(norm)) {
                        errors.push(`Duplicate field name: '${f.name}' (row ${idx + 1}).`);
                    }
                    seen.add(norm);
                }
                if (f.type === 'array' && !f.item_type) {
                    errors.push(`Field '${f.name}' is an array but has no item_type.`);
                }
            });
        }

        state.currentErrors = errors;
        if (errors.length) {
            if (errs) {
                errs.innerHTML = '<strong>⚠️ Schema issues:</strong><ul>' + errors.map(e => '<li>' + escapeHtml(e) + '</li>').join('') + '</ul>';
                errs.classList.remove('hidden');
            }
        } else {
            if (errs) errs.classList.add('hidden');
        }

        const canConfirm = !errors.length && fields.length > 0 && !!docType;
        if ($('confirmBtn')) $('confirmBtn').disabled = !canConfirm;
    }

    function addNewSchemaField() {
        if (!state.currentSchema) {
            state.currentSchema = { document_type: '', fields: [] };
        }
        const count = (state.currentSchema.fields || []).length + 1;
        state.currentSchema.fields.push({
            name: 'field_' + count,
            type: 'string',
            required: false,
            description: ''
        });
        renderSchemaPanel();
        validateAndRefreshUI();
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
        if (!panel) return;

        if (!schema || !schema.fields || !schema.fields.length) {
            panel.innerHTML = `
                <div style="padding: 24px; text-align: center; background: rgba(0,0,0,0.1); border-radius: var(--radius-sm); border: 1px dashed var(--border);">
                    <p class="muted" style="margin-bottom: 8px;">No fields defined yet. Choose a Quick Preset above or click below to add your first field.</p>
                    <button type="button" class="btn btn-outline btn-sm" id="addFieldInlineBtn">➕ Add First Field</button>
                </div>
            `;
            const inlineBtn = $('addFieldInlineBtn');
            if (inlineBtn) inlineBtn.addEventListener('click', addNewSchemaField);
            validateAndRefreshUI();
            return;
        }

        const fields = schema.fields;
        const typeOptions = [
            'string', 'number', 'integer', 'boolean', 'date',
            'array[string]', 'array[object]', 'object'
        ];

        let html = `
            <table class="schema-table">
                <thead>
                    <tr>
                        <th style="width: 26%;">Field Name</th>
                        <th style="width: 22%;">Type</th>
                        <th style="width: 14%; text-align: center;">Required</th>
                        <th style="width: 32%;">Extraction Guidance / Description</th>
                        <th style="width: 6%; text-align: center;">Action</th>
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
                        <input type="text" class="schema-input field-name-input" data-field="name" value="${escapeHtml(f.name || '')}" placeholder="e.g. patient_name">
                    </td>
                    <td>
                        <select class="schema-select field-type-select" data-field="type">
                            ${opts}
                        </select>
                    </td>
                    <td style="text-align: center;">
                        <button type="button" class="req-toggle ${isReq ? 'is-req' : 'is-opt'}" data-idx="${idx}">
                            ${isReq ? 'YES' : 'NO'}
                        </button>
                    </td>
                    <td>
                        <input type="text" class="schema-input field-desc-input" data-field="description" value="${escapeHtml(f.description || '')}" placeholder="Instructions for Layer 3 extraction...">
                    </td>
                    <td style="text-align: center;">
                        <button type="button" class="btn-del-field" data-idx="${idx}" title="Delete field">✕</button>
                    </td>
                </tr>
            `;
        });

        html += `
                </tbody>
            </table>
            <div class="schema-bottom-actions mt-1" style="display: flex; justify-content: space-between; align-items: center;">
                <button type="button" class="btn btn-ghost btn-sm" id="addFieldInlineBtn">➕ Add Field</button>
                <span class="muted small">${fields.length} field(s) configured</span>
            </div>
        `;

        panel.innerHTML = html;
        const inlineBtn = $('addFieldInlineBtn');
        if (inlineBtn) inlineBtn.addEventListener('click', addNewSchemaField);

        validateAndRefreshUI();
    }

    function applyPreset(presetKey) {
        const preset = SCHEMA_PRESETS[presetKey];
        if (!preset) return;
        state.currentSchema = JSON.parse(JSON.stringify(preset));
        state.completed = false;
        const docTypeInput = $('editDocType');
        if (docTypeInput) docTypeInput.value = state.currentSchema.document_type;
        const postBox = $('postConfirmBox');
        if (postBox) postBox.classList.add('hidden');
        renderSchemaPanel();
        validateAndRefreshUI();
    }

    function clearSchema() {
        if (state.currentSchema && state.currentSchema.fields && state.currentSchema.fields.length) {
            if (!confirm('Are you sure you want to clear all fields in the builder?')) return;
        }
        applyPreset('blank');
    }

    async function copySchemaJson() {
        collectSchemaFromInputs();
        if (!state.currentSchema) return;
        try {
            await navigator.clipboard.writeText(JSON.stringify(state.currentSchema, null, 2));
            const btn = $('copySchemaBtn');
            if (btn) {
                const orig = btn.textContent;
                btn.textContent = '✓ Copied!';
                setTimeout(() => { btn.textContent = orig; }, 1500);
            }
        } catch (e) {
            alert('Failed to copy: ' + e.message);
        }
    }

    function toggleJsonPreview() {
        collectSchemaFromInputs();
        const box = $('schemaJsonPreviewBox');
        if (!box) return;
        box.classList.toggle('hidden');
        const view = $('schemaJsonView');
        if (view && state.currentSchema) {
            view.textContent = JSON.stringify(state.currentSchema, null, 2);
        }
    }

    async function saveCustomSchema() {
        collectSchemaFromInputs();
        validateAndRefreshUI();

        const schema = state.currentSchema;
        if (!schema || !schema.document_type || !schema.fields || !schema.fields.length) {
            showStatus('confirmStatus', 'Please specify a document type and at least one field.', 'error');
            return;
        }

        if (state.currentErrors && state.currentErrors.length) {
            showStatus('confirmStatus', 'Please fix schema issues before confirming: ' + state.currentErrors.join(', '), 'error');
            return;
        }

        const btn = $('confirmBtn');
        if (btn) {
            btn.disabled = true;
            btn.textContent = '⏳ Saving & Registering...';
        }

        try {
            const payload = {
                document_type: schema.document_type,
                fields: schema.fields.map(f => ({
                    name: f.name,
                    type: f.type || 'string',
                    required: !!f.required,
                    description: f.description || '',
                    item_type: f.item_type || null,
                    pattern: f.pattern || null,
                    currency: f.currency || null,
                }))
            };

            const data = await api('/schema/custom', {
                method: 'POST',
                json: payload
            });

            state.lastConfirmedSchemaId = data.schema_id;
            state.completed = true;

            showStatus('confirmStatus', `🎉 Schema saved and confirmed! Schema ID: <code>${escapeHtml(data.schema_id)}</code>`, 'success');

            const confIdText = $('confirmedSchemaIdText');
            if (confIdText) confIdText.textContent = `ID: ${data.schema_id}`;

            const postBox = $('postConfirmBox');
            if (postBox) postBox.classList.remove('hidden');

            const quickBtn = $('quickRunPipelineBtn');
            if (quickBtn) quickBtn.disabled = false;

            const downloadPdfBtn = $('downloadSchemaPdfBtn');
            if (downloadPdfBtn) downloadPdfBtn.disabled = false;
            const downloadJsonBtn = $('downloadSchemaJsonBtn');
            if (downloadJsonBtn) downloadJsonBtn.disabled = false;

            // Refresh pipeline schema selector and pre-select the newly created schema
            if (state.role === 'admin') {
                await loadSchemas();
                const sel = $('schemaSelect');
                if (sel) {
                    sel.value = data.schema_id;
                    updateRunBtn();
                }
            }
        } catch (err) {
            showStatus('confirmStatus', 'Failed to save schema: ' + err.message, 'error');
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.textContent = '✓ Save & Confirm Schema';
            }
        }
    }

    function initSchemaBuilder() {
        if (!state.currentSchema) {
            applyPreset('medical');
        } else {
            const docTypeInput = $('editDocType');
            if (docTypeInput && state.currentSchema.document_type) {
                docTypeInput.value = state.currentSchema.document_type;
            }
            renderSchemaPanel();
            validateAndRefreshUI();
        }
    }

    if ($('addSchemaFieldBtn')) $('addSchemaFieldBtn').addEventListener('click', addNewSchemaField);
    if ($('clearSchemaBtn')) $('clearSchemaBtn').addEventListener('click', clearSchema);
    if ($('copySchemaBtn')) $('copySchemaBtn').addEventListener('click', copySchemaJson);
    if ($('toggleJsonPreviewBtn')) $('toggleJsonPreviewBtn').addEventListener('click', toggleJsonPreview);
    if ($('confirmBtn')) $('confirmBtn').addEventListener('click', saveCustomSchema);

    document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const presetKey = e.currentTarget.getAttribute('data-preset');
            if (presetKey) applyPreset(presetKey);
        });
    });

    if ($('editDocType')) {
        $('editDocType').addEventListener('input', () => {
            collectSchemaFromInputs();
            validateAndRefreshUI();
        });
        $('editDocType').addEventListener('change', () => {
            collectSchemaFromInputs();
            validateAndRefreshUI();
        });
    }

    if ($('schemaPanel')) {
        $('schemaPanel').addEventListener('input', () => {
            collectSchemaFromInputs();
            validateAndRefreshUI();
        });
        $('schemaPanel').addEventListener('change', () => {
            collectSchemaFromInputs();
            validateAndRefreshUI();
        });
        $('schemaPanel').addEventListener('click', (e) => {
            const reqToggle = e.target.closest('.req-toggle');
            if (reqToggle) {
                reqToggle.classList.toggle('is-req');
                reqToggle.classList.toggle('is-opt');
                const isReq = reqToggle.classList.contains('is-req');
                reqToggle.textContent = isReq ? 'YES' : 'NO';
                collectSchemaFromInputs();
                validateAndRefreshUI();
                return;
            }

            const delBtn = e.target.closest('.btn-del-field');
            if (delBtn) {
                const tr = delBtn.closest('tr');
                const idx = parseInt(tr.getAttribute('data-idx'), 10);
                if (!isNaN(idx) && state.currentSchema && state.currentSchema.fields) {
                    state.currentSchema.fields.splice(idx, 1);
                    renderSchemaPanel();
                    validateAndRefreshUI();
                }
                return;
            }
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
            if (data.layer3_strategy && $('layer3StrategySelect') && !window.__userChangedStrategy) {
                $('layer3StrategySelect').value = data.layer3_strategy;
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
                    <div class="summary-box"><div class="lbl">Strategy</div><div class="val" style="font-size:12px; font-weight:600; color:${j.strategy === 'graph_memory' ? '#c084fc' : '#94a3b8'}">${j.strategy === 'graph_memory' ? '🧠 Graph Memory' : '📄 Page Scan'}</div></div>
                    <div class="summary-box"><div class="lbl">Docs</div><div class="val">${sucs.length + fails.length} / ${(j.targets || []).length || 0}</div></div>
                    <div class="summary-box"><div class="lbl">Success</div><div class="val" style="color:#6ee7b7">${sucs.length}</div></div>
                    <div class="summary-box"><div class="lbl">Wall time</div><div class="val">${wall}</div></div>
                </div>
                ${sucs.length ? `
                <div class="job-section">
                    <h3>✓ Successful (${sucs.length})</h3>
                    ${sucs.map((s, idx) => `
                        <div class="result-row ok" style="flex-direction: column; align-items: stretch; gap: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                                <div>
                                    <span class="result-name">${escapeHtml(s.pdf)}</span>
                                    <div class="result-meta">pages: ${s.pages} • conf: ${(s.avg_conf || 0).toFixed(3)} • Layer 1+2: ${s.elapsed_s}s ${s.extract_elapsed_s ? `• Layer 3 (${s.strategy || 'extract'}): ${s.extract_elapsed_s}s` : ''}</div>
                                </div>
                                <div style="display: flex; gap: 6px; align-items: center; flex-wrap: wrap;">
                                    ${s.strategy === 'graph_memory'
                                        ? `<span class="badge badge-primary" style="background:rgba(124,58,237,0.2);color:#c084fc;border:1px solid rgba(124,58,237,0.4)">🧠 Graph Memory (${s.graph_nodes || 0} nodes, ${s.graph_edges || 0} edges)</span>`
                                        : `<span class="badge badge-mute">📄 Classic Page Scan</span>`}
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
                            ${s.graph_memory && s.graph_memory.snapshot ? `
                            <details class="graph-memory-details" style="margin-top: 6px; background: rgba(124,58,237,0.08); border: 1px solid rgba(124,58,237,0.25); border-radius: 6px; padding: 8px 12px;">
                                <summary style="cursor: pointer; font-weight: 600; color: #c084fc; font-size: 12px; user-select: none;">
                                    🧠 Document Knowledge Graph (${s.graph_nodes || 0} Entities, ${s.graph_edges || 0} Relationships)
                                </summary>
                                <div style="margin-top: 8px; font-size: 11px;">
                                    <div style="display: flex; gap: 8px; margin-bottom: 6px; flex-wrap: wrap;">
                                        <span class="badge badge-info font-mono">Entities: ${s.graph_nodes || 0}</span>
                                        <span class="badge badge-warn font-mono">Relationships: ${s.graph_edges || 0}</span>
                                        <span class="badge badge-primary font-mono">Strategy: Graph Memory</span>
                                    </div>
                                    <div style="max-height: 220px; overflow-y: auto; background: rgba(0,0,0,0.3); border-radius: 4px; padding: 8px; margin-bottom: 6px;">
                                        <strong>Captured Graph Nodes &amp; Evidence:</strong>
                                        <ul style="margin: 4px 0 0 16px; padding: 0;">
                                            ${(s.graph_memory.snapshot.nodes || []).map(n => `
                                                <li style="margin-bottom: 4px;">
                                                    <span style="color:#c084fc; font-weight:600;">[${escapeHtml(n.type)}]</span>
                                                    <strong>${escapeHtml(n.label || '')}:</strong> <code>${escapeHtml(n.value || '')}</code>
                                                    <span class="muted small">(Pages: ${(n.source_pages || []).join(',')}, conf: ${n.confidence})</span>
                                                    ${(n.evidence && n.evidence.length) ? `<div class="muted small" style="padding-left: 12px; font-style: italic;">"${escapeHtml(n.evidence[0].text || '')}"</div>` : ''}
                                                </li>
                                            `).join('')}
                                        </ul>
                                    </div>
                                    ${(s.graph_memory.snapshot.edges || []).length ? `
                                    <div style="max-height: 180px; overflow-y: auto; background: rgba(0,0,0,0.3); border-radius: 4px; padding: 8px;">
                                        <strong>Entity Relationships &amp; Cross-Page Anaphora:</strong>
                                        <ul style="margin: 4px 0 0 16px; padding: 0;">
                                            ${(s.graph_memory.snapshot.edges || []).map(e => `
                                                <li style="margin-bottom: 4px;">
                                                    <code>${escapeHtml(e.source_node)}</code> --[<strong style="color:#6ee7b7">${escapeHtml(e.relationship)}</strong> (status: ${escapeHtml(e.status)})]--> <code>${escapeHtml(e.target_node)}</code>
                                                    <span class="muted small">(P${e.source_page})</span>
                                                    ${e.evidence ? `<div class="muted small" style="padding-left: 12px; font-style: italic;">"${escapeHtml(e.evidence)}"</div>` : ''}
                                                </li>
                                            `).join('')}
                                        </ul>
                                    </div>
                                    ` : ''}
                                </div>
                            </details>
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
                const strategy = ($('layer3StrategySelect') && $('layer3StrategySelect').value) || 'graph_memory';
                fd.append('strategy', strategy);
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
            const fd = new FormData();
            const strategy = ($('userLayer3StrategySelect') && $('userLayer3StrategySelect').value) || 'graph_memory';
            fd.append('strategy', strategy);
            const res = await api('/me/pipeline/run', { method: 'POST', body: fd });
            showStatus('userAutoRunStatus', `✓ Job ${res.job_id} queued for ${res.targets} document(s).`, 'success');
            loadUserJobs();
        } catch (e) {
            showStatus('userAutoRunStatus', 'Failed to run pipeline: ' + e.message, 'error');
        } finally {
            if (btn) btn.disabled = false;
        }
    }

    if ($('userAutoRunPipelineBtn')) $('userAutoRunPipelineBtn').addEventListener('click', runUserPipeline);
    if ($('layer3StrategySelect')) {
        $('layer3StrategySelect').addEventListener('change', () => { window.__userChangedStrategy = true; });
    }
    if ($('userLayer3StrategySelect')) {
        $('userLayer3StrategySelect').addEventListener('change', () => { window.__userChangedStrategy = true; });
    }
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

    // Holds all extracted data and document map
    let qbAllExtractedData = null;
    let qbDocsMap = {};

    function updateQbModeBadge() {
        const docSelect = $('tab_qb_doc_select');
        const modeBadge = $('tab_qb_mode_badge');
        if (!docSelect || !modeBadge) return;
        const selectedDocId = docSelect.value;
        const m = selectedDocId ? qbDocsMap[selectedDocId] : null;
        if (!m) {
            modeBadge.style.display = 'none';
            return;
        }
        const isGraph = m.strategy === 'graph_memory' || m.has_graph;
        modeBadge.style.display = 'inline-block';
        modeBadge.className = 'badge ' + (isGraph ? 'badge-primary' : 'badge-secondary');
        modeBadge.textContent = isGraph ? '⚡ Graph Memory' : '📄 JSON Fallback';
    }

    async function loadAllExtractedData() {
        try {
            const docSelect = $('tab_qb_doc_select');
            const modeBadge = $('tab_qb_mode_badge');

            // 1. Fetch available documents from dedicated query-bot endpoint
            let docs = [];
            try {
                const docRes = await api('/api/query-bot/documents');
                docs = docRes.documents || [];
            } catch (err) {
                console.warn('Failed to load /api/query-bot/documents, fallback to jobs', err);
            }

            // 2. Fetch /pipeline/status for any in-flight active jobs
            const statusData = await api('/pipeline/status').catch(() => ({}));
            const jobs = Object.values(statusData.jobs || {});
            const completedJobs = jobs.filter(j => j.status === 'completed');

            qbDocsMap = {};
            const merged = {};

            // Register documents from /api/query-bot/documents
            docs.forEach(d => {
                qbDocsMap[d.doc_id] = {
                    doc_id: d.doc_id,
                    stem: d.stem,
                    label: d.label || d.doc_id,
                    strategy: d.strategy || (d.has_graph ? 'graph_memory' : 'page_scan'),
                    has_graph: !!d.has_graph,
                    has_extracted: !!d.has_extracted,
                    job_id: d.job_id,
                    extracted_data: null,
                };
            });

            // Merge details from completed in-memory jobs if available
            if (completedJobs.length > 0) {
                const details = await Promise.all(
                    completedJobs.map(j => api('/pipeline/jobs/' + j.job_id).catch(() => null))
                );
                details.forEach(j => {
                    if (!j) return;
                    const sucs = j.successes || [];
                    sucs.forEach(s => {
                        const docId = s.pdf || s.doc_id || (s.extracted_json ? s.extracted_json.replace('.extracted.json', '.pdf') : null);
                        if (docId) {
                            if (!qbDocsMap[docId]) {
                                qbDocsMap[docId] = {
                                    doc_id: docId,
                                    job_id: j.job_id,
                                    strategy: s.strategy || j.strategy || 'graph_memory',
                                    has_graph: (s.strategy || j.strategy) === 'graph_memory',
                                };
                            }
                            qbDocsMap[docId].extracted_data = s.extracted_data;
                        }
                        if (s.extracted_data) {
                            const key = s.extracted_json || s.pdf || s.doc_id || 'document';
                            merged[key] = s.extracted_data;
                        }
                    });
                    if (!sucs.length && j.extracted_data) {
                        merged[j.job_id] = j.extracted_data;
                        qbDocsMap[j.job_id] = {
                            doc_id: j.job_id,
                            job_id: j.job_id,
                            extracted_data: j.extracted_data,
                            strategy: j.strategy || 'graph_memory',
                            has_graph: (j.strategy || 'graph_memory') === 'graph_memory',
                        };
                    }
                });
            }

            const docKeys = Object.keys(qbDocsMap);

            if (!docKeys.length) {
                qbAllExtractedData = null;
                qbDocsMap = {};
                if (docSelect) {
                    docSelect.innerHTML = '<option value="">-- No documents processed yet --</option>';
                }
                if (modeBadge) modeBadge.style.display = 'none';
                return;
            }

            qbAllExtractedData = Object.keys(merged).length ? merged : null;

            if (docSelect) {
                const prevVal = docSelect.value;
                docSelect.innerHTML = docKeys.map(d => {
                    const m = qbDocsMap[d];
                    const stratLabel = (m.strategy === 'graph_memory' || m.has_graph) ? 'graph_memory' : 'page_scan';
                    return `<option value="${escapeHtml(d)}">${escapeHtml(d)} (${stratLabel})</option>`;
                }).join('');

                if (prevVal && qbDocsMap[prevVal]) {
                    docSelect.value = prevVal;
                } else if (docKeys.length > 0) {
                    docSelect.value = docKeys[0];
                }

                updateQbModeBadge();
            }
        } catch (e) {
            console.warn('loadAllExtractedData error', e);
        }
    }

    async function askTabQueryBot() {
        const input = $('tab_qb_input');
        const btn = $('tab_qb_btn');
        const msgs = $('tab_qb_msgs');
        const docSelect = $('tab_qb_doc_select');
        const modeBadge = $('tab_qb_mode_badge');
        if (!input || !btn || !msgs) return;

        const question = input.value.trim();
        if (!question) { input.focus(); return; }

        const selectedDocId = docSelect ? docSelect.value : null;
        const selectedMeta = selectedDocId ? qbDocsMap[selectedDocId] : null;

        if (!selectedDocId) {
            const errDiv = document.createElement('div');
            errDiv.className = 'query-bot-msg bot error';
            errDiv.textContent = 'Please select a document from the dropdown first.';
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
        botDiv.textContent = 'Searching document graph…';
        msgs.appendChild(botDiv);
        msgs.scrollTop = msgs.scrollHeight;

        input.value = '';
        input.disabled = true;
        btn.disabled = true;
        btn.textContent = 'Asking…';

        try {
            const reqPayload = {
                question,
                doc_id: selectedDocId || undefined,
                job_id: selectedMeta ? selectedMeta.job_id : undefined,
                extracted_data: (selectedMeta && selectedMeta.extracted_data) ? selectedMeta.extracted_data : undefined,
            };

            const res = await api('/api/query-bot/ask', {
                method: 'POST',
                json: reqPayload,
            });

            botDiv.className = 'query-bot-msg bot';

            // Build rich answer HTML with source citations & mode badge
            let ansHtml = `<div>${escapeHtml(res.answer || 'No answer returned.')}</div>`;

            if (res.sources && res.sources.length > 0) {
                ansHtml += '<div style="margin-top: 8px; padding-top: 6px; border-top: 1px dashed var(--border-light); font-size: 11px; opacity: 0.85;">';
                res.sources.forEach(s => {
                    const pg = s.page ? `Page ${s.page}` : 'Document';
                    const ev = s.evidence ? `: "${escapeHtml(s.evidence)}"` : '';
                    ansHtml += `<div style="margin-top: 2px;">📍 <strong>[Source: ${escapeHtml(pg)}]</strong>${ev}</div>`;
                });
                ansHtml += '</div>';
            }

            const isGraph = res.mode === 'graph';
            const badgeClass = isGraph ? 'badge-primary' : 'badge-secondary';
            const badgeText = isGraph ? '⚡ Graph Memory' : '📄 JSON Fallback';
            ansHtml += `<div style="margin-top: 6px;"><span class="badge ${badgeClass}" style="font-size: 10px; font-weight: 500;">${badgeText}</span></div>`;

            botDiv.innerHTML = ansHtml;

            if (modeBadge) {
                modeBadge.style.display = 'inline-block';
                modeBadge.className = `badge ${badgeClass}`;
                modeBadge.textContent = badgeText;
            }
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

    if ($('tab_qb_doc_select')) $('tab_qb_doc_select').addEventListener('change', updateQbModeBadge);
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


