(function () {
    'use strict';

    const API_BASE = window.location.origin;

    let state = {
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
        const resp = await fetch(url, {
            headers: opts.json ? { 'Content-Type': 'application/json' } : undefined,
            ...opts,
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
            errs.classList.add('hidden');
            $('confirmBtn').disabled = true;
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
            errs.innerHTML = '<strong>⚠️ Schema issues:</strong><ul>' + errors.map(e => '<li>' + escapeHtml(e) + '</li>').join('') + '</ul>';
            errs.classList.remove('hidden');
        } else {
            errs.classList.add('hidden');
        }

        const canConfirm = (state.currentState === 'REVIEW' || state.currentState === 'START') && !errors.length && (schema.fields || []).length > 0 && !!schema.document_type;
        $('confirmBtn').disabled = state.completed || !canConfirm;
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
                inputs[inputs.length - 1].focus();
                inputs[inputs.length - 1].select();
            }
        }, 50);
    }

    function renderSchemaPanel() {
        const schema = state.currentSchema;
        const panel = $('schemaPanel');
        const errs = $('schemaErrors');
        const addBtn = $('addSchemaFieldBtn');

        if (!schema) {
            panel.innerHTML = '<p class="muted">No schema yet. Start a session, upload samples, or click "+ Add Field".</p>';
            errs.classList.add('hidden');
            $('copySchemaBtn').disabled = true;
            $('printSchemaBtn').disabled = true;
            $('confirmBtn').disabled = true;
            if (addBtn) addBtn.disabled = !state.sessionId;
            return;
        }

        $('copySchemaBtn').disabled = false;
        $('printSchemaBtn').disabled = false;
        if (addBtn) addBtn.disabled = state.completed;

        const docType = schema.document_type || '';
        const fields = schema.fields || [];

        const typeOptions = [
            'string',
            'number',
            'integer',
            'boolean',
            'date',
            'array[string]',
            'array[object]',
            'object'
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
        $('chatInput').disabled = false;
        $('sendBtn').disabled = false;
        if (data.completed) {
            showStatus('confirmStatus', ` Schema confirmed! schema_id = <code>${data.schema_id}</code>. Saved to schema_registry/.`, 'success');
            // Show the "run pipeline now" shortcut box
            $('postConfirmBox').classList.remove('hidden');
            $('quickRunPipelineBtn').disabled = false;
            // Refresh schemas + pipeline status in background so dropdown is ready
            setTimeout(() => {
                loadSchemas();
                loadPipelineStatus();
                if (state.docs.length === 0) loadDocuments().then(renderDocChecklist);
                else renderDocChecklist();
            }, 300);
        } else {
            $('postConfirmBox').classList.add('hidden');
        }
    }

    async function newSession() {
        try {
            state.lastConfirmedSchemaId = null;
            $('postConfirmBox').classList.add('hidden');
            hideStatus('confirmStatus');
            const data = await api('/chat', { method: 'POST', json: { session_id: null, message: null } });
            state.sessionId = data.session_id;
            $('chatLog').innerHTML = '';
            handleChatResponse(data);
            $('chatInput').focus();
        } catch (e) {
            addChatMsg('bot', 'Failed to start session: ' + e.message);
        }
    }

    $('newSessionBtn').addEventListener('click', newSession);

    async function quickRunPipeline() {
        const schemaId = state.lastConfirmedSchemaId;
        if (!schemaId) {
            alert('No confirmed schema yet. Confirm a schema in the chat first.');
            return;
        }
        // Pre-select the schema, check all docs, switch tab, then fire
        if (!state.docs.length) {
            await loadDocuments();
        }
        await loadSchemas();
        await loadPipelineStatus();
        document.querySelectorAll('.doc-check').forEach(c => c.checked = true);
        const sel = $('schemaSelect');
        if (![...sel.options].map(o => o.value).includes(schemaId)) {
            alert(`Schema ${schemaId} not yet loaded in the dropdown. Try again in a second.`);
            return;
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

    async function sendChat() {
        const msg = $('chatInput').value.trim();
        if (!msg || !state.sessionId) return;
        $('chatInput').value = '';
        addChatMsg('user', msg);
        $('chatInput').disabled = true;
        $('sendBtn').disabled = true;
        try {
            const data = await api('/chat', {
                method: 'POST',
                json: { session_id: state.sessionId, message: msg }
            });
            handleChatResponse(data);
        } catch (e) {
            addChatMsg('bot', 'Error: ' + e.message);
            $('chatInput').disabled = false;
            $('sendBtn').disabled = false;
        }
    }

    $('sendBtn').addEventListener('click', sendChat);
    $('chatInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') sendChat();
    });

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

    $('copySchemaBtn').addEventListener('click', async () => {
        if (!state.currentSchema) return;
        try {
            await navigator.clipboard.writeText(JSON.stringify(state.currentSchema, null, 2));
            const old = $('copySchemaBtn').textContent;
            $('copySchemaBtn').textContent = ' Copied!';
            setTimeout(() => { $('copySchemaBtn').textContent = old; }, 1500);
        } catch (e) {
            addChatMsg('bot', 'Copy failed: ' + e.message);
        }
    });

    $('confirmBtn').addEventListener('click', async () => {
        if (!state.sessionId || state.completed) return;
        $('chatInput').value = '/confirm';
        sendChat();
    });

    const addSchemaFieldHeaderBtn = $('addSchemaFieldBtn');
    if (addSchemaFieldHeaderBtn) {
        addSchemaFieldHeaderBtn.addEventListener('click', addNewSchemaField);
    }

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

    // Sample inference upload
    (function setupInferUpload() {
        const input = $('inferInput');
        const browse = $('inferBrowseBtn');
        const sel = $('inferSelected');
        const run = $('inferBtn');
        let files = [];

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
                if (!$('chatLog').children.length) {
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

    // ======================= Pipeline Tab =======================

    function renderDocChecklist() {
        const host = $('docChecklist');
        const allBtn = $('selAllBtn');
        const noneBtn = $('selNoneBtn');
        if (!state.docs.length) {
            host.innerHTML = '<p class="muted small">Load documents from the Documents tab first.</p>';
            allBtn.disabled = true;
            noneBtn.disabled = true;
            return;
        }
        allBtn.disabled = false;
        noneBtn.disabled = false;
        host.innerHTML = '';
        state.docs.forEach(d => {
            const row = el('label', 'check-item');
            row.innerHTML = `
                <input type="checkbox" class="doc-check" value="${d.name}" checked>
                <span class="cname">${d.name}</span>
                <span class="cmeta">${fmtSize(d.size)}</span>
                <span class="cmeta">${d.has_output ? 're-run' : 'pending'}</span>
            `;
            host.appendChild(row);
        });
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
        const anyChecked = document.querySelectorAll('.doc-check:checked').length > 0;
        const sel = $('schemaSelect');
        $('runPipelineBtn').disabled = !(anyChecked && state.pipelineAvailable && !!sel.value);
    }

    $('docChecklist').addEventListener('change', updateRunBtn);

    async function loadSchemas() {
        try {
            const data = await api('/schemas');
            state.schemas = data.schemas || [];
            const sel = $('schemaSelect');
            if (!state.schemas.length) {
                sel.innerHTML = '<option value="">-- no confirmed schemas --</option>';
                sel.disabled = true;
            } else {
                sel.innerHTML = '<option value="">-- select a schema --</option>' +
                    state.schemas.map(s => `<option value="${s.schema_id}">
                        ${s.document_type || 'untitled'}  ${s.field_count} fields  ${s.schema_id.slice(0, 12)}
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

    $('refreshSchemasBtn').addEventListener('click', loadSchemas);

    async function loadPipelineStatus() {
        try {
            const data = await api('/pipeline/status');
            state.pipelineAvailable = !!data.available;
            const badge = $('pipelineAvail');
            if (data.available) {
                badge.textContent = ' available  ' + (data.routing_mode || '');
                badge.className = 'badge badge-success';
            } else {
                badge.textContent = ' unavailable';
                badge.className = 'badge badge-danger';
            }
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
            const host = $('jobList');
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
                        <div class="job-id">${j.job_id}</div>
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
            if (state.selectedJobId) loadJobDetail(state.selectedJobId);
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
                    <div class="summary-box"><div class="lbl">Docs</div><div class="val">${sucs.length + fails.length} / ${(j.targets || []).length || 0}</div></div>
                    <div class="summary-box"><div class="lbl">Success</div><div class="val" style="color:#6ee7b7">${sucs.length}</div></div>
                    <div class="summary-box"><div class="lbl">Failed</div><div class="val" style="color:#fca5a5">${fails.length}</div></div>
                    <div class="summary-box"><div class="lbl">Wall time</div><div class="val">${wall}</div></div>
                </div>
                ${sucs.length ? `
                <div class="job-section">
                    <h3> Successful (${sucs.length})</h3>
                    ${sucs.map(s => `
                        <div class="result-row ok" style="flex-direction: column; align-items: stretch; gap: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span class="result-name">${s.pdf}</span>
                                    <div class="result-meta">pages: ${s.pages} &nbsp;•&nbsp; conf: ${(s.avg_conf || 0).toFixed(3)} &nbsp;•&nbsp; Layer 1+2: ${s.elapsed_s}s ${s.extract_elapsed_s ? `&nbsp;•&nbsp; Layer 3 (Sarvam 105B): ${s.extract_elapsed_s}s` : ''}</div>
                                </div>
                                <div style="display: flex; gap: 6px; align-items: center;">
                                    ${s.db_run_id ? `<span class="badge badge-success">PostgreSQL: Run #${s.db_run_id}</span>` : ''}
                                    <span class="result-meta"><code>${s.md}</code></span>
                                </div>
                            </div>
                            ${s.extracted_data ? `
                            <div style="background: rgba(0,0,0,0.25); border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 10px 14px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                    <strong style="color: #38bdf8; font-size: 12.5px;">⚡ Layer 3 Extracted JSON (${s.extracted_json})</strong>
                                    <button class="btn btn-sm btn-ghost" onclick="navigator.clipboard.writeText(JSON.stringify(${JSON.stringify(s.extracted_data).replace(/"/g, '&quot;')}, null, 2)); this.textContent='✓ Copied!'; setTimeout(()=>this.textContent='Copy JSON', 1500)">Copy JSON</button>
                                </div>
                                <pre style="margin: 0; font-size: 12px; color: #a5f3fc; max-height: 160px; overflow: auto; background: transparent; border: none; padding: 0;">${JSON.stringify(s.extracted_data, null, 2)}</pre>
                            </div>
                            ` : ''}
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



    // ======================= Init =======================

    initTheme();
    checkHealth();
    loadDocuments();
    loadSchemas();
    loadPipelineStatus();
    loadJobs();
    setInterval(checkHealth, 15000);

    setTimeout(() => {
        if (!state.sessionId) newSession();
    }, 400);

})();


