// app.js
const API_BASE = '/api';

// State
let patients = [];
let currentPatientId = null;
let chatHistory = [];

// DOM Elements
const patientListEl = document.getElementById('patient-list');
const btnAddPatient = document.getElementById('btn-add-patient');
const btnDeletePatient = document.getElementById('btn-delete-patient');
const currentPatientNameEl = document.getElementById('current-patient-name');
const workspace = document.getElementById('workspace');
const emptyState = document.getElementById('empty-state');

const uploadForm = document.getElementById('upload-form');
const noteUploadInput = document.getElementById('note-upload');
const fileNameDisplay = document.getElementById('file-name-display');
const uploadStatus = document.getElementById('upload-status');
const notesListEl = document.getElementById('notes-list');

const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatHistoryEl = document.getElementById('chat-history');
const btnSendChat = document.getElementById('btn-send-chat');

const modalOverlay = document.getElementById('modal-overlay');
const modalInput = document.getElementById('modal-input');
const btnModalCancel = document.getElementById('btn-modal-cancel');
const btnModalConfirm = document.getElementById('btn-modal-confirm');

// Initialization
async function init() {
    await fetchPatients();
    setupEventListeners();
}

// API Calls
async function api(endpoint, options = {}) {
    try {
        const res = await fetch(`${API_BASE}${endpoint}`, options);
        if (!res.ok) {
            let detail = res.statusText;
            try {
                const body = await res.json();
                detail = body.detail || detail;
            } catch (e) {}
            throw new Error(detail);
        }
        if (res.status === 204) return null;
        return await res.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// Patient Management
async function fetchPatients() {
    try {
        patients = await api('/patients');
        renderPatients();
    } catch (e) {
        console.error('Failed to load patients', e);
    }
}

function renderPatients() {
    patientListEl.innerHTML = '';
    patients.forEach(p => {
        const li = document.createElement('li');
        li.className = `list-item ${p.id === currentPatientId ? 'active' : ''}`;
        li.textContent = p.name;
        li.onclick = () => selectPatient(p.id, p.name);
        patientListEl.appendChild(li);
    });
}

async function selectPatient(id, name) {
    currentPatientId = id;
    currentPatientNameEl.textContent = name;
    btnDeletePatient.style.display = 'block';
    
    emptyState.style.display = 'none';
    workspace.style.display = 'flex';
    
    // Reset state for new patient
    chatHistory = [];
    chatHistoryEl.innerHTML = `
        <div class="message system">
            Hello! I'm ready to answer questions about ${name}'s notes.
        </div>
    `;
    uploadStatus.textContent = '';
    
    renderPatients();
    await fetchNotes();
}

async function createPatient(name) {
    try {
        const newPatient = await api('/patients', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
        await fetchPatients();
        selectPatient(newPatient.id, newPatient.name);
    } catch (e) {
        alert('Failed to create patient: ' + e.message);
    }
}

async function deletePatient() {
    if (!currentPatientId) return;
    if (!confirm('Are you sure you want to delete this patient?')) return;
    
    try {
        await api(`/patients/${currentPatientId}`, { method: 'DELETE' });
        currentPatientId = null;
        currentPatientNameEl.textContent = 'Select a Patient';
        btnDeletePatient.style.display = 'none';
        workspace.style.display = 'none';
        emptyState.style.display = 'flex';
        await fetchPatients();
    } catch (e) {
        alert('Failed to delete patient: ' + e.message);
    }
}

// Notes Management
async function fetchNotes() {
    if (!currentPatientId) return;
    try {
        const notes = await api(`/patients/${currentPatientId}/notes`);
        renderNotes(notes);
    } catch (e) {
        console.error('Failed to load notes', e);
    }
}

function renderNotes(notes) {
    notesListEl.innerHTML = '';
    if (notes.length === 0) {
        notesListEl.innerHTML = '<li class="note-item" style="color: var(--text-muted); justify-content: center;">No notes ingested yet.</li>';
        return;
    }
    notes.forEach(note => {
        const li = document.createElement('li');
        li.className = 'note-item';
        
        let statusClass = 'completed';
        if (note.status === 'failed') statusClass = 'failed';
        else if (note.status === 'processing') statusClass = '';

        li.innerHTML = `
            <span title="${note.filename}">${note.filename.length > 25 ? note.filename.substring(0, 25) + '...' : note.filename}</span>
            <span class="note-status ${statusClass}">${note.status}</span>
        `;
        if (note.error) {
            li.title = note.error;
        }
        notesListEl.appendChild(li);
    });
}

async function uploadNote(e) {
    e.preventDefault();
    if (!currentPatientId || !noteUploadInput.files[0]) return;
    
    const file = noteUploadInput.files[0];
    const formData = new FormData();
    formData.append('patient_id', currentPatientId);
    formData.append('file', file);
    
    uploadStatus.textContent = 'Uploading and processing...';
    uploadStatus.style.color = 'var(--text-main)';
    
    try {
        const res = await fetch(`${API_BASE}/ingest`, {
            method: 'POST',
            body: formData
        });
        
        const data = await res.json();
        
        if (!res.ok) {
            throw new Error(data.detail || res.statusText);
        }
        
        if (data.status === 'failed') {
             uploadStatus.textContent = `Failed: ${data.error}`;
             uploadStatus.style.color = 'var(--danger)';
        } else {
             uploadStatus.textContent = `Success! Chunked into ${data.chunk_count} parts.`;
             uploadStatus.style.color = '#34d399';
        }
        
        uploadForm.reset();
        fileNameDisplay.textContent = 'Choose a PDF or TXT file...';
        await fetchNotes();
    } catch (error) {
        uploadStatus.textContent = `Error: ${error.message}`;
        uploadStatus.style.color = 'var(--danger)';
    }
}

// Chat Management
function appendMessage(role, content, citations = null, safetyHits = null) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;
    
    let htmlContent = content.replace(/\n/g, '<br>');
    msgDiv.innerHTML = `<div>${htmlContent}</div>`;
    
    if (citations && citations.length > 0) {
        const citHtml = citations.map((c, i) => `<span title="${c.note_id} (Score: ${c.score.toFixed(2)})">[Source ${i+1}]</span>`).join(' ');
        msgDiv.innerHTML += `<div class="citations">Sources: ${citHtml}</div>`;
    }
    
    if (safetyHits && safetyHits.length > 0) {
        const hitsHtml = safetyHits.map(h => `<span>⚠️ Flagged: ${h.category} (${h.matched_term})</span>`).join('');
        msgDiv.innerHTML += `<div class="safety-warning">${hitsHtml}</div>`;
    }
    
    chatHistoryEl.appendChild(msgDiv);
    chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
}

async function sendChat(e) {
    e.preventDefault();
    if (!currentPatientId) return;
    
    const question = chatInput.value.trim();
    if (!question) return;
    
    appendMessage('user', question);
    chatInput.value = '';
    btnSendChat.disabled = true;
    
    try {
        const res = await api('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                patient_id: currentPatientId,
                question: question,
                chat_history: chatHistory
            })
        });
        
        appendMessage('assistant', res.answer, res.citations, res.safety_hits);
        
        // Update history
        chatHistory.push({ role: 'user', content: question });
        chatHistory.push({ role: 'assistant', content: res.answer });
        
    } catch (error) {
        appendMessage('system', `Error: ${error.message}`);
    } finally {
        btnSendChat.disabled = false;
        chatInput.focus();
    }
}

// Event Listeners
function setupEventListeners() {
    // Modal
    btnAddPatient.onclick = () => {
        modalInput.value = '';
        modalOverlay.style.display = 'flex';
        modalInput.focus();
    };
    
    btnModalCancel.onclick = () => {
        modalOverlay.style.display = 'none';
    };
    
    btnModalConfirm.onclick = () => {
        const name = modalInput.value.trim();
        if (name) {
            createPatient(name);
            modalOverlay.style.display = 'none';
        }
    };
    
    modalInput.onkeypress = (e) => {
        if (e.key === 'Enter') btnModalConfirm.click();
    };
    
    // Other
    btnDeletePatient.onclick = deletePatient;
    
    uploadForm.onsubmit = uploadNote;
    noteUploadInput.onchange = () => {
        if (noteUploadInput.files.length > 0) {
            fileNameDisplay.textContent = noteUploadInput.files[0].name;
        } else {
            fileNameDisplay.textContent = 'Choose a PDF or TXT file...';
        }
    };
    
    chatForm.onsubmit = sendChat;
}

// Start
init();
