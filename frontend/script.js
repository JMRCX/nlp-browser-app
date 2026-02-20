// Configuração
const API_BASE = 'http://localhost:8000';
const topKInput = document.getElementById('topK');
const topKValue = document.getElementById('topKValue');
const promptInput = document.getElementById('prompt');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const resultsSection = document.getElementById('resultsSection');

// Atualizar o valor do slider
topKInput.addEventListener('input', (e) => {
    topKValue.textContent = e.target.value;
});

// Botões
document.getElementById('btnAnalisarCompleta').addEventListener('click', analisarCompleta);
document.getElementById('btnBuscarSimilares').addEventListener('click', buscarSimilares);
document.getElementById('btnClassificar').addEventListener('click', classificar);
document.getElementById('btnSentimento').addEventListener('click', analisarSentimento);
document.getElementById('btnLimpar').addEventListener('click', limpar);

// Função para mostrar loading
function showLoading() {
    loading.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    error.classList.add('hidden');
}

// Função para esconder loading
function hideLoading() {
    loading.classList.add('hidden');
}

// Função para mostrar erro
function showError(mensagem) {
    hideLoading();
    error.classList.remove('hidden');
    error.textContent = '❌ Erro: ' + mensagem;
}

// Validar input
function validateInput() {
    const texto = promptInput.value.trim();
    if (!texto) {
        showError('Por favor, insira um texto para análise');
        return null;
    }
    return texto;
}

// Buscar textos similares
async function buscarSimilares() {
    const prompt = validateInput();
    if (!prompt) return;

    showLoading();
    try {
        const response = await fetch(`${API_BASE}/buscar_similares`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                prompt: prompt, 
                top_k: parseInt(topKInput.value) 
            })
        });

        if (!response.ok) throw new Error(`Erro ${response.status}`);
        const data = await response.json();

        hideLoading();
        displaySimilares(data.textos);
    } catch (err) {
        showError(err.message);
    }
}

// Classificar
async function classificar() {
    const prompt = validateInput();
    if (!prompt) return;

    showLoading();
    try {
        const response = await fetch(`${API_BASE}/classificar`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: prompt })
        });

        if (!response.ok) throw new Error(`Erro ${response.status}`);
        const data = await response.json();

        hideLoading();
        displayClassificacao(data.classificacao);
    } catch (err) {
        showError(err.message);
    }
}

// Analisar Sentimento
async function analisarSentimento() {
    const prompt = validateInput();
    if (!prompt) return;

    showLoading();
    try {
        const response = await fetch(`${API_BASE}/sentimento`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: prompt })
        });

        if (!response.ok) throw new Error(`Erro ${response.status}`);
        const data = await response.json();

        hideLoading();
        displaySentimento(data.sentimento);
    } catch (err) {
        showError(err.message);
    }
}

// Análise Completa
async function analisarCompleta() {
    const prompt = validateInput();
    if (!prompt) return;

    showLoading();
    try {
        const response = await fetch(`${API_BASE}/analise_completa`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                prompt: prompt, 
                top_k: parseInt(topKInput.value) 
            })
        });

        if (!response.ok) throw new Error(`Erro ${response.status}`);
        const data = await response.json();

        hideLoading();
        displayAnalisaCompleta(data.resultado);
    } catch (err) {
        showError(err.message);
    }
}

// Display Similares
function displaySimilares(textos) {
    resultsSection.classList.remove('hidden');
    
    // Limpar outras seções
    document.getElementById('analisaCompleta').classList.add('hidden');
    document.getElementById('textosSimilares').classList.remove('hidden');
    document.getElementById('classificacaoBox').classList.add('hidden');
    document.getElementById('sentimentoBox').classList.add('hidden');

    const content = document.getElementById('similaresContent');
    if (!textos || textos.length === 0) {
        content.innerHTML = '<p>Nenhum resultado encontrado.</p>';
        return;
    }

    content.innerHTML = textos.map((item, idx) => `
        <div class="similar-item">
            <div class="similar-texto"><strong>#${idx + 1}:</strong> "${item.texto}"</div>
            <div class="similar-meta">
                <span><strong>Categoria:</strong> ${item.categoria}</span>
                <span><strong>Similaridade:</strong> ${(item.similitude * 100).toFixed(1)}%</span>
                <span class="badge badge-${item.idioma}">${item.idioma.toUpperCase()}</span>
            </div>
        </div>
    `).join('');
}

// Display Classificacao
function displayClassificacao(classificacao) {
    resultsSection.classList.remove('hidden');
    document.getElementById('analisaCompleta').classList.add('hidden');
    document.getElementById('textosSimilares').classList.add('hidden');
    document.getElementById('classificacaoBox').classList.remove('hidden');
    document.getElementById('sentimentoBox').classList.add('hidden');

    const content = document.getElementById('classificacaoContent');
    
    if (classificacao.erro) {
        content.innerHTML = `<p>Erro: ${classificacao.erro}</p>`;
        return;
    }

    const html = `
        <div style="margin-bottom: 20px;">
            <h3 style="color: var(--primary-color); margin-bottom: 10px;">Categoria Principal</h3>
            <div style="padding: 15px; background: var(--bg-color); border-radius: 8px;">
                <strong style="font-size: 1.2rem; color: var(--success-color);">${classificacao.categoria}</strong>
                <div style="margin-top: 8px; color: var(--text-light);">
                    Confiança: ${(classificacao.confianca * 100).toFixed(1)}%
                </div>
            </div>
        </div>

        <div>
            <h3 style="color: var(--primary-color); margin-bottom: 10px;">Todas as Categorias</h3>
            ${classificacao.todas_categorias.map(cat => {
                const percentage = cat.score * 100;
                return `
                    <div class="classificacao-item">
                        <span class="categoria">${cat.categoria}</span>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${percentage}%"></div>
                        </div>
                        <span class="score">${percentage.toFixed(1)}%</span>
                    </div>
                `;
            }).join('')}
        </div>
    `;

    content.innerHTML = html;
}

// Display Sentimento
function displaySentimento(sentimento) {
    resultsSection.classList.remove('hidden');
    document.getElementById('analisaCompleta').classList.add('hidden');
    document.getElementById('textosSimilares').classList.add('hidden');
    document.getElementById('classificacaoBox').classList.add('hidden');
    document.getElementById('sentimentoBox').classList.remove('hidden');

    const content = document.getElementById('sentimentoContent');
    
    if (sentimento.erro) {
        content.innerHTML = `<p>Erro: ${sentimento.erro}</p>`;
        return;
    }

    let sentimentoClass = 'sentimento-neutro';
    let emoji = '😐';

    if (sentimento.sentimento.includes('Muito Positivo') || sentimento.sentimento.includes('5 stars')) {
        sentimentoClass = 'sentimento-positivo';
        emoji = '😍';
    } else if (sentimento.sentimento.includes('Positivo') && !sentimento.sentimento.includes('Muito')) {
        sentimentoClass = 'sentimento-positivo';
        emoji = '😊';
    } else if (sentimento.sentimento.includes('Muito Negativo') || sentimento.sentimento.includes('1 star')) {
        sentimentoClass = 'sentimento-negativo';
        emoji = '😤';
    } else if (sentimento.sentimento.includes('Negativo') && !sentimento.sentimento.includes('Muito')) {
        sentimentoClass = 'sentimento-negativo';
        emoji = '😞';
    }

    const html = `
        <div class="sentimento-box">
            <div style="font-size: 3rem; margin-bottom: 10px;">${emoji}</div>
            <div class="sentimento-label ${sentimentoClass}">${sentimento.sentimento}</div>
            <div class="sentimento-score">Confiança: ${(sentimento.confianca * 100).toFixed(1)}%</div>
        </div>
    `;

    content.innerHTML = html;
}

// Display Análise Completa
function displayAnalisaCompleta(resultado) {
    resultsSection.classList.remove('hidden');
    document.getElementById('analisaCompleta').classList.remove('hidden');
    document.getElementById('textosSimilares').classList.add('hidden');
    document.getElementById('classificacaoBox').classList.add('hidden');
    document.getElementById('sentimentoBox').classList.add('hidden');

    // Classificação
    const classContent = document.getElementById('classificacaoResult');
    const classData = resultado.classificacao;
    classContent.innerHTML = `
        <div style="padding: 15px; background: var(--bg-color); border-radius: 8px;">
            <strong style="font-size: 1.1rem; color: var(--success-color);">${classData.categoria}</strong>
            <div style="margin-top: 8px; color: var(--text-light);">
                Confiança: ${(classData.confianca * 100).toFixed(1)}%
            </div>
        </div>
    `;

    // Sentimento
    const sentContent = document.getElementById('sentimentoResult');
    const sentData = resultado.sentimento;
    let emoji = '😐';
    if (sentData.sentimento.includes('Muito Positivo') || sentData.sentimento.includes('5 stars')) {
        emoji = '😍';
    } else if (sentData.sentimento.includes('Positivo') && !sentData.sentimento.includes('Muito')) {
        emoji = '😊';
    } else if (sentData.sentimento.includes('Muito Negativo') || sentData.sentimento.includes('1 star')) {
        emoji = '😤';
    } else if (sentData.sentimento.includes('Negativo') && !sentData.sentimento.includes('Muito')) {
        emoji = '😞';
    }

    sentContent.innerHTML = `
        <div style="text-align: center; padding: 15px;">
            <div style="font-size: 2rem; margin-bottom: 10px;">${emoji}</div>
            <div style="font-weight: 600; color: var(--primary-color);">${sentData.sentimento}</div>
            <div style="color: var(--text-light); margin-top: 8px;">
                Confiança: ${(sentData.confianca * 100).toFixed(1)}%
            </div>
        </div>
    `;

    // Similares
    const simContent = document.getElementById('similaresResult');
    simContent.innerHTML = resultado.textos_similares.map((item, idx) => `
        <div class="similar-item">
            <div class="similar-texto"><strong>#${idx + 1}:</strong> "${item.texto}"</div>
            <div class="similar-meta">
                <span><strong>Categoria:</strong> ${item.categoria}</span>
                <span><strong>Similaridade:</strong> ${(item.similitude * 100).toFixed(1)}%</span>
                <span class="badge badge-${item.idioma}">${item.idioma.toUpperCase()}</span>
            </div>
        </div>
    `).join('');
}

// Limpar
function limpar() {
    promptInput.value = '';
    topKInput.value = 5;
    topKValue.textContent = 5;
    resultsSection.classList.add('hidden');
    error.classList.add('hidden');
    promptInput.focus();
}

// Keyboard shortcut (Ctrl+Enter para análise completa)
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        analisarCompleta();
    }
});
