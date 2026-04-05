import './styles.css';

const app = document.querySelector('#app');

app.innerHTML = `
  <main class="page-shell">
    <section class="hero">
      <p class="eyebrow">Machine Learning Project Foundation</p>
      <h1>Paraphrase Detection System</h1>
      <p class="hero-copy">
        Compare one reference document against multiple text files and estimate whether another document
        is likely paraphrased using both TF-IDF overlap and embedding-based meaning similarity.
      </p>
    </section>

    <section class="panel upload-panel">
      <div class="panel-header">
        <h2>Upload Documents</h2>
        <p>Select one original document and one or more test documents in .txt format.</p>
      </div>

      <form id="analysis-form" class="upload-form">
        <label class="field">
          <span>Original Document</span>
          <input id="query-file" name="query_file" type="file" accept=".txt" required />
        </label>

        <label class="field">
          <span>Test Document</span>
          <input id="documents" name="documents" type="file" accept=".txt" multiple required />
        </label>

        <button id="submit-button" class="primary-button" type="submit">Analyze Documents</button>
      </form>

      <p id="status-message" class="status-message">Ready for analysis.</p>
    </section>

    <section id="results-section" class="results-grid hidden">
      <article class="panel summary-panel">
        <div class="panel-header">
          <h2>Analysis Summary</h2>
          <p>Overview of the current run.</p>
        </div>
        <div id="summary-content" class="summary-content"></div>
      </article>

      <article class="panel results-panel">
        <div class="panel-header">
          <h2>Ranked Results</h2>
          <p>All uploaded test documents are summarized here and ordered by similarity strength.</p>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Document</th>
                <th>Summary</th>
                <th>Paraphrase Gauge</th>
                <th>TF-IDF Score</th>
                <th>Embedding Score</th>
                <th>Verdict</th>
              </tr>
            </thead>
            <tbody id="results-body"></tbody>
          </table>
        </div>

      </article>

      <article class="panel filter-panel">
        <div class="panel-header">
          <h2>Filter Result</h2>
          <p>Choose which test document you want to inspect in detail.</p>
        </div>
        <div class="filter-row">
          <label class="field filter-field">
            <span>Test Document</span>
            <select id="document-filter"></select>
          </label>
        </div>
      </article>

      <article class="panel detector-panel">
        <div class="panel-header">
          <h2>Paraphrase Gauge</h2>
          <p>The selected test document is summarized here as a paraphrase likelihood.</p>
        </div>
        <div id="detector-content" class="detector-content"></div>
      </article>

      <article class="panel explanation-panel">
        <div class="panel-header explanation-header">
          <h2>Why It Looks Like a Paraphrase</h2>
          <p>The evidence below explains the currently selected test document.</p>
        </div>
        <div id="explanation-content" class="explanation-content"></div>
      </article>

      <article class="panel recommendation-panel">
        <div class="panel-header">
          <h2>Recommendation</h2>
          <p>Interpreted output beyond raw scores.</p>
        </div>
        <p id="recommendation-text" class="recommendation-text"></p>
      </article>
    </section>
  </main>
`;

const form = document.querySelector('#analysis-form');
const queryFileInput = document.querySelector('#query-file');
const documentsInput = document.querySelector('#documents');
const statusMessage = document.querySelector('#status-message');
const resultsSection = document.querySelector('#results-section');
const summaryContent = document.querySelector('#summary-content');
const recommendationText = document.querySelector('#recommendation-text');
const detectorContent = document.querySelector('#detector-content');
const explanationContent = document.querySelector('#explanation-content');
const documentFilter = document.querySelector('#document-filter');
const resultsBody = document.querySelector('#results-body');
const submitButton = document.querySelector('#submit-button');
let currentResults = [];

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const queryFile = queryFileInput.files[0];
  const documentFiles = Array.from(documentsInput.files);

  if (!queryFile) {
    statusMessage.textContent = 'Please choose a query .txt file.';
    return;
  }

  if (documentFiles.length === 0) {
    statusMessage.textContent = 'Please choose at least one comparison .txt file.';
    return;
  }

  const formData = new FormData();
  formData.append('query_file', queryFile);
  documentFiles.forEach((file) => formData.append('documents', file));

  submitButton.disabled = true;
  statusMessage.textContent = 'Running analysis...';

  try {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      body: formData,
    });
    const rawText = await response.text();
    let data = null;

    if (rawText) {
      try {
        data = JSON.parse(rawText);
      } catch {
        data = null;
      }
    }

    if (!response.ok) {
      const fallbackMessage = rawText
        ? `Analysis failed with status ${response.status}: ${rawText.slice(0, 180)}`
        : `Analysis failed with status ${response.status}.`;
      throw new Error(data?.error || fallbackMessage);
    }

    if (!data) {
      throw new Error(
        rawText
          ? `The backend returned a non-JSON response: ${rawText.slice(0, 180)}`
          : 'The backend returned an empty response.'
      );
    }

    renderResults(data);
    statusMessage.textContent = 'Analysis completed successfully.';
  } catch (error) {
    resultsSection.classList.add('hidden');
    statusMessage.textContent =
      error instanceof TypeError
        ? 'Could not connect to the backend. Make sure "python api.py" is running.'
        : error.message;
  } finally {
    submitButton.disabled = false;
  }
});

function renderResults(data) {
  resultsSection.classList.remove('hidden');
  currentResults = Array.isArray(data.ranked_documents) ? data.ranked_documents : [];

  summaryContent.innerHTML = `
    <div class="summary-card">
      <span class="summary-label">Original Document</span>
      <strong>${escapeHtml(getText(data.query_name, 'Unknown query'))}</strong>
    </div>
    <div class="summary-card">
      <span class="summary-label">Compared Documents</span>
      <strong>${getNumber(data.document_count, 0)}</strong>
    </div>
    <div class="summary-card">
      <span class="summary-label">Found Similar</span>
      <strong>${getNumber(data.similar_count, 0)}</strong>
    </div>
    <div class="summary-card">
      <span class="summary-label">Found Paraphrased</span>
      <strong>${getNumber(data.paraphrased_count, 0)}</strong>
    </div>
  `;

  resultsBody.innerHTML = currentResults
    .map((item, index) => {
      return `
        <tr>
          <td>${index + 1}</td>
          <td>${escapeHtml(getText(item.document_name, 'Unknown document'))}</td>
          <td>${escapeHtml(getText(item.document_summary, 'No summary available.'))}</td>
          <td>
            <strong>${formatPercent(item.paraphrase_score)}</strong>
            <div class="metric-note">${escapeHtml(getText(item.relationship_summary, 'No relationship summary'))}</div>
          </td>
          <td>
            <strong>${formatScore(item.tfidf_cosine_similarity)}</strong>
            <div class="metric-note">${escapeHtml(getText(item.tfidf_relevance_level, 'Unavailable'))}</div>
          </td>
          <td>
            <strong>${formatScore(item.semantic_cosine_similarity)}</strong>
            <div class="metric-note">${escapeHtml(getText(item.semantic_relevance_level, 'Unavailable'))}</div>
          </td>
          <td>${escapeHtml(getText(item.paraphrase_label, 'Unavailable'))}</td>
        </tr>
      `;
    })
    .join('');

  renderFilter(currentResults);
  updateDetailView(currentResults[0], data.interpretation);
}

function renderFilter(results) {
  documentFilter.innerHTML = results
    .map(
      (item, index) =>
        `<option value="${index}">${escapeHtml(getText(item.document_name, `Document ${index + 1}`))}</option>`
    )
    .join('');

  documentFilter.onchange = () => {
    const selected = results[Number(documentFilter.value)] || results[0];
    updateDetailView(selected);
  };
}

function updateDetailView(item, defaultRecommendation = null) {
  detectorContent.innerHTML = item
    ? renderGauge(item)
    : '<p class="empty-state">No comparison documents were returned.</p>';

  explanationContent.innerHTML = item
    ? renderExplanation(item)
    : '<p class="empty-state">No explanation is available.</p>';

  recommendationText.textContent = item
    ? buildRecommendation(item)
    : getText(defaultRecommendation, 'No recommendation text was returned by the backend.');
}

function renderGauge(item) {
  const score = clampNumber(item.paraphrase_score, 0, 100);
  const label = getText(item.paraphrase_label, 'Unavailable');
  const relationship = getText(item.relationship_summary, 'No relationship summary was returned.');
  const documentName = getText(item.document_name, 'Unknown document');
  const gaugeColorClass = score >= 78 ? 'is-high' : score >= 58 ? 'is-medium' : 'is-low';

  return `
    <div class="detector-layout">
      <div class="gauge-card">
        <div class="gauge-shell ${gaugeColorClass}" style="--gauge-score: ${score};">
          <div class="gauge-inner">
            <div class="gauge-value">${Math.round(score)}%</div>
            <div class="gauge-label">${escapeHtml(label)}</div>
          </div>
        </div>
      </div>
      <div class="detector-copy">
        <p class="detector-kicker">Top matched document</p>
        <h3>${escapeHtml(documentName)}</h3>
        <p class="detector-summary">${escapeHtml(relationship)}</p>
      </div>
    </div>
  `;
}

function renderExplanation(item) {
  const terms = Array.isArray(item.top_terms) && item.top_terms.length > 0
    ? item.top_terms.join(', ')
    : 'No strong shared terms';
  const explanation = Array.isArray(item.paraphrase_explanation) ? item.paraphrase_explanation : [];

  return `
    <div class="explanation-grid">
      <div class="evidence-card">
        <span class="summary-label">Shared Terms</span>
        <strong>${escapeHtml(terms)}</strong>
      </div>
      ${explanation
        .map(
          (line) => `
            <div class="evidence-card">
              <span class="summary-label">Evidence</span>
              <p>${escapeHtml(getText(line, 'Unavailable evidence.'))}</p>
            </div>
          `
        )
        .join('')}
    </div>
  `;
}

function buildRecommendation(item) {
  return `${getText(item.document_name, 'This document')} has a paraphrase gauge of ${formatPercent(
    item.paraphrase_score
  )}. ${getText(item.paraphrase_label, 'Unavailable')}. ${getText(
    item.relationship_summary,
    'No relationship summary was returned.'
  )}`;
}

function formatScore(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(4) : 'N/A';
}

function formatPercent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${Math.round(number)}%` : 'N/A';
}

function getText(value, fallback) {
  return typeof value === 'string' && value.trim() ? value : fallback;
}

function getNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function clampNumber(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, getNumber(value, minimum)));
}

function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}
