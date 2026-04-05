import './styles.css';

const app = document.querySelector('#app');

app.innerHTML = `
  <main class="page-shell">
    <section class="hero">
      <p class="eyebrow">Machine Learning Project Foundation</p>
      <h1>Document Analysis System</h1>
      <p class="hero-copy">
        Analyze one reference document against multiple text files using TF-IDF, cosine similarity,
        ranked relevance, and generated recommendation output.
      </p>
    </section>

    <section class="panel upload-panel">
      <div class="panel-header">
        <h2>Upload Documents</h2>
        <p>Select one query document and multiple comparison documents in .txt format.</p>
      </div>

      <form id="analysis-form" class="upload-form">
        <label class="field">
          <span>Query Document</span>
          <input id="query-file" name="query_file" type="file" accept=".txt" required />
        </label>

        <label class="field">
          <span>Comparison Documents</span>
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

      <article class="panel recommendation-panel">
        <div class="panel-header">
          <h2>Recommendation</h2>
          <p>Interpreted output beyond raw scores.</p>
        </div>
        <p id="recommendation-text" class="recommendation-text"></p>
      </article>

      <article class="panel results-panel">
        <div class="panel-header">
          <h2>Ranked Results</h2>
          <p>Documents ordered from most relevant to least relevant.</p>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Rank</th>
                <th>Document</th>
                <th>Cosine Similarity</th>
                <th>Angle</th>
                <th>Interpretation</th>
                <th>Key Terms</th>
              </tr>
            </thead>
            <tbody id="results-body"></tbody>
          </table>
        </div>
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
const resultsBody = document.querySelector('#results-body');
const submitButton = document.querySelector('#submit-button');

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

  summaryContent.innerHTML = `
    <div class="summary-card">
      <span class="summary-label">Query Document</span>
      <strong>${escapeHtml(data.query_name)}</strong>
    </div>
    <div class="summary-card">
      <span class="summary-label">Compared Documents</span>
      <strong>${data.document_count}</strong>
    </div>
    <div class="summary-card">
      <span class="summary-label">Vocabulary Size</span>
      <strong>${data.vocabulary_size}</strong>
    </div>
  `;

  recommendationText.textContent = data.interpretation;

  resultsBody.innerHTML = data.ranked_documents
    .map((item, index) => {
      const terms = item.top_terms.length > 0 ? item.top_terms.join(', ') : 'No strong shared terms';

      return `
        <tr>
          <td>${index + 1}</td>
          <td>${escapeHtml(item.document_name)}</td>
          <td>${Number(item.cosine_similarity).toFixed(4)}</td>
          <td>${Number(item.angle_degrees).toFixed(2)} deg</td>
          <td>${escapeHtml(item.relevance_level)}</td>
          <td>${escapeHtml(terms)}</td>
        </tr>
      `;
    })
    .join('');
}

function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}
