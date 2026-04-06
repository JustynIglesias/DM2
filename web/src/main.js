import './styles.css';

let inputMode = 'upload';

function renderApp() {
const app = document.querySelector('#app');
if (!app) {
  return;
}

app.innerHTML = `
  <main class="page-shell">
    <section class="hero">
      <p class="eyebrow">Machine Learning Project Foundation</p>
      <h1>Classmate Work Comparison & Feedback System</h1>
      <p class="hero-copy">
        Compare one original classwork submission against one or more classmate submissions, review similarity
        patterns, and generate helpful feedback using TF-IDF and bag-of-words cosine similarity.
      </p>
    </section>

    <section class="panel upload-panel">
      <div class="panel-header">
        <h2>Input Options</h2>
        <p>Choose whether you want to upload ".txt" files or type the original and test text directly.</p>
      </div>

      <div class="input-tabs" role="tablist" aria-label="Input mode">
        <button id="upload-tab" class="tab-button is-active" type="button" data-mode="upload">Upload Documents</button>
        <button id="type-tab" class="tab-button" type="button" data-mode="type">Type Text</button>
        <button id="professor-tab" class="tab-button" type="button" data-mode="professor">Professor Batch</button>
      </div>

      <form id="analysis-form">
        <div id="upload-panel" class="input-panel is-active">
          <div class="upload-form">
            <label class="field">
              <span>Original Document</span>
              <input id="query-file" name="query_file" type="file" accept=".txt,.pdf,.docx" />
            </label>

            <label class="field">
              <span>Test Document</span>
              <input id="documents" name="documents" type="file" accept=".txt,.pdf,.docx" multiple />
            </label>
          </div>
        </div>

        <div id="type-panel" class="input-panel">
          <div class="typed-form">
            <label class="field">
              <span>Original Name</span>
              <input id="query-name" name="query_name" type="text" placeholder="typed_original.txt" />
            </label>

            <label class="field">
              <span>Test Name</span>
              <input id="test-name" name="test_name" type="text" placeholder="typed_test.txt" />
            </label>

            <label class="field field-wide">
              <span>Original Text</span>
              <textarea id="query-text" name="query_text" rows="8" placeholder="Paste or type the original document text here."></textarea>
            </label>

            <label class="field field-wide">
              <span>Test Text</span>
              <textarea id="test-text" name="test_text" rows="8" placeholder="Paste or type the test document text here."></textarea>
            </label>
          </div>
        </div>

        <div id="professor-panel" class="input-panel">
          <div class="professor-form">
            <label class="field field-wide">
              <span>Student Submissions</span>
              <input id="batch-documents" name="batch_documents" type="file" accept=".txt,.pdf,.docx" multiple />
            </label>
          </div>

          <div class="batch-preview">
            <div class="batch-preview-header">
              <div>
                <h3>Submission Set</h3>
                <p>Add all student .txt, .pdf, or .docx files here first. We can build the professor review flow on top of this set next.</p>
              </div>
              <div class="batch-count" id="batch-count">0 files loaded</div>
            </div>
            <div id="batch-list" class="batch-list">
              <p class="empty-state">No student submissions loaded yet.</p>
            </div>
            <div class="professor-actions">
              <button id="professor-submit-button" class="primary-button" type="submit">Test the documents</button>
            </div>
          </div>

          <div class="professor-results hidden" id="professor-results">
            <div class="professor-result-tabs" role="tablist" aria-label="Professor results">
              <button id="submission-overview-tab" class="tab-button is-active" type="button">Submission Set</button>
              <button id="class-ranking-tab" class="tab-button" type="button">Class Ranking</button>
              <button id="similarity-count-tab" class="tab-button" type="button">Similarity Count</button>
              <button id="heat-map-tab" class="tab-button" type="button">Similarity Heat Map</button>
              <button id="influence-tab" class="tab-button" type="button">Influence Detection</button>
            </div>

            <div id="submission-overview-panel" class="professor-result-panel is-active">
              <div class="panel-header">
                <h2>Submission Overview</h2>
                <p id="professor-overview-text">The loaded student files will appear here.</p>
              </div>
            </div>

            <div id="class-ranking-panel" class="professor-result-panel">
              <div class="panel-header">
                <h2>Class Ranking by Writing Quality and TF-IDF Uniqueness</h2>
                <p>Ranked view of class submissions based on writing quality and originality of vocabulary use.</p>
              </div>
              <div class="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Document</th>
                      <th>Writing Quality</th>
                      <th>TF-IDF Uniqueness</th>
                      <th class="score-header">TF-IDF Score <span>(cosine similarity)</span></th>
                      <th class="score-header">Bag-of-Words Score <span>(cosine similarity)</span></th>
                      <th>Word Count</th>
                      <th>Feedback</th>
                    </tr>
                  </thead>
                  <tbody id="professor-ranking-body"></tbody>
                </table>
              </div>
            </div>

            <div id="similarity-count-panel" class="professor-result-panel">
              <div class="panel-header">
                <h2>Similarity Count</h2>
                <p id="similarity-count-summary-text">Each student's similarity count will appear here.</p>
              </div>
              <div class="filter-row">
                <label class="field filter-field">
                  <span>Filter</span>
                  <select id="similarity-count-filter">
                    <option value="all">All</option>
                    <option value="tfidf">Only TF-IDF</option>
                    <option value="bow">Only BOW</option>
                  </select>
                </label>
                <label class="field filter-field">
                  <span>Rule Threshold</span>
                  <input id="similarity-count-threshold" type="range" min="0" max="1" step="0.01" value="0.50" />
                  <strong id="similarity-count-threshold-value">0.50</strong>
                </label>
              </div>
              <div id="similarity-count-highlights" class="heat-map-highlights"></div>
              <section class="heat-map-card">
                <h3>Classmate Matches</h3>
                <div id="similarity-count-list" class="similarity-count-list"></div>
              </section>
            </div>

            <div id="heat-map-panel" class="professor-result-panel">
              <div class="panel-header">
                <h2>Similarity Heat Map</h2>
                <p id="heat-map-summary-text">Student-to-student similarity scores will appear here.</p>
              </div>
              <section class="heat-map-card heat-map-matrix-card">
                <h3>All Students vs All Students</h3>
                <div class="table-wrap">
                  <table class="heat-matrix-table">
                    <thead id="heat-matrix-head"></thead>
                    <tbody id="heat-matrix-body"></tbody>
                  </table>
                </div>
              </section>
            </div>

            <div id="influence-panel" class="professor-result-panel">
              <div class="panel-header">
                <h2>Influence Detection</h2>
                <p id="influence-summary-text">Possible direction of influence and shared-source patterns will appear here.</p>
              </div>
              <div class="filter-row">
                <label class="field filter-field">
                  <span>Influence Score Filter</span>
                  <select id="influence-score-filter">
                    <option value="all">All Scores</option>
                    <option value="0-20">0-20%</option>
                    <option value="21-39">21-39%</option>
                    <option value="40-59">40-59%</option>
                    <option value="60-79">60-79%</option>
                    <option value="80-100">80-100%</option>
                  </select>
                </label>
              </div>
              <div class="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>1st Document</th>
                      <th>2nd Document</th>
                      <th>Flag</th>
                      <th>Decision</th>
                      <th>Influence Score</th>
                      <th class="score-header">TF-IDF Score <span>(cosine similarity)</span></th>
                      <th class="score-header">Bag-of-Words Score <span>(cosine similarity)</span></th>
                      <th>Explanation</th>
                    </tr>
                  </thead>
                  <tbody id="influence-body"></tbody>
                </table>
              </div>
              <div class="pagination-bar">
                <p id="influence-pagination-summary" class="pagination-summary">Showing 0 to 0 of 0 pairs</p>
                <div class="pagination-actions">
                  <button id="influence-prev-button" class="tab-button pagination-button" type="button">Previous</button>
                  <span id="influence-page-indicator" class="pagination-indicator">Page 1 of 1</span>
                  <button id="influence-next-button" class="tab-button pagination-button" type="button">Next</button>
                </div>
              </div>
            </div>
          </div>
        </div>

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
          <h2>Copycat Gauge</h2>
          <p>Compared documents are shown here with their copycat gauge and similarity scores.</p>
        </div>
        <div id="copycat-list" class="copycat-list"></div>
      </article>

      <article class="panel detector-panel">
        <div class="panel-header">
          <h2>Copycat Gauge</h2>
          <p>The selected test document is summarized here as a copycat likelihood.</p>
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
const uploadTab = document.querySelector('#upload-tab');
const typeTab = document.querySelector('#type-tab');
const professorTab = document.querySelector('#professor-tab');
const uploadPanel = document.querySelector('#upload-panel');
const typePanel = document.querySelector('#type-panel');
const professorPanel = document.querySelector('#professor-panel');
const queryFileInput = document.querySelector('#query-file');
const documentsInput = document.querySelector('#documents');
const queryNameInput = document.querySelector('#query-name');
const testNameInput = document.querySelector('#test-name');
const queryTextInput = document.querySelector('#query-text');
const testTextInput = document.querySelector('#test-text');
const batchDocumentsInput = document.querySelector('#batch-documents');
const batchList = document.querySelector('#batch-list');
const batchCount = document.querySelector('#batch-count');
const professorResults = document.querySelector('#professor-results');
const professorOverviewText = document.querySelector('#professor-overview-text');
const submissionOverviewTab = document.querySelector('#submission-overview-tab');
const classRankingTab = document.querySelector('#class-ranking-tab');
const similarityCountTab = document.querySelector('#similarity-count-tab');
const heatMapTab = document.querySelector('#heat-map-tab');
const influenceTab = document.querySelector('#influence-tab');
const submissionOverviewPanel = document.querySelector('#submission-overview-panel');
const classRankingPanel = document.querySelector('#class-ranking-panel');
const similarityCountPanel = document.querySelector('#similarity-count-panel');
const heatMapPanel = document.querySelector('#heat-map-panel');
const influencePanel = document.querySelector('#influence-panel');
const professorRankingBody = document.querySelector('#professor-ranking-body');
const similarityCountSummaryText = document.querySelector('#similarity-count-summary-text');
const similarityCountHighlights = document.querySelector('#similarity-count-highlights');
const similarityCountList = document.querySelector('#similarity-count-list');
const similarityCountFilter = document.querySelector('#similarity-count-filter');
const similarityCountThreshold = document.querySelector('#similarity-count-threshold');
const similarityCountThresholdValue = document.querySelector('#similarity-count-threshold-value');
const heatMapSummaryText = document.querySelector('#heat-map-summary-text');
const heatMatrixHead = document.querySelector('#heat-matrix-head');
const heatMatrixBody = document.querySelector('#heat-matrix-body');
const influenceSummaryText = document.querySelector('#influence-summary-text');
const influenceBody = document.querySelector('#influence-body');
const influenceScoreFilter = document.querySelector('#influence-score-filter');
const influencePaginationSummary = document.querySelector('#influence-pagination-summary');
const influencePageIndicator = document.querySelector('#influence-page-indicator');
const influencePrevButton = document.querySelector('#influence-prev-button');
const influenceNextButton = document.querySelector('#influence-next-button');
const statusMessage = document.querySelector('#status-message');
const resultsSection = document.querySelector('#results-section');
const summaryContent = document.querySelector('#summary-content');
const recommendationText = document.querySelector('#recommendation-text');
const detectorContent = document.querySelector('#detector-content');
const explanationContent = document.querySelector('#explanation-content');
const copycatList = document.querySelector('#copycat-list');
const submitButton = document.querySelector('#submit-button');
const professorSubmitButton = document.querySelector('#professor-submit-button');
let currentResults = [];
let currentSimilarityHeatMap = null;
let currentInfluenceFindings = [];
let influencePage = 1;
const INFLUENCE_PAGE_SIZE = 15;

uploadTab.addEventListener('click', () => setInputMode('upload'));
typeTab.addEventListener('click', () => setInputMode('type'));
professorTab.addEventListener('click', () => setInputMode('professor'));
batchDocumentsInput.addEventListener('change', renderBatchPreview);
submissionOverviewTab.addEventListener('click', () => setProfessorResultTab('overview'));
classRankingTab.addEventListener('click', () => setProfessorResultTab('ranking'));
similarityCountTab.addEventListener('click', () => setProfessorResultTab('similarity-count'));
similarityCountFilter.addEventListener('change', () => renderSimilarityCount(currentSimilarityHeatMap));
similarityCountThreshold.addEventListener('input', () => {
  similarityCountThresholdValue.textContent = Number(similarityCountThreshold.value).toFixed(2);
  renderSimilarityCount(currentSimilarityHeatMap);
});
heatMapTab.addEventListener('click', () => setProfessorResultTab('heat-map'));
influenceTab.addEventListener('click', () => setProfessorResultTab('influence'));
influencePrevButton.addEventListener('click', () => changeInfluencePage(-1));
influenceNextButton.addEventListener('click', () => changeInfluencePage(1));
influenceScoreFilter.addEventListener('change', () => {
  influencePage = 1;
  renderInfluenceFindings();
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  if (inputMode === 'professor') {
    const batchFiles = Array.from(batchDocumentsInput.files);
    if (batchFiles.length === 0) {
      statusMessage.textContent = 'Please add at least one student submission.';
      return;
    }

    const batchFormData = new FormData();
    batchFiles.forEach((file) => batchFormData.append('batch_documents', file));

    submitButton.disabled = true;
    professorSubmitButton.disabled = true;
    statusMessage.textContent = 'Preparing professor ranking...';

    try {
      const response = await fetch('/api/professor/ranking', {
        method: 'POST',
        body: batchFormData,
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
          ? `Professor ranking failed with status ${response.status}: ${rawText.slice(0, 180)}`
          : `Professor ranking failed with status ${response.status}.`;
        throw new Error(data?.error || fallbackMessage);
      }

      if (!data) {
        throw new Error(
          rawText
            ? `The backend returned a non-JSON response: ${rawText.slice(0, 180)}`
            : 'The backend returned an empty response.'
        );
      }

      renderBatchPreview();
      renderProfessorResults(data);
      resultsSection.classList.add('hidden');
      statusMessage.textContent = 'Professor ranking prepared successfully.';
    } catch (error) {
      professorResults.classList.add('hidden');
      statusMessage.textContent =
        error instanceof TypeError
          ? 'Could not connect to the backend. Make sure "python api.py" is running.'
          : error.message;
    } finally {
      submitButton.disabled = false;
      professorSubmitButton.disabled = false;
    }

    return;
  }

  const formData = new FormData();

  if (inputMode === 'upload') {
    const queryFile = queryFileInput.files[0];
    const documentFiles = Array.from(documentsInput.files);

    if (!queryFile) {
      statusMessage.textContent = 'Please choose an original .txt, .pdf, or .docx file.';
      return;
    }

    if (documentFiles.length === 0) {
      statusMessage.textContent = 'Please choose at least one test .txt, .pdf, or .docx file.';
      return;
    }

    formData.append('query_file', queryFile);
    documentFiles.forEach((file) => formData.append('documents', file));
  } else {
    const queryText = queryTextInput.value.trim();
    const testText = testTextInput.value.trim();

    if (!queryText) {
      statusMessage.textContent = 'Please type the original text.';
      return;
    }

    if (!testText) {
      statusMessage.textContent = 'Please type the test text.';
      return;
    }

    formData.append('query_text', queryText);
    formData.append('test_text', testText);
    formData.append('query_name', queryNameInput.value.trim());
    formData.append('test_name', testNameInput.value.trim());
  }

  submitButton.disabled = true;
  professorSubmitButton.disabled = true;
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
    professorSubmitButton.disabled = false;
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

  copycatList.innerHTML = currentResults
    .map((item, index) => {
      return `
        <article class="copycat-item">
          <div class="copycat-item-head">
            <span class="summary-label">Rank ${index + 1}</span>
            <strong>${escapeHtml(getText(item.document_name, 'Unknown document'))}</strong>
          </div>
          <p class="copycat-item-summary">${escapeHtml(getText(item.document_summary, 'No summary available.'))}</p>
          <div class="copycat-item-metrics">
            <div class="summary-card">
              <span class="summary-label">Copycat Gauge</span>
              <strong>${formatPercent(item.influence_score)}</strong>
            </div>
            <div class="summary-card">
              <span class="summary-label">TF-IDF</span>
              <strong>${formatScore(item.tfidf_cosine_similarity)}</strong>
            </div>
            <div class="summary-card">
              <span class="summary-label">BOW</span>
              <strong>${formatScore(item.bow_cosine_similarity)}</strong>
            </div>
            <div class="summary-card">
              <span class="summary-label">Verdict</span>
              <strong>${escapeHtml(getText(item.paraphrase_label, 'Unavailable'))}</strong>
            </div>
          </div>
        </article>
      `;
    })
    .join('');

  updateDetailView(currentResults[0], data.interpretation);
}

function renderProfessorResults(data) {
  const rankedSubmissions = Array.isArray(data.ranked_submissions) ? data.ranked_submissions : [];
  const influenceFindings = Array.isArray(data.influence_findings) ? data.influence_findings : [];
  const similarityHeatMap = data.similarity_heat_map && typeof data.similarity_heat_map === 'object'
    ? data.similarity_heat_map
    : null;
  professorResults.classList.remove('hidden');
  professorOverviewText.textContent = `${getNumber(data.submission_count, 0)} submission(s) loaded for ${getText(
    data.batch_name,
    'this class activity'
  )}. ${getText(data.interpretation, 'Class ranking is ready.')}`;

  professorRankingBody.innerHTML = rankedSubmissions
    .map(
      (item, index) => `
        <tr>
          <td>${index + 1}</td>
          <td>${escapeHtml(getText(item.document_name, 'Unknown document'))}</td>
          <td>
            <strong>${formatPercent(item.writing_quality_score)}</strong>
            <div class="metric-note">${escapeHtml(getText(item.writing_feedback, 'No feedback available.'))}</div>
          </td>
          <td>
            <strong>${formatPercent(item.tfidf_uniqueness_score)}</strong>
            <div class="metric-note">${getNumber(item.unique_term_count, 0)} unique weighted terms</div>
          </td>
          <td>
            <strong>${formatScore(item.tfidf_score)}</strong>
            <div class="metric-note">${escapeHtml(getText(item.tfidf_score_level, 'Unavailable'))}</div>
          </td>
          <td>
            <strong>${formatScore(item.bow_score)}</strong>
            <div class="metric-note">${escapeHtml(getText(item.bow_score_level, 'Unavailable'))}</div>
          </td>
          <td>
            <strong>${getNumber(item.word_count, 0)}</strong>
            <div class="metric-note">${getNumber(item.sentence_count, 0)} sentences</div>
          </td>
          <td>${escapeHtml(getText(item.writing_feedback, 'No feedback available.'))}</td>
        </tr>
      `
    )
    .join('');

  influenceSummaryText.textContent = getText(
    data.influence_summary,
    'Possible direction of influence and shared-source patterns will appear here.'
  );
  currentInfluenceFindings = influenceFindings;
  influencePage = 1;
  renderInfluenceFindings();

  renderSimilarityHeatMap(similarityHeatMap);
  renderSimilarityCount(similarityHeatMap);
  setProfessorResultTab('ranking');
}

function changeInfluencePage(step) {
  const totalPages = Math.max(1, Math.ceil(currentInfluenceFindings.length / INFLUENCE_PAGE_SIZE));
  const nextPage = Math.min(totalPages, Math.max(1, influencePage + step));

  if (nextPage === influencePage) {
    return;
  }

  influencePage = nextPage;
  renderInfluenceFindings();
}

function renderInfluenceFindings() {
  const filteredFindings = currentInfluenceFindings.filter(matchesInfluenceScoreFilter);
  const totalItems = filteredFindings.length;
  const totalPages = Math.max(1, Math.ceil(totalItems / INFLUENCE_PAGE_SIZE));
  const safePage = Math.min(totalPages, Math.max(1, influencePage));
  const startIndex = (safePage - 1) * INFLUENCE_PAGE_SIZE;
  const pagedItems = filteredFindings.slice(startIndex, startIndex + INFLUENCE_PAGE_SIZE);

  influencePage = safePage;

  influenceBody.innerHTML = pagedItems.length
    ? pagedItems
        .map(
          (item, index) => `
            <tr>
              <td>${startIndex + index + 1}</td>
              <td>
                <strong>${escapeHtml(getText(item.source_document, 'Unknown document'))}</strong>
                <div class="metric-note">Loaded earlier as file ${getNumber(item.source_position, 0)}</div>
              </td>
              <td>
                <strong>${escapeHtml(getText(item.target_document, 'Unknown document'))}</strong>
                <div class="metric-note">Loaded later as file ${getNumber(item.target_position, 0)}</div>
              </td>
              <td>${escapeHtml(getText(item.influence_type, 'Unavailable'))}</td>
              <td>${escapeHtml(getText(item.decision, 'No decision available.'))}</td>
              <td>
                <strong>${formatPercent(item.influence_score)}</strong>
                <div class="metric-note">Combined pairwise overlap signal</div>
              </td>
              <td>
                <strong>${formatScore(item.tfidf_cosine_similarity)}</strong>
                <div class="metric-note">${escapeHtml(getText(item.tfidf_relevance_level, 'Weighted wording overlap'))}</div>
              </td>
              <td>
                <strong>${formatScore(item.bow_cosine_similarity)}</strong>
                <div class="metric-note">${escapeHtml(getText(item.bow_relevance_level, 'Direct wording overlap'))}</div>
              </td>
              <td>${escapeHtml(getText(item.explanation, 'No explanation available.'))}</td>
            </tr>
          `
        )
        .join('')
    : `
      <tr>
        <td colspan="9">
          <p class="empty-state">No influence relationships matched the selected score range.</p>
        </td>
      </tr>
    `;

  const visibleStart = totalItems === 0 ? 0 : startIndex + 1;
  const visibleEnd = totalItems === 0 ? 0 : Math.min(startIndex + pagedItems.length, totalItems);
  influencePaginationSummary.textContent = `Showing ${visibleStart} to ${visibleEnd} of ${totalItems} pairs`;
  influencePageIndicator.textContent = `Page ${safePage} of ${totalPages}`;
  influencePrevButton.disabled = safePage <= 1;
  influenceNextButton.disabled = safePage >= totalPages;
}

function matchesInfluenceScoreFilter(item) {
  const filterValue = influenceScoreFilter.value;
  const score = Number(item?.influence_score ?? 0);

  if (filterValue === 'all') {
    return true;
  }
  if (filterValue === '0-20') {
    return score >= 0 && score <= 20;
  }
  if (filterValue === '21-39') {
    return score >= 21 && score <= 39;
  }
  if (filterValue === '40-59') {
    return score >= 40 && score <= 59;
  }
  if (filterValue === '60-79') {
    return score >= 60 && score <= 79;
  }
  if (filterValue === '80-100') {
    return score >= 80 && score <= 100;
  }

  return true;
}

function setInputMode(mode) {
  inputMode = mode;
  const isUpload = mode === 'upload';
  const isType = mode === 'type';
  const isProfessor = mode === 'professor';

  uploadTab.classList.toggle('is-active', isUpload);
  typeTab.classList.toggle('is-active', isType);
  professorTab.classList.toggle('is-active', isProfessor);
  uploadPanel.classList.toggle('is-active', isUpload);
  typePanel.classList.toggle('is-active', isType);
  professorPanel.classList.toggle('is-active', isProfessor);
  submitButton.classList.toggle('hidden', isProfessor);
  professorSubmitButton.classList.toggle('hidden', !isProfessor);
  submitButton.textContent = 'Analyze Documents';
}

function setProfessorResultTab(tab) {
  const showOverview = tab === 'overview';
  const showRanking = tab === 'ranking';
  const showSimilarityCount = tab === 'similarity-count';
  const showHeatMap = tab === 'heat-map';
  const showInfluence = tab === 'influence';
  submissionOverviewTab.classList.toggle('is-active', showOverview);
  classRankingTab.classList.toggle('is-active', showRanking);
  similarityCountTab.classList.toggle('is-active', showSimilarityCount);
  heatMapTab.classList.toggle('is-active', showHeatMap);
  influenceTab.classList.toggle('is-active', showInfluence);
  submissionOverviewPanel.classList.toggle('is-active', showOverview);
  classRankingPanel.classList.toggle('is-active', showRanking);
  similarityCountPanel.classList.toggle('is-active', showSimilarityCount);
  heatMapPanel.classList.toggle('is-active', showHeatMap);
  influencePanel.classList.toggle('is-active', showInfluence);
}

function renderBatchPreview() {
  const files = Array.from(batchDocumentsInput.files || []);
  batchCount.textContent = `${files.length} file${files.length === 1 ? '' : 's'} loaded`;

  if (files.length === 0) {
    batchList.innerHTML = '<p class="empty-state">No student submissions loaded yet.</p>';
    return;
  }

  batchList.innerHTML = files
    .map((file, index) => {
      const fileSizeKb = Math.max(1, Math.round(file.size / 1024));
      return `
        <article class="batch-item">
          <div class="batch-item-index">${index + 1}</div>
          <div class="batch-item-copy">
            <strong>${escapeHtml(file.name)}</strong>
            <p>${fileSizeKb} KB text submission</p>
          </div>
        </article>
      `;
    })
    .join('');
}

function renderSimilarityHeatMap(mapData) {
  if (!mapData) {
    heatMapSummaryText.textContent = 'Similarity heat map data is unavailable.';
    heatMatrixHead.innerHTML = '';
    heatMatrixBody.innerHTML = '';
    return;
  }

  const matrixRows = Array.isArray(mapData.matrix?.rows) ? mapData.matrix.rows : [];
  const matrixLabels = Array.isArray(mapData.matrix?.labels) ? mapData.matrix.labels : [];

  heatMapSummaryText.textContent = getText(
    mapData.summary,
    'Student-to-student similarity scores will appear here.'
  );

  heatMatrixHead.innerHTML = `
    <tr>
      <th>Student</th>
      ${matrixLabels.map((label) => `<th>${escapeHtml(label)}</th>`).join('')}
    </tr>
  `;

  heatMatrixBody.innerHTML = matrixRows.length
    ? matrixRows
        .map((row) => {
          const cells = Array.isArray(row.similarities) ? row.similarities : [];
          return `
            <tr>
              <th>${escapeHtml(getText(row.document_name, 'Unknown document'))}</th>
              ${cells
                .map((cell) => {
                  const similarity = Number(cell.similarity_score);
                  return `
                    <td class="heat-matrix-cell ${getMatrixToneClass(similarity)}">
                      <strong>${formatScore(similarity)}</strong>
                      <div class="metric-note">${escapeHtml(getText(cell.similarity_level, 'Unavailable'))}</div>
                    </td>
                  `;
                })
                .join('')}
            </tr>
          `;
        })
        .join('')
    : `
      <tr>
        <td colspan="${Math.max(1, matrixLabels.length + 1)}">
          <p class="empty-state">Not enough similarity heat map data was returned to build the matrix.</p>
        </td>
      </tr>
    `;
}

function renderSimilarityCount(mapData) {
  currentSimilarityHeatMap = mapData;

  if (!mapData) {
    similarityCountSummaryText.textContent = 'Similarity count data is unavailable.';
    similarityCountHighlights.innerHTML = '<p class="empty-state">No classmate similarity counts were returned.</p>';
    similarityCountList.innerHTML = '<p class="empty-state">No student pairs available to count.</p>';
    return;
  }

  const labels = Array.isArray(mapData.matrix?.labels) ? mapData.matrix.labels : [];
  const matrixRows = Array.isArray(mapData.matrix?.rows) ? mapData.matrix.rows : [];
  const threshold = getNumber(similarityCountThreshold?.value, 0.5);
  const filterMode = similarityCountFilter?.value || 'all';
  const matchesByStudent = new Map(labels.map((label) => [label, []]));

  similarityCountThresholdValue.textContent = threshold.toFixed(2);

  matrixRows.forEach((row, rowIndex) => {
    const sourceName = getText(row.document_name, labels[rowIndex] || 'Unknown document');
    const cells = Array.isArray(row.similarities) ? row.similarities : [];

    cells.forEach((cell, cellIndex) => {
      if (cellIndex <= rowIndex) {
        return;
      }

      const tfidfSimilarity = getNumber(cell.tfidf_cosine_similarity, 0);
      const bowSimilarity = getNumber(cell.bow_cosine_similarity, 0);
      const matchesTfidf = tfidfSimilarity >= threshold;
      const matchesBow = bowSimilarity >= threshold;
      const isSimilar = getSimilarityMatchState(filterMode, matchesTfidf, matchesBow);

      if (!isSimilar) {
        return;
      }

      const targetName = getText(cell.target_document, labels[cellIndex] || 'Unknown document');
      const sourceMatches = matchesByStudent.get(sourceName) || [];
      const targetMatches = matchesByStudent.get(targetName) || [];

      sourceMatches.push({
        classmate: targetName,
        tfidfSimilarity,
        bowSimilarity,
      });
      targetMatches.push({
        classmate: sourceName,
        tfidfSimilarity,
        bowSimilarity,
      });

      matchesByStudent.set(sourceName, sourceMatches);
      matchesByStudent.set(targetName, targetMatches);
    });
  });

  const entries = Array.from(matchesByStudent.entries())
    .map(([name, matches]) => ({
      name,
      matches: matches.sort(
        (left, right) =>
          getSimilaritySortValue(filterMode, right) - getSimilaritySortValue(filterMode, left) ||
          left.classmate.localeCompare(right.classmate)
      ),
      count: matches.length,
    }))
    .sort((left, right) => right.count - left.count || left.name.localeCompare(right.name));
  const matchedStudents = entries.filter((item) => item.count > 0).length;
  const highestCount = entries.length ? entries[0].count : 0;
  const ruleLabel = getSimilarityRuleLabel(filterMode, threshold);

  similarityCountSummaryText.textContent =
    `${ruleLabel}. ` +
    `${matchedStudents} student(s) matched at least one classmate.`;

  similarityCountHighlights.innerHTML = `
    <article class="summary-card">
      <span class="summary-label">Students</span>
      <strong>${entries.length}</strong>
    </article>
    <article class="summary-card">
      <span class="summary-label">Matched Students</span>
      <strong>${matchedStudents}</strong>
    </article>
    <article class="summary-card">
      <span class="summary-label">Highest Count</span>
      <strong>${highestCount}</strong>
    </article>
    <article class="summary-card">
      <span class="summary-label">Filter</span>
      <strong>${escapeHtml(getSimilarityFilterLabel(filterMode))}</strong>
    </article>
  `;

  similarityCountList.innerHTML = entries.length
    ? entries
        .map(
          (item) => `
            <article class="similarity-count-item">
              <strong>${escapeHtml(item.name)}</strong>
              <p>${escapeHtml(buildSimilarityCountSentence(item.name, item.count))}</p>
              ${renderSimilarityMatchList(item.matches)}
            </article>
          `
        )
        .join('')
    : '<p class="empty-state">No student submissions were available for counting.</p>';
}

function buildSimilarityCountSentence(name, count) {
  const classmateLabel = count === 1 ? 'classmate' : 'classmates';
  return `${name} is similar to ${count} ${classmateLabel}.`;
}

function getSimilarityMatchState(filterMode, matchesTfidf, matchesBow) {
  if (filterMode === 'tfidf') {
    return matchesTfidf;
  }
  if (filterMode === 'bow') {
    return matchesBow;
  }
  return matchesTfidf || matchesBow;
}

function getSimilaritySortValue(filterMode, match) {
  if (filterMode === 'tfidf') {
    return getNumber(match.tfidfSimilarity, 0);
  }
  if (filterMode === 'bow') {
    return getNumber(match.bowSimilarity, 0);
  }
  return Math.max(getNumber(match.tfidfSimilarity, 0), getNumber(match.bowSimilarity, 0));
}

function getSimilarityFilterLabel(filterMode) {
  if (filterMode === 'tfidf') {
    return 'Only TF-IDF';
  }
  if (filterMode === 'bow') {
    return 'Only BOW';
  }
  return 'All';
}

function getSimilarityRuleLabel(filterMode, threshold) {
  const thresholdLabel = threshold.toFixed(2);
  if (filterMode === 'tfidf') {
    return `Students are counted as similar when TF-IDF >= ${thresholdLabel}`;
  }
  if (filterMode === 'bow') {
    return `Students are counted as similar when Bag-of-Words >= ${thresholdLabel}`;
  }
  return `Students are counted as similar when TF-IDF >= ${thresholdLabel} or Bag-of-Words >= ${thresholdLabel}`;
}

function renderSimilarityMatchList(matches) {
  if (!Array.isArray(matches) || matches.length === 0) {
    return '<ul class="similarity-match-list"><li>No classmates met the similarity rule.</li></ul>';
  }

  return `
    <details class="similarity-match-details">
      <summary>See more</summary>
      <ul class="similarity-match-list">
        ${matches
          .map(
            (match) => `
              <li>
                ${escapeHtml(match.classmate)}
                <span>TF-IDF: ${formatScore(match.tfidfSimilarity)} | BOW: ${formatScore(match.bowSimilarity)}</span>
              </li>
            `
          )
          .join('')}
      </ul>
    </details>
  `;
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
  const score = clampNumber(item.influence_score, 0, 100);
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
  return `${getText(item.document_name, 'This document')} has a copycat gauge of ${formatPercent(
    item.influence_score
  )}. ${getText(item.paraphrase_label, 'Unavailable')}. ${getText(
    item.relationship_summary,
    'No relationship summary was returned.'
  )}`;
}

function formatScore(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(3) : 'N/A';
}

function formatPercent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${Math.round(number)}%` : 'N/A';
}

function formatTermList(terms) {
  return Array.isArray(terms) && terms.length > 0 ? terms.join(', ') : 'No strong shared terms';
}

function getMatrixToneClass(score) {
  if (score >= 0.72) {
    return 'is-very-strong';
  }
  if (score >= 0.58) {
    return 'is-strong';
  }
  if (score >= 0.4) {
    return 'is-medium';
  }
  return 'is-low';
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
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', renderApp, { once: true });
} else {
  renderApp();
}
