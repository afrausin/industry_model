# Engineering Autonomous Financial Data Pipelines

## A Comprehensive Technical Framework for AI Agents

---

## Executive Summary

The integration of Large Language Models (LLMs) into the financial data supply chain represents a discontinuous leap in quantitative capability. We are transitioning from a regime of deterministic, script-based data extraction to one of semantic, intent-driven information retrieval. In this new paradigm, coding agents like Claude Code act not merely as automation tools, but as autonomous semantic routers capable of navigating the complex topologies of institutional data lakes. However, the efficacy of these agents is strictly bounded by the structural opacity and semantic ambiguity of traditional financial databases.

This report establishes a rigorous technical standard for "Agent-Ready" financial data architecture. By synthesizing analyses of proprietary ecosystems (Refinitiv/LSEG, S&P Global Market Intelligence, AlphaSense) and open-access protocols (Financial Modeling Prep, Google Trends), we define the requisite patterns for context management, symbology resolution, and point-in-time (PIT) integrity. Furthermore, we articulate a governance framework utilizing statistical validation libraries (Pandera, Great Expectations) and explicit instruction sets (.cursorrules) to prevent hallucinatory data retrieval. This document serves as a blueprint for quantitative developers architecting the next generation of self-describing, AI-navigable financial systems.

---

## 1. The Context Engineering Paradigm: Bridging the Semantic Gap

The primary obstacle preventing autonomous agents from effectively navigating financial databases is the "Context Bottleneck." Institutional financial APIs are characterized by high dimensionality, sparse matrices, and arcane mnemonics that resist zero-shot interpretation. A standard API documentation set for a platform like Refinitiv or Bloomberg can exceed tens of thousands of tokens, rendering it impossible to inject fully into an agent's working memory. Therefore, the foundational task of the modern quantitative engineer is Context Engineering—the systematic curation, compression, and structuring of metadata to maximize agentic reasoning.

### 1.1 The llms.txt Standard for Financial Libraries

To facilitate autonomous exploration, documentation must be decoupled from human-centric HTML presentation layers and restructured for machine consumption. The llms.txt specification has emerged as the de facto standard for this purpose, functioning as a "robots.txt" for reasoning engines rather than crawlers.

#### 1.1.1 Architectural Hierarchy and Schema

An effective llms.txt file for a financial data library must act as a curated index, guiding the agent to the specific documentation subsets required for a given task. Unlike a generic sitemap, the llms.txt schema prioritizes semantic density.

| Section | Purpose in Financial Context | Agent-Optimized Content Strategy |
|---------|------------------------------|----------------------------------|
| Project Identity | Establishing Domain Scope | Clearly define if the library handles Market Data (Tick/Quote), Fundamentals (10-K/10-Q), or Alternative Data. This prevents category errors (e.g., asking a pricing API for balance sheet items). |
| Core Documentation | Functional Primitives | Links to markdown files defining Authentication (Session management), Request Patterns (get_data vs get_history), and Error Handling. |
| Semantic Map | Field Definitions | Links to "Data Dictionaries" or "Field Catalogs" that map natural language terms ("Revenue") to vendor-specific mnemonics (TR.Revenue, IQ_TOTAL_REV). |
| Operational Guides | Implementation Patterns | Usage examples for handling rate limits, pagination, and asynchronous streaming. |

The file structure leverages Markdown's hierarchical headers to create logical partitions. For instance, separating "Real-Time Streaming" documentation from "Historical Fundamental" documentation allows the agent to selectively retrieve only the context relevant to the user's intent, minimizing context window pollution.

#### 1.1.2 The llms-full.txt Knowledge Base

While llms.txt serves as a routing layer, complex financial libraries often require deep context to use correctly. The llms-full.txt specification involves concatenating the full content of the referenced documentation into a single, massive text file. For complex SDKs like the Refinitiv Data Library, which utilizes intricate nested object responses and specific session initialization protocols, providing an llms-full.txt allows the agent to ingest the entire API surface area. This enables "multi-hop" reasoning—where the agent can verify if a specific parameter (e.g., SDate) is compatible with a specific function (e.g., get_history) without making external network calls that might hallucinate non-existent kwargs.

### 1.2 Dynamic Context via Model Context Protocol (MCP)

Static files (llms.txt) are necessary but insufficient for dynamic enterprise environments where database schemas evolve daily. The Model Context Protocol (MCP) provides a standardized mechanism to "serve" live documentation to agents.

In a sophisticated quantitative setup, an MCP server acts as a middleware between the agent and the organization's internal knowledge base (e.g., Confluence, Notion, or Git repositories). When an agent encounters an unknown dataset—for example, a new "Credit Card Transaction Panel"—it can query the MCP server for the schema definition, column statistics, and sampling methodology. This allows for a "Just-In-Time" (JIT) context loading strategy, where the agent retrieves schema definitions only when the specific analytical path requires them. This is particularly crucial for handling proprietary internal datasets where public training data is non-existent.

### 1.3 Self-Describing Database Architectures

The ultimate goal is to move beyond external documentation and embed context directly into the data storage layer. A Self-Describing Database utilizes vector embeddings to link table and column definitions to natural language concepts.

#### 1.3.1 Vectorized Schema Linking

By embedding schema descriptions (e.g., "Table FACT_SALES_TXN: Daily aggregated credit card spend by merchant category") into a vector store, an agent can perform semantic searches to identify relevant tables. When a user asks, "Analyze consumer spending trends in the hospitality sector," the system retrieves the FACT_SALES_TXN schema based on vector similarity, rather than relying on exact keyword matches.

#### 1.3.2 Few-Shot Prompting with Representative Samples

Documentation often fails to capture the messy reality of data. A column defined as VARCHAR(20) labeled status provides no information about the valid values. Research indicates that injecting 1-3 rows of sample data into the agent's prompt significantly reduces SQL generation errors. Seeing a sample row where status takes values `(Active, Inactive, Suspended)` rather than `['Active', 'Inactive']` allows the agent to generate correct WHERE clauses immediately, avoiding the trial-and-error loop.

---

## 2. Deep Dive: Proprietary Financial API Architectures

To build robust agents, we must dissect the specific architectural patterns of the dominant financial data vendors. Each platform enforces unique paradigms for authentication, data retrieval, and session management that an autonomous agent must navigate.

### 2.1 Refinitiv (LSEG) Data Platform: The Hybrid Abstraction

The Refinitiv Data Library for Python (formerly Eikon Data API) is a complex abstraction layer designed to unify access across desktop terminals (Eikon/Workspace) and cloud environments (Refinitiv Data Platform). However, this unification introduces ambiguity that often confuses AI agents.

#### 2.1.1 Session Management Hierarchy

The library uses a Session object to handle authentication and connection state. The agent must be capable of discerning the deployment environment to instantiate the correct session type:

- **DesktopSession:** Relies on a locally running Eikon or Workspace application acting as a proxy. This is common in analyst workflows but unsuitable for headless servers.
- **PlatformSession:** Connects directly to the cloud API using an App Key and User credentials. This is the standard for autonomous servers.

**Agent Design Pattern:** The agent should verify the existence of a local process or a configuration file (lseg-data.config.json) to determine the session strategy. Hardcoding credentials in scripts is a security violation; agents must be instructed to read from environment variables or secure vaults.

#### 2.1.2 The get_data vs. get_history Dichotomy

A prevalent failure mode for coding agents interacting with Refinitiv is the misuse of get_data for time-series analysis.

- **get_data:** Designed for snapshots—the most recent or real-time value of a field. While it accepts SDate (Start Date) and EDate parameters, using it for historical series often results in misaligned data frames where dates are returned as values rather than indices.
- **get_history:** The specialized function for historical time series. It enforces interval parameters (e.g., 'daily', '1h') and returns a properly indexed DataFrame.

**Critical Constraint:** The get_history endpoint has strict row limits (often 10,000 rows per request). An agent must implement pagination logic—breaking a long request (e.g., 10 years of hourly data) into smaller chunks and concatenating the results. Failure to do so results in a `UserWarning: Search result not full` and truncated data.

#### 2.1.3 Event-Driven Streaming

For real-time applications, Refinitiv uses an event-driven model via open_pricing_stream. Agents accustomed to synchronous REST calls often struggle here. The agent must define callback functions (on_update, on_status) to handle incoming data packets. The structure of the streamed JSON differs from the REST response, necessitating distinct parsing logic.

### 2.2 S&P Global Market Intelligence (SPGCI): Modular SDK Design

The S&P ecosystem is bifurcated between the newer, Pythonic spgci SDK (focused on commodities and energy) and the traditional XpressAPI (focused on fundamentals and credit).

#### 2.2.1 The spgci Python SDK

This SDK adopts a modern, object-oriented design. Data is organized into domain-specific classes like MarketData, ForwardCurves, and WorldRefineryDatabase.

- **Method Chaining:** The SDK encourages method chaining (e.g., `ci.MarketData().get_assessments_by_symbol_current()`). Agents must be trained on this fluent interface style.
- **Pagination:** Unlike Refinitiv's implicit limits, spgci exposes an explicit `paginate=True` parameter. Agents must set this to `True` for any bulk extraction to ensure complete datasets.
- **Raw vs. DataFrame:** The SDK returns Pandas DataFrames by default (`raw=False`). However, for debugging schema mismatches, the agent can request `raw=True` to inspect the underlying JSON response.

#### 2.2.2 XpressAPI and Capital IQ Fundamentals

For core financial data (Income Statements, Balance Sheets), the interface changes. The XpressAPI uses a RESTful structure where requests are defined by specific templates (e.g., Financial Institutions vs. Corporates).

- **Period Mnemonics:** S&P uses a rigid syntax for time periods: `IQ_FY` (Fiscal Year), `IQ_LTM` (Last Twelve Months), `IQ_NTM` (Next Twelve Months). An agent requesting "last year's revenue" must translate this intent into the specific `IQ_FY-1` mnemonic to retrieve accurate data.
- **Sector Specificity:** The schema for a Bank (using "Gross Loans") differs entirely from a Manufacturing firm (using "Gross Margin"). Agents must first query the templatefinder endpoint to determine the correct schema for a given entity before requesting fundamental fields.

### 2.3 AlphaSense: The Semantic Graph

AlphaSense offers a GraphQL API, representing a distinct architectural approach compared to the REST/SDK models of S&P and Refinitiv.

#### 2.3.1 GraphQL Query Construction

The power of AlphaSense lies in semantic search over unstructured text. Agents must construct GraphQL queries that specify:

- **Filters:** Constraints on keyword, dateRange, sourceType (e.g., Broker Research, Transcripts), and companies (by Ticker or CUSIP).
- **Return Fields:** Explicit selection of fields like snippet, documentId, sentiment, and relevanceScore.

This eliminates over-fetching, but requires the agent to have precise knowledge of the schema graph.

#### 2.3.2 Ingestion and RAG

AlphaSense supports an ingestion API (`/ingestion-api/v1/upload-document`) allowing users to upload internal content (PDFs, memos) into the AlphaSense index. This enables a powerful RAG (Retrieval-Augmented Generation) workflow where an agent can cross-reference internal proprietary research against global market documents in a single query. The agent must handle metadata tagging (e.g., attaching tickers to uploaded documents) to ensure discoverability.

---

## 3. Open and Quasi-Open Market Data Architectures

While institutional APIs offer depth, open APIs like Financial Modeling Prep (FMP) and Google Trends offer accessibility and unique alternative signals. These come with their own set of engineering challenges.

### 3.1 Financial Modeling Prep (FMP): The REST Workhorse

FMP is widely used due to its simple REST architecture, but it requires careful handling of data standardization issues.

#### 3.1.1 "As Reported" vs. Standardized Taxonomies

A critical distinction in FMP is the dual-layer data model:

- **Standardized Data (`/income-statement`):** FMP maps company-specific line items to a normalized taxonomy (e.g., revenue, costOfRevenue, grossProfit). This is ideal for cross-sectional screening and comparative analysis.
- **As Reported Data (`/financial-statement-full-as-reported`):** This endpoint returns the raw XBRL tags exactly as filed with the SEC. The JSON keys are highly variable and specific to each company's accounting choices (e.g., `revenuefromcontractwithcustomerexcludingassessedtax` vs `salesrevenuegoodsnet`).

**Agent Strategy:** Agents must be programmed with a decision tree. If the user intent is comparison (e.g., "Compare margins of AAPL and MSFT"), use Standardized endpoints. If the intent is audit or precision (e.g., "What was the exact tax benefit reported in Q3?"), use As Reported endpoints and employ fuzzy matching or semantic search on the JSON keys.

#### 3.1.2 High-Throughput Bulk Extraction

For universe-wide analysis, standard endpoints are too slow. FMP provides Bulk Endpoints (e.g., `income-statement-bulk`) that return massive CSV payloads.

**Memory Safety:** Agents must be explicitly instructed not to load these files entirely into RAM. Instead, they should utilize the `pandas.read_csv(chunksize=N)` pattern or stream the response line-by-line to process data in manageable batches. This prevents out-of-memory (OOM) crashes in production environments.

---

## 4. Alternative Data Engineering: Trends and Transactions

The frontier of alpha generation lies in alternative data. However, datasets like search trends and credit card transactions are rarely "model-ready" and require sophisticated normalization pipelines.

### 4.1 Google Trends: The Stitching Problem

Google Trends does not provide absolute search volumes. It returns an index (0-100) scaled relative to the highest point in the requested timeframe. This creates a "stitching" problem: a value of "80" in a 3-month window is not comparable to a "80" in a 5-year window.

#### 4.1.1 The Overlap Normalization Algorithm

To construct a consistent long-term daily time series, the agent must implement an iterative stitching algorithm:

1. **Windowing:** Request data in overlapping chunks (e.g., 90-day windows with a 15-day overlap).
2. **Anchor Calculation:** Identify the overlapping dates between Window_T and Window_T-1.
3. **Scaling Factor:** Calculate the ratio of values in the overlap period:

$$\text{Scaling Factor} = \frac{\text{Mean}(\text{Overlap}_{T-1})}{\text{Mean}(\text{Overlap}_{T})}$$

4. **Adjustment:** Multiply the values in Window_T by the Scaling Factor to align them with the scale of Window_T-1.
5. **Concatenation:** Merge the adjusted windows into a continuous series.

**Library & Error Handling:** The pytrends library automates some of this, but it is prone to aggressive rate limiting by Google. Agents must implement robust Exponential Backoff logic (e.g., `time.sleep(2**retries)`) to handle `429 Too Many Requests` errors gracefully.

### 4.2 Credit Card Transaction Data: Bias Correction

Transaction data is a powerful proxy for revenue but suffers from Panel Bias. A dataset of 5 million cardholders is not a random sample of the population; it is often skewed towards specific geographies, income brackets, or issuing banks.

#### 4.2.1 Iterative Proportional Fitting (IPF)

Raw aggregation of transaction data leads to skewed forecasts. The data must be re-weighted to match the census demographics of the target population.

**Algorithm:** The Random Iterative Method (RIM) or Iterative Proportional Fitting (IPF) (also known as the Deming-Stephan algorithm) is the standard solution.

1. **Target Marginals:** Define the known population distribution (e.g., Census Bureau data: 51% Female, 12% Age 65+).
2. **Sample Marginals:** Calculate the distribution in the transaction panel.
3. **Raking:** Iteratively adjust the weights of each panelist until the weighted sample distribution converges to the target marginals across all dimensions (Age, Gender, Region, Income).

**Agent Implementation:** Agents should leverage libraries like `ipfn` or `rim-weighting` to perform this adjustment. The agent must automatically detect the demographic columns in the panel data and map them to the corresponding census categories before running the weighting algorithm.

#### 4.2.2 Aggregation for Signal Extraction

Individual transaction streams are noisy and sparse. Research demonstrates that Random Forest classifiers significantly outperform other models (including deep learning) on aggregated transaction data for tasks like fraud detection or revenue forecasting. The aggregation process—summing spend by merchant category, day of week, and zip code—acts as a feature engineering step that stabilizes the signal against noise and population drift.

---

## 5. Quantitative Engineering Patterns

In a production environment, simply "getting the data" is insufficient. The data must be structurally sound, temporally accurate, and correctly identified.

### 5.1 Robust Symbology and Identifier Resolution

Financial instruments are identified by a bewildering array of codes: Tickers, CUSIPs, ISINs, SEDOLs, FIGIs, and RICs. Tickers are volatile; "FB" becomes "META", "GOOG" splits into "GOOG" and "GOOGL".

#### 5.1.1 The Symbology Resolver Pattern

Agents must never rely on tickers for historical analysis. They must implement a Symbology Resolver pattern:

- **Immutable Anchor:** Use a persistent ID internally (CIK for US stocks, FIGI for global assets, or a vendor-specific PermID).
- **Time-Aware Mapping:** A map is not a dictionary; it is a function of time. `Map(Ticker='FB', Date='2020-01-01')` returns Meta's ID.
- **Cross-Reference APIs:**
  - **Refinitiv:** Use `lseg.data.discovery.convert_symbols` to translate between RICs and ISINs.
  - **EODHD:** Use the ID Mapping API to resolve ISINs to Tickers.
  - **SEC:** Map CUSIPs to Tickers by scraping 13F filings, which provide a snapshot of valid CUSIP-Ticker pairs at specific quarterly intervals.

### 5.2 Point-in-Time (PIT) Alignment and Look-Ahead Bias

Backtesting requires data exactly as it was known at the moment of the trade. Using restated financial statements or future corporate actions introduces Look-Ahead Bias, rendering the backtest invalid.

#### 5.2.1 The merge_asof Pattern

The Pandas `merge_asof` function is the cryptographic key to PIT integrity. Unlike a standard join, `merge_asof` joins based on the nearest key in a specific direction.

**Usage:** When joining a Price DataFrame (indexed by trading timestamps) with a Fundamentals DataFrame (indexed by Release Date), the agent must specify `direction='backward'`. This ensures that for a trade at T, the system joins the fundamental data released at t <= T.

**Tolerance:** Agents should utilize the `tolerance` parameter to prevent matching stale data (e.g., joining a price today with an earnings report from 2 years ago).

### 5.3 Corporate Actions: Vectorized Adjustments

Stock splits (e.g., 2-for-1) and dividends cause artificial discontinuities in price series. While many APIs provide Adj Close, sophisticated strategies often require custom adjustment logic (e.g., adjusting for splits but not dividends).

#### 5.3.1 Vectorized Cumulative Products

Iterative loops to adjust prices are computationally expensive. Agents must use Vectorized Pandas operations:

1. **Split Factor Column:** Create a column `split_factor` (default 1.0, takes value of split ratio on ex-date).
2. **Cumulative Product:** Calculate `adjustment_factor = split_factor.cumprod()` scanning backwards from the present.
3. **Broadcasting:** Apply the adjustment: `adjusted_price = raw_price * adjustment_factor`.

This approach handles thousands of assets and decades of history in milliseconds.

### 5.4 Factor Analysis with Alphalens

Once data is cleaned and aligned, the agent needs to evaluate its predictive power. Alphalens is the industry-standard library for this factor analysis.

- **Quantile Analysis:** Alphalens bins the asset universe into quantiles (e.g., deciles) based on the factor value and calculates the forward returns for each bin. This reveals if the signal is monotonic (higher factor score = higher return).
- **Information Coefficient (IC):** It calculates the Spearman rank correlation between factor values and forward returns, providing a robust metric of predictive power.
- **Turnover Analysis:** High-turnover factors incur transaction costs. Alphalens quantifies the stability of the factor signal over time.

**Agent Instruction:** Agents should use `alphalens.utils.get_clean_factor_and_forward_returns` to prepare the data, ensuring that the `periods` parameter aligns with the investment horizon.

---

## 6. Governance and Validation: "Trust but Verify"

Autonomous agents are prone to silent failures—generating code that runs without error but produces garbage data. A robust pipeline must enforce strict data contracts.

### 6.1 Statistical Typing with Pandera

Standard Python type hints (`int`, `float`) are insufficient for DataFrames. A column can be of type float but contain invalid negative prices or NaN values. Pandera enables Statistical Typing—validating the content and distribution of the data.

#### 6.1.1 Schema Definition

Agents should be instructed to define `DataFrameSchema` objects that enforce business logic:

- **Range Checks:** `pa.Check.gt(0)` for prices and volumes.
- **Set Membership:** `pa.Check.isin()` for currency columns.
- **Null Safety:** Explicitly defining `nullable=False` for critical keys like Date or Ticker.

#### 6.1.2 Runtime Validation

By decorating data ingestion functions with `@pa.check_output(schema)`, the agent ensures that any API response violating the contract triggers an immediate `SchemaError`. This "fail-fast" mechanism prevents corrupted data from propagating downstream into trading models.

### 6.2 Data Contracts with Great Expectations

For pipeline-level integration tests, Great Expectations (GX) provides a declarative framework for data quality.

- **Expectations:** The agent should generate "Expectations" that describe the valid shape of the data (e.g., `expect_column_values_to_be_unique("transaction_id")`, `expect_column_mean_to_be_between("daily_return", -0.1, 0.1)`).
- **Data Docs:** GX automatically compiles these validation results into HTML "Data Docs," providing a human-readable audit trail. This is essential for compliance and debugging in automated systems.

### 6.3 The .cursorrules Governance Layer

To enforce these best practices, developers must define a `.cursorrules` file (or system prompt) that constrains the agent's behavior. This file acts as a "Constitution" for the AI.

**Example .cursorrules for Financial Engineering:**

```
Financial Data Engineering Standards

1. Data Integrity
- NEVER drop rows silently. Log all dropped data.
- ALWAYS validate schemas using Pandera at ingestion and egress points.
- Check for NaN, Inf, and negative values in Price/Volume fields.

2. Time Series Handling
- ALWAYS use pd.merge_asof with direction='backward' for merging time series.
- ENSURE all DataFrames have a DatetimeIndex that is localized (prefer UTC).
- EXPLICITLY handle Look-Ahead Bias by verifying release dates vs. observation dates.

3. API Usage
- Refinitiv: Use get_history for timeseries, get_data only for static reference.
- S&P: Use spgci SDK with paginate=True. Use strict mnemonics (e.g., IQ_LTM).
- Google Trends: Implement backoff/retry logic for 429 errors.

4. Coding Style
- Prefer Vectorized operations (Pandas/Numpy) over loops.
- Use functional patterns for data transformations.
- Structure code into modular ETL pipelines compatible with Great Expectations.
```

---

## 7. Conclusion

The transition to AI-native financial analysis is not simply about using better models; it is about engineering better context. By restructuring documentation into llms.txt, implementing rigorous symbology and PIT patterns, and enforcing data quality via Pandera and Great Expectations, we transform the chaotic noise of financial data into a structured signal that autonomous agents can reliably navigate.

This architecture shifts the role of the quantitative developer from writing ETL scripts to designing the "cognitive environment" in which the agent operates. As APIs evolve towards self-description and agents gain larger context windows, this framework provides the necessary guardrails to turn generative AI from a novelty into a production-grade infrastructure for alpha generation. The future belongs to those who can effectively "teach" their data to speak the language of the agent.

---

## Data Source Index

| Category | Key Sources |
|----------|-------------|
| Refinitiv (LSEG) | Session management, get_data vs get_history, streaming |
| S&P Global | spgci SDK, XpressAPI, Capital IQ mnemonics |
| FMP | Standardized vs As-Reported, Bulk endpoints |
| AlphaSense | GraphQL API, semantic search, RAG ingestion |
| Google Trends | Overlap normalization, pytrends, rate limiting |
| Credit Card/Bias | IPF algorithm, demographic weighting |
| Validation/Governance | Pandera, Great Expectations, schema validation |
| Context/llms.txt | Documentation standards, MCP protocol |
| Factor Analysis | Alphalens, quantile analysis, IC calculation |
| Symbology/Mapping | FIGI, CIK, PermID, time-aware resolution |
| PIT/Merging | merge_asof, direction='backward', tolerance |

---

## Quick Reference: API Patterns

### Refinitiv Data Library

```python
import lseg.data as ld

# Session initialization
ld.open_session()

# Historical data (CORRECT for time series)
df = ld.get_history(
    universe=['AAPL.O', 'MSFT.O'],
    fields=['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME'],
    interval='daily',
    start='2020-01-01',
    end='2024-01-01'
)

# Snapshot data (for current/static values only)
df = ld.get_data(
    universe=['AAPL.O'],
    fields=['TR.Revenue', 'TR.CompanyMarketCap']
)
```

### S&P Global (spgci)

```python
import spglobal.marketintelligence as ci

# Always use paginate=True for bulk extraction
df = ci.MarketData().get_assessments_by_symbol_current(
    symbol='PCAAS00',
    paginate=True
)

# Period mnemonics
# IQ_FY = Fiscal Year
# IQ_LTM = Last Twelve Months
# IQ_NTM = Next Twelve Months
# IQ_FY-1 = Previous Fiscal Year
```

### Financial Modeling Prep (FMP)

```python
import requests
import pandas as pd

# Standardized (for comparison)
url = f"https://financialmodelingprep.com/api/v3/income-statement/AAPL?apikey={API_KEY}"

# As-Reported (for precision/audit)
url = f"https://financialmodelingprep.com/api/v3/financial-statement-full-as-reported/AAPL?apikey={API_KEY}"

# Bulk extraction with chunking (memory safe)
for chunk in pd.read_csv(bulk_url, chunksize=10000):
    process(chunk)
```

### Point-in-Time Merge Pattern

```python
import pandas as pd

# CORRECT: merge_asof with backward direction
merged = pd.merge_asof(
    prices.sort_values('trade_date'),
    fundamentals.sort_values('release_date'),
    left_on='trade_date',
    right_on='release_date',
    by='ticker',
    direction='backward',
    tolerance=pd.Timedelta('90 days')
)
```

### Pandera Schema Validation

```python
import pandera as pa

price_schema = pa.DataFrameSchema({
    "date": pa.Column(pa.DateTime, nullable=False),
    "ticker": pa.Column(str, nullable=False),
    "close": pa.Column(float, pa.Check.gt(0)),
    "volume": pa.Column(int, pa.Check.ge(0)),
})

@pa.check_output(price_schema)
def load_prices(ticker: str) -> pd.DataFrame:
    # Data loading logic
    ...
```

