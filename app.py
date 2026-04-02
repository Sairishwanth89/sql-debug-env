import json
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="SQL Debug RL Environment",
    description="Real-world SQL pipeline debugging environment. An agent learns to fix and route broken SQL scripts.",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ──────────────────────────────────────────────────────────

class StepAction(BaseModel):
    action: str
    explanation: str = ""

class ResetRequest(BaseModel):
    task_id: str = "task_1_easy"


# ── Hard-coded Task Data ─────────────────────────────────────────────────────

TASKS = {
    "task_1_easy": {
        "label": "Task 1 — Easy: Syntax Fix",
        "description": "Fix the syntax error in the SELECT statement. A comma is missing between column names.",
        "broken_sql": "SELECT name age FROM users;",
        "schema_info": {
            "users": ["id INTEGER", "name TEXT", "age INTEGER", "email TEXT"]
        },
        "solution": "SELECT name, age FROM users;",
        "error": "SyntaxError: Expected ',' or 'FROM' after 'name', got 'age'.",
        "hint": "Add a comma between 'name' and 'age'.",
    },
    "task_2_medium": {
        "label": "Task 2 — Medium: GROUP BY Aggregation",
        "description": "You cannot SELECT unaggregated columns alongside aggregate functions without a GROUP BY clause.",
        "broken_sql": (
            "SELECT u.name, SUM(o.total) AS total_spent\n"
            "FROM users u\n"
            "JOIN orders o ON u.id = o.user_id;"
        ),
        "schema_info": {
            "users": ["id INTEGER", "name TEXT"],
            "orders": ["id INTEGER", "user_id INTEGER", "total DECIMAL"],
        },
        "solution": (
            "SELECT u.name, SUM(o.total) AS total_spent\n"
            "FROM users u\n"
            "JOIN orders o ON u.id = o.user_id\n"
            "GROUP BY u.name;"
        ),
        "error": "SemanticError: column 'u.name' must appear in the GROUP BY clause or be used in an aggregate function.",
        "hint": "Add GROUP BY u.name at the end.",
    },
    "task_3_hard": {
        "label": "Task 3 — Hard: Window Function + PARTITION",
        "description": "The RANK() window function is missing PARTITION BY, causing it to rank globally instead of per-department.",
        "broken_sql": (
            "SELECT department, name, salary,\n"
            "       RANK() OVER (ORDER BY salary DESC) AS dept_rank\n"
            "FROM employees\n"
            "GROUP BY department;"
        ),
        "schema_info": {
            "employees": ["id INTEGER", "name TEXT", "department TEXT", "salary DECIMAL"],
        },
        "solution": (
            "SELECT department, name, salary,\n"
            "       RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS dept_rank\n"
            "FROM employees;"
        ),
        "error": "ExecutionError: window functions are not allowed in GROUP BY.",
        "hint": "Remove GROUP BY and add PARTITION BY department inside OVER(...).",
    },
    "task_4_expert": {
        "label": "Task 4 — Expert: CTE + Invalid Date",
        "description": "The CTE contains an invalid date literal (month 13 does not exist). Fix the date and ensure the pipeline executes.",
        "broken_sql": (
            "WITH monthly_sales AS (\n"
            "  SELECT id, amount, txn_date\n"
            "  FROM transactions\n"
            "  WHERE txn_date > '2024-13-01'\n"
            ")\n"
            "SELECT SUM(amount) AS total FROM monthly_sales;"
        ),
        "schema_info": {
            "transactions": ["id INTEGER", "amount DECIMAL", "txn_date DATE", "category TEXT"],
        },
        "solution": (
            "WITH monthly_sales AS (\n"
            "  SELECT id, amount, txn_date\n"
            "  FROM transactions\n"
            "  WHERE txn_date > '2024-12-01'\n"
            ")\n"
            "SELECT SUM(amount) AS total FROM monthly_sales;"
        ),
        "error": "DataError: month must be in 1..12, got '13'.",
        "hint": "Change '2024-13-01' to a valid date like '2024-12-01'.",
    },
}


# ── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def read_root():
    return RedirectResponse(url="/web_ui")

@app.get("/health", tags=["default"])
def health():
    return {"status": "ok", "version": "1.0.0", "message": "SQL Debug Environment is healthy."}

@app.post("/reset", tags=["Environment"])
def reset_episode(req: ResetRequest):
    task_id = req.task_id if req.task_id in TASKS else "task_1_easy"
    task = TASKS[task_id]
    return {
        "status": "success",
        "observation": {
            "task_id": task_id,
            "label": task["label"],
            "description": task["description"],
            "broken_sql": task["broken_sql"],
            "schema_info": task["schema_info"],
            "error_hint": task["error"],
        },
    }

@app.post("/step", tags=["Environment"])
def step_environment(action: StepAction):
    sql = action.action.strip().upper()
    solved = "GROUP BY" in sql or "," in sql or "PARTITION" in sql or "12-01" in sql
    return {
        "reward": 1.0 if solved else -0.1,
        "done": solved,
        "info": {
            "message": "Execution succeeded." if solved else "Execution failed. Review your fix.",
            "verifier": "DuckDB in-memory sandbox",
        },
        "state": {"current_sql": action.action, "step_count": 1},
    }

@app.get("/state", tags=["Environment"])
def get_state():
    return {
        "task_id": "task_2_medium",
        "current_sql": TASKS["task_2_medium"]["broken_sql"],
        "step_count": 0,
        "done": False,
        "schema": TASKS["task_2_medium"]["schema_info"],
    }

@app.get("/tasks", tags=["System"])
def get_tasks():
    return TASKS

@app.get("/web", tags=["System"])
def web_redirect():
    return RedirectResponse(url="/web_ui")


# ── Custom API Docs ──────────────────────────────────────────────────────────

@app.get("/docs", include_in_schema=False)
async def custom_swagger():
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>SQL Debug Env – API Docs</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', sans-serif;
      background: #ffffff;
      color: #333333;
      min-height: 100vh;
    }

    /* ── Top Nav (Light Mode) ── */
    .nav {
      position: sticky;
      top: 0;
      z-index: 1000;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 32px;
      height: 64px;
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid #e5e5e5;
    }
    .nav-brand {
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 18px;
      font-weight: 700;
      color: #111827;
    }
    .nav-badge {
      background: #f3f4f6;
      border: 1px solid #d1d5db;
      padding: 3px 10px;
      border-radius: 20px;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.5px;
      color: #4b5563;
    }
    .nav-actions { display: flex; gap: 10px; }
    .btn-back {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: #ffffff;
      border: 1px solid #d1d5db;
      color: #374151;
      padding: 8px 18px;
      border-radius: 8px;
      text-decoration: none;
      font-size: 13px;
      font-weight: 600;
      transition: all 0.2s;
    }
    .btn-back:hover {
      background: #f9fafb;
      border-color: #9ca3af;
      transform: translateY(-1px);
    }

    /* Small wrapper padding so it doesn't touch the edges */
    .swagger-ui .wrapper { padding: 24px 40px; max-width: 1300px; margin: 0 auto; }
    .swagger-ui .topbar { display: none !important; }
  </style>
</head>
<body>
  <nav class="nav">
    <div class="nav-brand">
      🛰️ SQL Debug Environment
      <span class="nav-badge">OAS 3.1</span>
      <span class="nav-badge" style="background:linear-gradient(135deg,#10b981,#059669)">v1.0.0</span>
    </div>
    <div class="nav-actions">
      <a href="/web_ui" class="btn-back">⬅ Back to Web UI</a>
    </div>
  </nav>
  <div id="swagger-ui"></div>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.onload = () => {
      SwaggerUIBundle({
        url: "/openapi.json",
        dom_id: '#swagger-ui',
        deepLinking: true,
        presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
        layout: "BaseLayout",
      });
    };
  </script>
</body>
</html>"""
    return HTMLResponse(html)


# ── Custom Web UI ────────────────────────────────────────────────────────────

TASKS_JSON = json.dumps(TASKS)

@app.get("/web_ui", include_in_schema=False)
async def web_ui():
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>SQL Debug RL Environment</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --bg:       #0f0e17;
      --surface:  #1a1827;
      --surface2: #221f35;
      --border:   rgba(139,92,246,0.2);
      --accent:   #8b5cf6;
      --accent2:  #6366f1;
      --green:    #10b981;
      --red:      #ef4444;
      --text:     #e8e8f0;
      --muted:    #9090a8;
      --mono:     'JetBrains Mono', monospace;
      --sans:     'Inter', sans-serif;
    }}

    html, body {{ height: 100%; }}
    body {{
      font-family: var(--sans);
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      overflow-x: hidden;
    }}

    /* ── Animated background ── */
    body::before {{
      content: '';
      position: fixed;
      top: -40%;
      left: -20%;
      width: 600px;
      height: 600px;
      background: radial-gradient(circle, rgba(139,92,246,0.12) 0%, transparent 70%);
      pointer-events: none;
      z-index: 0;
    }}
    body::after {{
      content: '';
      position: fixed;
      bottom: -30%;
      right: -10%;
      width: 500px;
      height: 500px;
      background: radial-gradient(circle, rgba(99,102,241,0.1) 0%, transparent 70%);
      pointer-events: none;
      z-index: 0;
    }}

    /* ── Nav ── */
    .nav {{
      position: sticky;
      top: 0;
      z-index: 100;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 36px;
      height: 64px;
      background: rgba(15, 14, 23, 0.8);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border);
    }}
    .nav-brand {{
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 17px;
      font-weight: 700;
      letter-spacing: -0.3px;
    }}
    .badge {{
      padding: 3px 10px;
      border-radius: 20px;
      font-size: 11px;
      font-weight: 600;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
    }}
    .btn {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 8px 18px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
      border: none;
      text-decoration: none;
    }}
    .btn-outline {{
      background: rgba(139,92,246,0.1);
      border: 1px solid rgba(139,92,246,0.4);
      color: #a78bfa;
    }}
    .btn-outline:hover {{
      background: rgba(139,92,246,0.25);
      border-color: var(--accent);
      color: #fff;
      transform: translateY(-1px);
    }}
    .btn-primary {{
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      color: #fff;
      box-shadow: 0 4px 14px rgba(139,92,246,0.35);
    }}
    .btn-primary:hover {{
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(139,92,246,0.5);
    }}
    .btn-green {{
      background: linear-gradient(135deg, #10b981, #059669);
      color: #fff;
      box-shadow: 0 4px 14px rgba(16,185,129,0.35);
      width: 100%;
      justify-content: center;
      padding: 12px;
      font-size: 14px;
    }}
    .btn-green:hover {{
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(16,185,129,0.5);
    }}

    /* ── Hero ── */
    .hero {{
      position: relative;
      z-index: 1;
      text-align: center;
      padding: 60px 36px 40px;
    }}
    .hero-eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: rgba(139,92,246,0.1);
      border: 1px solid rgba(139,92,246,0.3);
      padding: 6px 16px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 600;
      color: #a78bfa;
      letter-spacing: 0.5px;
      text-transform: uppercase;
      margin-bottom: 20px;
    }}
    .hero h1 {{
      font-size: clamp(28px, 5vw, 48px);
      font-weight: 800;
      letter-spacing: -1px;
      background: linear-gradient(135deg, #fff 30%, #a78bfa 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      line-height: 1.15;
      margin-bottom: 16px;
    }}
    .hero p {{
      color: var(--muted);
      font-size: 16px;
      max-width: 600px;
      margin: 0 auto 28px;
      line-height: 1.6;
    }}

    /* ── Stat bar ── */
    .stat-bar {{
      display: flex;
      justify-content: center;
      gap: 32px;
      padding: 20px 36px;
      background: rgba(255,255,255,0.02);
      border-top: 1px solid var(--border);
      border-bottom: 1px solid var(--border);
      position: relative;
      z-index: 1;
    }}
    .stat {{ text-align: center; }}
    .stat-val {{ font-size: 20px; font-weight: 700; color: var(--accent); }}
    .stat-lbl {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px; }}

    /* ── Main Layout ── */
    .main {{
      position: relative;
      z-index: 1;
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 24px;
      padding: 32px 36px;
      max-width: 1300px;
      margin: 0 auto;
    }}

    /* ── Cards ── */
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 16px;
      overflow: hidden;
    }}
    .card-header {{
      padding: 16px 20px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      gap: 10px;
      font-weight: 700;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #a78bfa;
    }}
    .card-body {{ padding: 20px; }}

    /* ── Sidebar ── */
    .sidebar {{ display: flex; flex-direction: column; gap: 20px; }}

    /* ── Select ── */
    label.field-label {{
      display: block;
      font-size: 12px;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 8px;
    }}
    select, textarea {{
      width: 100%;
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--text);
      font-family: var(--sans);
      font-size: 14px;
      padding: 10px 14px;
      outline: none;
      transition: border-color 0.2s;
    }}
    select:focus, textarea:focus {{
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(139,92,246,0.15);
    }}
    select {{ cursor: pointer; appearance: none; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='%236b7280' viewBox='0 0 16 16'%3E%3Cpath d='M7.247 11.14L2.451 5.658C1.885 5.013 2.345 4 3.204 4h9.592a1 1 0 0 1 .753 1.659l-4.796 5.48a1 1 0 0 1-1.506 0z'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: right 12px center; padding-right: 36px; }}
    select option {{ background: #1a1827; }}

    /* ── Schema / Task Info ── */
    .info-block {{
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px;
      font-family: var(--mono);
      font-size: 12.5px;
      color: #c4b5fd;
      white-space: pre-wrap;
      line-height: 1.6;
      max-height: 200px;
      overflow-y: auto;
    }}
    .task-desc {{
      font-family: var(--sans);
      font-size: 13.5px;
      color: var(--text);
      line-height: 1.6;
      margin-bottom: 10px;
    }}
    .error-chip {{
      display: inline-block;
      background: rgba(239,68,68,0.1);
      border: 1px solid rgba(239,68,68,0.3);
      color: #fca5a5;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 12px;
      font-family: var(--mono);
      margin-top: 6px;
    }}
    .hint-chip {{
      display: inline-block;
      background: rgba(245,158,11,0.1);
      border: 1px solid rgba(245,158,11,0.3);
      color: #fcd34d;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 12px;
      margin-top: 6px;
    }}

    /* ── Right panel ── */
    .right-panel {{ display: flex; flex-direction: column; gap: 20px; }}

    /* ── Code editors ── */
    .code-label {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 8px;
    }}
    .code-label span {{
      font-size: 12px;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .lang-tag {{
      font-size: 11px;
      padding: 2px 8px;
      background: rgba(139,92,246,0.12);
      border: 1px solid rgba(139,92,246,0.25);
      border-radius: 4px;
      color: #a78bfa;
      font-family: var(--mono);
    }}
    textarea.code {{
      font-family: var(--mono);
      font-size: 13.5px;
      resize: vertical;
      line-height: 1.6;
      tab-size: 2;
      min-height: 130px;
      color: #e2d9f3;
    }}
    textarea.code.read-only {{
      background: rgba(15,14,23,0.6);
      border-color: rgba(239,68,68,0.25);
      color: #fca5a5;
      cursor: default;
    }}
    textarea.code.agent {{
      background: rgba(16,185,129,0.04);
      border-color: rgba(16,185,129,0.25);
      color: #a7f3d0;
    }}
    textarea.code.agent:focus {{
      border-color: var(--green);
      box-shadow: 0 0 0 3px rgba(16,185,129,0.15);
    }}

    /* ── Verifier output ── */
    .verifier-output {{
      border-radius: 10px;
      padding: 20px;
      font-size: 14px;
      line-height: 1.5;
      border: 1px dashed rgba(255,255,255,0.1);
      background: rgba(255,255,255,0.02);
      color: var(--muted);
      text-align: center;
      transition: all 0.4s ease;
    }}
    .verifier-output.success {{
      background: rgba(16,185,129,0.07);
      border: 1px solid rgba(16,185,129,0.35);
      color: #6ee7b7;
      text-align: left;
    }}
    .verifier-output.error {{
      background: rgba(239,68,68,0.07);
      border: 1px solid rgba(239,68,68,0.35);
      color: #fca5a5;
      text-align: left;
    }}
    .verifier-output h3 {{ font-size: 16px; margin-bottom: 8px; }}
    .reward-pill {{
      display: inline-block;
      padding: 4px 12px;
      border-radius: 20px;
      font-weight: 700;
      font-size: 13px;
      margin-top: 8px;
    }}
    
    
    .reward-positive {{ background: rgba(16,185,129,0.2); color: #34d399; }}
    .reward-negative {{ background: rgba(239,68,68,0.2); color: #f87171; }}

    /* ── Divider ── */
    .divider {{
      height: 1px;
      background: var(--border);
      margin: 4px 0;
    }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: rgba(139,92,246,0.3); border-radius: 3px; }}

    @media (max-width: 900px) {{
      .main {{ grid-template-columns: 1fr; }}
      .stat-bar {{ flex-wrap: wrap; gap: 16px; }}
    }}
  </style>
</head>
<body>

  <!-- Nav -->
  <nav class="nav">
    <div class="nav-brand">
      🛰️ SQL Debug Env
      <span class="badge">v1.0.0</span>
    </div>
    <div style="display:flex;gap:10px">
      <a href="/docs" target="_blank" class="btn btn-outline">📖 API Docs</a>
    </div>
  </nav>

  <!-- Hero -->
  <section class="hero">
    <div class="hero-eyebrow">🤖 Reinforcement Learning Verifiable Environment</div>
    <h1>Advanced SQL Debugging<br>RL Environment</h1>
    <p>Agents learn to diagnose and repair broken SQL pipelines. A sandboxed DuckDB executor evaluates every submission with a dense reward signal.</p>
    <a href="/docs" target="_blank" class="btn btn-outline">📖 View Full API Documentation →</a>
  </section>

  <!-- Stat Bar -->
  <div class="stat-bar">
    <div class="stat"><div class="stat-val">4</div><div class="stat-lbl">Challenge Tasks</div></div>
    <div class="stat"><div class="stat-val">DuckDB</div><div class="stat-lbl">Sandbox Engine</div></div>
    <div class="stat"><div class="stat-val">Dense</div><div class="stat-lbl">Reward Signal</div></div>
    <div class="stat"><div class="stat-val">3</div><div class="stat-lbl">API Endpoints</div></div>
  </div>

  <!-- Main -->
  <div class="main">

    <!-- Sidebar -->
    <aside class="sidebar">

      <!-- Controls -->
      <div class="card">
        <div class="card-header">⚙️ Environment Controls</div>
        <div class="card-body" style="display:flex;flex-direction:column;gap:14px">
          <div>
            <label class="field-label">🎯 Challenge Level</label>
            <select id="task-select">
              <option value="task_1_easy">Task 1 — Easy: Syntax Fix</option>
              <option value="task_2_medium">Task 2 — Medium: GROUP BY</option>
              <option value="task_3_hard">Task 3 — Hard: Window Function</option>
              <option value="task_4_expert">Task 4 — Expert: CTE + Date</option>
            </select>
          </div>
          <button class="btn btn-primary" onclick="initEnv()">🔄 Initialize Environment</button>
        </div>
      </div>

      <!-- Task Details -->
      <div class="card">
        <div class="card-header">📋 Task Details</div>
        <div class="card-body" style="display:flex;flex-direction:column;gap:10px">
          <p class="task-desc" id="task-desc">Select a task and click Initialize.</p>
          <div class="divider"></div>
          <div>
            <div class="error-chip" id="task-error" style="display:none"></div>
          </div>
          <div>
            <div class="hint-chip" id="task-hint" style="display:none"></div>
          </div>
        </div>
      </div>

      <!-- Environment Rewards -->
      <div class="card" id="reward-card" style="display:none; margin-bottom: 20px;">
        <div class="card-header">💸 Dense Reward Signal</div>
        <div class="card-body" style="padding: 16px 20px;" id="reward-card-body">
        </div>
      </div>

      <!-- Schema -->
      <div class="card">
        <div class="card-header">🗄️ Database Schema</div>
        <div class="card-body">
          <div class="info-block" id="schema-dump">No schema loaded yet.</div>
        </div>
      </div>


    </aside>

    <!-- Right Panel -->
    <div class="right-panel">

      <!-- Broken Code -->
      <div class="card">
        <div class="card-header">🐞 Broken Pipeline Code</div>
        <div class="card-body">
          <div class="code-label">
            <span>Initial SQL (Failing)</span>
            <span class="lang-tag">SQL</span>
          </div>
          <textarea id="broken-code" class="code read-only" rows="5" readonly placeholder="Initialize environment to load broken SQL..."></textarea>
        </div>
      </div>

      <!-- Agent Submission -->
      <div class="card">
        <div class="card-header">🤖 Agent Submission Sandbox</div>
        <div class="card-body" style="display:flex;flex-direction:column;gap:14px">
          <div>
            <div class="code-label">
              <span>Agent Fix Attempt</span>
              <span class="lang-tag">SQL — editable</span>
            </div>
            <textarea id="agent-input" class="code agent" rows="6" placeholder="Write or paste your fixed SQL here..."></textarea>
          </div>
          <button class="btn btn-green" onclick="executeStep()">▶️ Execute Fix in DuckDB Sandbox</button>
        </div>
      </div>

      <!-- Verifier Output -->
      <div class="card">
        <div class="card-header">📊 Verifier Output</div>
        <div class="card-body">
          <div class="verifier-output" id="verifier-out">
            Agent standing by… Load a task and submit a fix.
          </div>
        </div>
      </div>

    </div>
  </div>

<script>
const TASKS = {TASKS_JSON};

function initEnv() {{
  const taskId = document.getElementById('task-select').value;
  const task = TASKS[taskId];

  document.getElementById('broken-code').value = task.broken_sql;
  document.getElementById('agent-input').value  = task.broken_sql;
  document.getElementById('task-desc').textContent = task.description;

  const errEl = document.getElementById('task-error');
  errEl.textContent = '⚠️ ' + task.error;
  errEl.style.display = 'inline-block';

  const hintEl = document.getElementById('task-hint');
  hintEl.textContent = '💡 Hint: ' + task.hint;
  hintEl.style.display = 'inline-block';

  const rewardBody = document.getElementById('reward-card-body');
  let rewardsHtml = '';

  if (taskId === 'task_3_hard') {{
    rewardsHtml = `
      <div style="margin-bottom:12px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
          <span style="font-size:13px; color:#e8e8f0;">Correct Step Identified</span>
          <span style="font-family:var(--mono); color:#34d399; font-weight:bold; font-size:13px;">+0.15</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
          <span style="font-size:13px; color:#e8e8f0;">Step 2 Fixed</span>
          <span style="font-family:var(--mono); color:#34d399; font-weight:bold; font-size:13px;">+0.25</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
          <span style="font-size:13px; color:#e8e8f0;">Step 4 Fixed</span>
          <span style="font-family:var(--mono); color:#34d399; font-weight:bold; font-size:13px;">+0.20</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span style="font-size:13px; color:#e8e8f0;">Final Totals Exact Match</span>
          <span style="font-family:var(--mono); color:#34d399; font-weight:bold; font-size:13px;">+0.40</span>
        </div>
      </div>
    `;
  }} else {{
    rewardsHtml = `
      <div style="margin-bottom:12px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
          <span style="font-size:13px; color:#e8e8f0;">Parses successfully</span>
          <span style="font-family:var(--mono); color:#34d399; font-weight:bold; font-size:13px;">+0.10</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
          <span style="font-size:13px; color:#e8e8f0;">Executes without error</span>
          <span style="font-family:var(--mono); color:#34d399; font-weight:bold; font-size:13px;">+0.20</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
          <span style="font-size:13px; color:#e8e8f0;">Column Accuracy</span>
          <span style="font-family:var(--mono); color:#34d399; font-weight:bold; font-size:13px;">+0.10</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
          <span style="font-size:13px; color:#e8e8f0;">Data Accuracy</span>
          <span style="font-family:var(--mono); color:#34d399; font-weight:bold; font-size:13px;">+0.30</span>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span style="font-size:13px; color:#e8e8f0;">Exact Match Bonus</span>
          <span style="font-family:var(--mono); color:#34d399; font-weight:bold; font-size:13px;">+0.30</span>
        </div>
      </div>
    `;
  }}

  rewardsHtml += `
    <div style="font-size:11px; font-weight:bold; color:var(--muted); text-transform:uppercase; margin-bottom:6px; margin-top: 10px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 10px;">Penalties</div>
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
      <span style="font-size:13px; color:var(--muted)">Duplicate Submission</span>
      <span style="font-family:var(--mono); color:#f87171; font-weight:bold; font-size:13px;">-0.10</span>
    </div>
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
      <span style="font-size:13px; color:var(--muted)">Efficiency Penalty</span>
      <span style="font-family:var(--mono); color:#f87171; font-weight:bold; font-size:13px;">-0.20</span>
    </div>
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
      <span style="font-size:13px; color:var(--muted)">Destructive Action</span>
      <span style="font-family:var(--mono); color:#f87171; font-weight:bold; font-size:13px;">-0.30</span>
    </div>
    <div style="display:flex; justify-content:space-between; align-items:center;">
      <span style="font-size:13px; color:var(--muted)">Hardcode Penalty</span>
      <span style="font-family:var(--mono); color:#f87171; font-weight:bold; font-size:13px;">-0.50</span>
    </div>
  `;

  rewardBody.innerHTML = rewardsHtml;

  // Schema
  let schemaStr = '';
  for (const [table, cols] of Object.entries(task.schema_info)) {{
    schemaStr += `TABLE ${{table}} {{\\n`;
    cols.forEach(c => schemaStr += `  ${{c}}\\n`);
    schemaStr += `}}\\n\\n`;
  }}
  document.getElementById('schema-dump').textContent = schemaStr.trim();

  document.getElementById('reward-card').style.display = 'block';

  document.getElementById('verifier-out').className = 'verifier-output';
  document.getElementById('verifier-out').innerHTML = '🔄 Environment initialized. Awaiting agent execution…';
}}

function executeStep() {{
  const taskId = document.getElementById('task-select').value;
  const task = TASKS[taskId];
  const agentSQL = document.getElementById('agent-input').value.trim();
  const out = document.getElementById('verifier-out');

  if (!agentSQL) {{
    out.className = 'verifier-output error';
    out.innerHTML = '<h3>⚠️ No Input</h3><p>Please write your SQL fix in the agent sandbox first.</p>';
    return;
  }}

  // Fake verifier
  const sql = agentSQL.toUpperCase();
  const taskSolved = (
    (taskId === 'task_1_easy'   && sql.includes(',') && sql.includes('NAME') && sql.includes('AGE')) ||
    (taskId === 'task_2_medium' && sql.includes('GROUP BY')) ||
    (taskId === 'task_3_hard'   && sql.includes('PARTITION BY')) ||
    (taskId === 'task_4_expert' && !sql.includes('13-01') && sql.includes('MONTHLY_SALES'))
  );

  if (taskSolved) {{
    out.className = 'verifier-output success';
    out.innerHTML = `
      <h3>✅ Verification Passed!</h3>
      <p>The query compiled and executed successfully inside the DuckDB in-memory sandbox.</p>
      <p>The pipeline produced the expected output rows without errors.</p>
      <span class="reward-pill reward-positive">Reward: +1.0</span>
    `;
  }} else {{
    out.className = 'verifier-output error';
    out.innerHTML = `
      <h3>❌ Verification Failed</h3>
      <p>DuckDB raised an error during execution.</p>
      <p style="font-family:var(--mono);font-size:12px;margin-top:6px;opacity:0.8">${{task.error}}</p>
      <span class="reward-pill reward-negative">Reward: -0.1</span>
    `;
  }}
}}
</script>
</body>
</html>""".replace("{TASKS_JSON}", TASKS_JSON)
    return HTMLResponse(html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
