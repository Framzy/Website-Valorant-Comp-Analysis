# README.md

````md
# Valorant Agent Comp Analyst — Frontend V2

Frontend application for the **Valorant Agent Comp Analyst V2** project.

The application provides two main features:

- **General Analysis** — analyzes historical agent compositions based on map, year, and selected agents.
- **Team Prediction** — predicts composition performance based on team, map, year, and selected agents.

The frontend communicates with the V2 backend through a REST API.

---

## Tech Stack

### Frontend

- HTML5
- CSS3
- Vanilla JavaScript
- ES Modules
- jQuery
- Axios
- Vite
- npm

### Backend

- Python
- Flask
- REST API

The frontend and backend are developed separately.

```text
Frontend
HTML + CSS + Vanilla JavaScript
        │
        ├── jQuery
        │
        └── Axios
              │
              ▼
        Flask REST API
```
````

---

## Why Vanilla JavaScript?

This project intentionally uses **Vanilla JavaScript** as the primary application layer instead of a frontend framework such as React.

The purpose is to maintain direct control over:

- DOM manipulation
- Event handling
- Application state
- JavaScript modules
- Functions
- Asynchronous operations
- Promises
- `async` / `await`
- API integration
- Error handling

The project also provides an opportunity to compare Vanilla JavaScript with libraries and frameworks that are commonly used in modern frontend development.

---

## Why jQuery?

jQuery is included as a practical frontend library for:

- DOM selection
- DOM manipulation
- Event handling
- UI utilities

Example:

```javascript
$(".agent-card").on("click", handleAgentClick);
```

jQuery is not intended to replace the application's JavaScript logic.

The project separates responsibilities between Vanilla JavaScript and jQuery:

```text
Vanilla JavaScript
├── Application logic
├── State
├── Validation
├── Modules
└── Application flow

jQuery
├── DOM selection
├── DOM manipulation
└── Event utilities
```

This allows jQuery to be used in a controlled way while maintaining understanding of the underlying JavaScript and browser APIs.

---

## Why Axios?

Axios is used as the HTTP client for communication between the frontend and Flask backend.

Example:

```javascript
import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:5000/api",
});
```

Axios is responsible for:

- GET requests
- POST requests
- Sending request payloads
- Receiving API responses
- HTTP error handling
- Centralized API configuration

API communication will be centralized in `api.js` instead of being distributed throughout the application.

---

## Why Vite?

Vite is used as the frontend development and build tool.

Vite provides:

- Development server
- ES Module support
- Fast development workflow
- npm dependency management
- Production build
- Modern frontend project structure

Vite is used independently of React.

This project uses:

```text
Vite + Vanilla JavaScript
```

rather than:

```text
Vite + React
```

---

## Why npm and Node.js?

npm is used to manage frontend dependencies.

Current dependencies include:

- Vite
- Axios
- jQuery

Node.js is used as the runtime for frontend development tools such as npm and Vite.

Node.js is **not used as the backend server**.

The backend remains:

```text
Python + Flask
```

---

# Project Structure

The current project structure is:

```text
app/
├── .gitignore
├── index.html
├── package.json
├── package-lock.json
├── README.md
│
├── public/
│
└── src/
    ├── assets/
    │
    ├── main.js
    └── style.css
```

The JavaScript structure will be gradually refactored into ES Modules.

The planned structure is:

```text
app/
├── .gitignore
├── index.html
├── package.json
├── package-lock.json
├── README.md
│
├── public/
│
└── src/
    ├── assets/
    │   └── logo_valorant.png
    │
    ├── js/
    │   ├── main.js
    │   ├── state.js
    │   ├── api.js
    │   ├── agents.js
    │   ├── general.js
    │   ├── team.js
    │   ├── ui.js
    │   └── utils.js
    │
    └── style.css
```

The module structure will be introduced gradually during the frontend migration.

---

# JavaScript Module Responsibilities

## `main.js`

Application entry point.

Responsibilities:

- Initialize the application
- Initialize modules
- Register global event listeners
- Start the application

---

## `state.js`

Stores shared application state.

Examples:

```javascript
currentMode;
selectedAgents;
selectedYear;
selectedMap;
selectedTeam;
```

The goal is to avoid scattering important application state throughout different files.

---

## `api.js`

Centralized Axios communication with the Flask backend.

Responsibilities:

- Axios configuration
- API requests
- Request payloads
- Response handling
- API errors

Planned API functions include:

```text
analyzeGeneral()
predictTeam()
getTeamYears()
getTeamMaps()
getTeamTeams()
```

---

## `agents.js`

Responsible for agent-related functionality.

Examples:

- Agent metadata
- Agent cards
- Agent selection
- Selected-agent counter
- Role filtering
- Agent validation

---

## `general.js`

Responsible for the General Analysis feature.

Responsibilities:

- General Analysis input
- Input validation
- API request
- Response processing
- Result rendering
- Reset functionality

---

## `team.js`

Responsible for the Team Prediction feature.

Responsibilities:

- Year selection
- Map selection
- Team selection
- Agent selection
- Cascading options
- Prediction request
- Prediction result rendering
- Reset functionality

The intended selection flow is:

```text
Year
  ↓
Map
  ↓
Team
  ↓
Agents
  ↓
Predict
```

---

## `ui.js`

Contains reusable UI functionality.

Examples:

- Loading state
- Error popup
- Result visibility
- Gauge
- Generic UI transitions

---

## `utils.js`

Contains genuinely reusable helper functions.

This file should not become a dumping ground for unrelated application logic.

---

# Backend API

The frontend communicates with the V2 Flask backend.

Default backend:

```text
http://localhost:5000
```

---

## General Analysis

### Endpoint

```http
POST /api/general/analyze
```

### Example Request

```json
{
  "map": "Abyss",
  "year": 2024,
  "agents": ["cypher", "jett", "kayo", "omen", "sova"]
}
```

The response contains information such as:

- Input
- Historical composition
- Pick rate
- Winrate
- Total maps
- Playstyle
- Role pattern
- Historical recommendations
- Fallback information

General Analysis is based on historical data rather than machine-learning prediction.

---

# Team Prediction

### Endpoint

```http
POST /api/team/predict
```

### Example Request

```json
{
  "team": "FNATIC",
  "map": "Abyss",
  "year": 2024,
  "agents": ["astra", "killjoy", "sova", "viper", "yoru"]
}
```

The response contains:

- Predicted winrate
- Predicted percentage
- Role composition
- Historical information
- Composition strength
- Inference information

Team Prediction uses the trained V2 machine-learning model.

---

# Team Options API

The Team Prediction form uses cascading options.

## Available Years

```http
GET /api/team/options/years
```

Example:

```json
[2024, 2025, 2026]
```

---

## Available Maps

```http
GET /api/team/options/maps?year=2025
```

Example:

```json
{
  "year": 2025,
  "maps": ["Abyss", "Ascent"]
}
```

---

## Available Teams

```http
GET /api/team/options/teams?year=2025&map=Abyss
```

Example:

```json
{
  "year": 2025,
  "map": "Abyss",
  "teams": ["FNATIC"]
}
```

The intended frontend flow is:

```text
Year
 ↓
Map
 ↓
Team
```

---

# Application Modes

## General Analysis

General Analysis analyzes historical compositions.

The main information includes:

- Historical winrate
- Pick rate
- Total maps played
- Playstyle
- Role pattern
- Historical recommendations

General Analysis does **not** represent an ML prediction.

---

## Team Prediction

Team Prediction estimates the performance of a composition using the V2 machine-learning model.

The prediction is based on:

```text
Team
+
Map
+
Year
+
5 Agents
```

The resulting prediction is represented as a winrate.

Historical information shown alongside the prediction refers to the submitted composition.

It is not a separate Top Historical Composition ranking.

---

# Development

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Build the application:

```bash
npm run build
```

Preview the production build:

```bash
npm run preview
```

---

# Development Workflow

The frontend will be migrated incrementally from the previous implementation.

```text
Existing Frontend
       ↓
Vite Setup
       ↓
Migrate HTML
       ↓
Migrate CSS
       ↓
Migrate Assets
       ↓
Refactor JavaScript
       ↓
ES Modules
       ↓
Axios API Layer
       ↓
General Analysis V2
       ↓
Team Prediction V2
       ↓
Testing
       ↓
UI Refinement
```

The existing UI will be preserved where possible.

The main changes are:

- Frontend tooling
- Project structure
- JavaScript architecture
- API integration
- V2 input requirements
- V2 result rendering

---

# Development Principles

## 1. Keep responsibilities separated

The application should avoid having one large JavaScript file.

```text
state.js   → State
api.js     → HTTP/API
agents.js  → Agent functionality
general.js → General Analysis
team.js    → Team Prediction
ui.js      → Generic UI
utils.js   → Generic helpers
```

---

## 2. Centralize API communication

Feature modules should not create their own Axios configuration.

Instead:

```text
general.js ──┐
             │
team.js ─────┼──→ api.js ──→ Axios ──→ Flask
             │
other.js ────┘
```

---

## 3. Keep jQuery usage controlled

jQuery is primarily used for DOM and event utilities.

Application logic should remain understandable as standard JavaScript.

---

## 4. Avoid unnecessary abstraction

The project should remain simple and maintainable.

Do not introduce additional:

- Frameworks
- State-management libraries
- UI libraries
- Utility libraries

unless they solve an actual project requirement.

---

## 5. Do not carry over obsolete V1 concepts

The V2 frontend should not depend on fields that only existed in the previous backend.

Examples of V1 concepts that should not automatically be carried forward:

```text
adjusted_pred
penalty_details
confidence
sim_score
most_common_combo
```

They should only be used if they are explicitly provided and required by the V2 backend.

---

# Current Status

## Backend V2

- [x] General Analysis API
- [x] Team Prediction API
- [x] Team available years API
- [x] Team available maps API
- [x] Team available teams API
- [x] Backend API testing

## Frontend V2

- [x] Existing frontend audit
- [x] Frontend architecture defined
- [x] Vite project setup
- [x] npm setup
- [x] Axios installed
- [x] jQuery installed
- [ ] Existing HTML migration
- [ ] Existing CSS migration
- [ ] Asset migration
- [ ] JavaScript module refactoring
- [ ] Axios API layer
- [ ] General Analysis V2 integration
- [ ] General result rendering
- [ ] Team cascading options
- [ ] Team Prediction V2 integration
- [ ] Team result rendering
- [ ] Final frontend testing

---

# Development Environment

Example development setup:

```text
Frontend
http://localhost:5173
       │
       │ Axios
       ▼
Backend
http://localhost:5000
```

The frontend and backend run as separate applications during development.

---

# Project Goal

The goal of Frontend V2 is to provide a clean and maintainable frontend for the Valorant Predictor V2 backend while providing practical experience with modern JavaScript development.

The project combines:

- Vanilla JavaScript
- ES Modules
- jQuery
- Axios
- Vite
- npm
- REST API
- Flask

without introducing a frontend framework.

````

### Status kita sekarang

Package Anda sudah sesuai:

```text
axios  → HTTP client
jquery → DOM/event utility
vite   → dev/build tooling
npm    → dependency management
````

Dan struktur saat ini **belum perlu diubah lagi**:

```text
app/
├── public/
├── src/
│   ├── assets/
│   ├── main.js
│   └── style.css
├── index.html
├── package.json
├── package-lock.json
├── README.md
└── .gitignore
```
