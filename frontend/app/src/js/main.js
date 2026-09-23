import "../css/base.css";
import "../css/components.css";
import "../css/results.css";
import "../css/responsive.css";

import { state } from "./state.js";
import { initAgents, resetAgents } from "./agents.js";
import { closeError, hideResults } from "./ui.js";
import { getAvailableYears, getAvailableMaps } from "./api.js";
import { logger } from "./logger.js";

function init() {
  logger.info("Application initialization started.");

  initModeTabs();
  logger.debug("Mode tabs initialized.");

  initSharedInputEvents();
  logger.debug("Shared input events initialized.");

  initErrorEvents();
  logger.debug("Error events initialized.");

  initAgents();
  logger.debug("Agent system initialized.");

  testSharedOptions();

  updateModeUI(state.currentMode);

  logger.info("Application initialization completed.", {
    mode: state.currentMode,
    selectedAgents: state.selectedAgents,
    maxAgents: state.maxAgents,
  });
}

/* =========================================================
   TEST SHARED OPTIONS API
   ========================================================= */

async function testSharedOptions() {
  try {
    const years = await getAvailableYears();

    logger.info("Available years loaded successfully.", years);

    const maps = await getAvailableMaps(2024);

    logger.info("Available maps loaded successfully.", maps);
  } catch (error) {
    logger.error("Failed to load shared options.", error);
  }
}

/* =========================================================
   MODE
   ========================================================= */

function initModeTabs() {
  const tabs = document.querySelectorAll(".mode-tab");

  logger.debug(`Found ${tabs.length} mode tabs.`);

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const mode = tab.dataset.mode;

      logger.debug("Mode tab clicked.", { mode });

      if (mode === state.currentMode) {
        return;
      }

      state.currentMode = mode;

      updateActiveTab(tab);
      updateModeUI(mode);
      hideModeResults();

      logger.info(`Mode changed to: ${mode}`);
    });
  });
}

function updateActiveTab(activeTab) {
  document
    .querySelectorAll(".mode-tab")
    .forEach((tab) => tab.classList.remove("active"));

  activeTab.classList.add("active");
}

function updateModeUI(mode) {
  const teamField = document.getElementById("teamField");
  const teamSummary = document.getElementById("teamSummary");
  const actionButton = document.getElementById("btnAction");

  const isTeamMode = mode === "team";

  if (teamField) {
    teamField.style.display = isTeamMode ? "" : "none";
  }

  if (teamSummary) {
    teamSummary.style.display = isTeamMode ? "" : "none";
  }

  if (actionButton) {
    actionButton.textContent = isTeamMode ? "Prediksi" : "Analisis";
  }

  logger.debug("Mode UI updated.", {
    mode,
    teamVisible: isTeamMode,
  });
}

function hideModeResults() {
  hideResults("generalResult");
  hideResults("teamResult");
}

/* =========================================================
   SHARED INPUT
   ========================================================= */

function initSharedInputEvents() {
  const year = document.getElementById("year");
  const map = document.getElementById("map");
  const team = document.getElementById("team");
  const actionButton = document.getElementById("btnAction");
  const resetButton = document.getElementById("btnReset");

  logger.debug("Shared input elements.", {
    yearFound: Boolean(year),
    mapFound: Boolean(map),
    teamFound: Boolean(team),
    actionButtonFound: Boolean(actionButton),
    resetButtonFound: Boolean(resetButton),
  });

  year?.addEventListener("change", handleYearChange);
  map?.addEventListener("change", handleMapChange);

  actionButton?.addEventListener("click", handleAction);

  resetButton?.addEventListener("click", handleReset);
}

/* =========================================================
   YEAR
   ========================================================= */

async function handleYearChange(event) {
  const year = event.target.value;

  logger.debug("Year changed.", { year });

  resetMap();
  resetTeam();

  updateYearSummary(year);

  if (!year) {
    return;
  }

  try {
    const maps = await getAvailableMaps(year);

    logger.info("Maps loaded for selected year.", {
      year,
      maps,
    });

    populateSelect(document.getElementById("map"), maps, "— Pilih Map —");
  } catch (error) {
    logger.error("Failed to load maps.", error);
  }
}

/* =========================================================
   MAP
   ========================================================= */

function handleMapChange(event) {
  const map = event.target.value;

  logger.debug("Map changed.", { map });

  updateMapSummary(map);

  if (state.currentMode === "team") {
    resetTeam();

    if (!map) {
      return;
    }

    // Team options API akan dihubungkan pada tahap berikutnya.
    logger.debug("Team mode map selected.", { map });
  }
}

/* =========================================================
   ACTION
   ========================================================= */

function handleAction() {
  logger.info("Action button clicked.", {
    mode: state.currentMode,
  });

  if (state.currentMode === "general") {
    handleGeneralAction();
    return;
  }

  handleTeamAction();
}

function handleGeneralAction() {
  logger.info("General analysis requested.");

  // Logic analyzeGeneral() akan dihubungkan setelah
  // shared input controller selesai diuji.
}

function handleTeamAction() {
  logger.info("Team prediction requested.");

  // Logic predictTeam() akan dihubungkan setelah
  // shared input controller selesai diuji.
}

/* =========================================================
   RESET
   ========================================================= */

function handleReset() {
  logger.info("Shared reset clicked.");

  resetAgents();

  resetYear();
  resetMap();
  resetTeam();

  resetSummaries();

  hideModeResults();
}

/* =========================================================
   SELECT HELPERS
   ========================================================= */

function populateSelect(select, items, placeholder) {
  if (!select) {
    return;
  }

  select.innerHTML = "";

  const placeholderOption = document.createElement("option");

  placeholderOption.value = "";
  placeholderOption.textContent = placeholder;
  placeholderOption.hidden = true;

  select.appendChild(placeholderOption);

  items.forEach((item) => {
    const option = document.createElement("option");

    option.value = item;
    option.textContent = item;

    select.appendChild(option);
  });

  select.value = "";

  select.removeAttribute("disabled");
}

function resetYear() {
  const year = document.getElementById("year");

  if (!year) {
    return;
  }

  year.value = "";
}

function resetMap() {
  const map = document.getElementById("map");

  if (!map) {
    return;
  }

  map.innerHTML = '<option value="" hidden>— Pilih Map —</option>';
  map.value = "";
  map.setAttribute("disabled", "");
}

function resetTeam() {
  const team = document.getElementById("team");

  if (!team) {
    return;
  }

  team.innerHTML = '<option value="" hidden>— Pilih Tim —</option>';
  team.value = "";
  team.setAttribute("disabled", "");
}

/* =========================================================
   SUMMARY
   ========================================================= */

function updateYearSummary(year) {
  const element = document.getElementById("namaTahun");

  if (element) {
    element.textContent = year || "—";
  }
}

function updateMapSummary(map) {
  const element = document.getElementById("namaMap");

  if (element) {
    element.textContent = map || "—";
  }
}

function resetSummaries() {
  document.getElementById("namaTahun")?.replaceChildren("—");
  document.getElementById("namaMap")?.replaceChildren("—");
  document.getElementById("namaTeam")?.replaceChildren("—");
}

/* =========================================================
   ERROR
   ========================================================= */

function initErrorEvents() {
  const closeButton = document.getElementById("errorCloseBtn");

  logger.debug("Error close button.", {
    found: Boolean(closeButton),
  });

  closeButton?.addEventListener("click", closeError);
}

/* =========================================================
   START APPLICATION
   ========================================================= */

document.addEventListener("DOMContentLoaded", init);
