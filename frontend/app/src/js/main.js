import "../css/base.css";
import "../css/components.css";
import "../css/results.css";
import "../css/responsive.css";

import { state } from "./state.js";

import {
  initAgents,
  resetAgents,
  getSelectedAgents,
  isAgentSelectionComplete,
} from "./agents.js";

import { closeError, hideResults, showError } from "./ui.js";

import {
  getAvailableYears,
  getAvailableMaps,
  getTeamTeams,
  analyzeGeneral,
  predictTeam,
} from "./api.js";

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

  loadAvailableYears();

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

async function loadAvailableYears() {
  try {
    const years = await getAvailableYears();

    logger.info("Available years loaded.", years);

    populateSelect(document.getElementById("year"), years, "— Pilih Tahun —");
  } catch (error) {
    logger.error("Failed to load available years.", error);
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
      handleReset();

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
  team?.addEventListener("change", handleTeamChange);

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
  updateMapSummary("");
  updateTeamSummary("");

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

async function handleMapChange(event) {
  const map = event.target.value;
  const year = document.getElementById("year")?.value;

  logger.debug("Map changed.", {
    year,
    map,
    mode: state.currentMode,
  });

  updateMapSummary(map);

  resetTeam();
  updateTeamSummary("");

  if (state.currentMode !== "team" || !year || !map) {
    return;
  }

  try {
    const teams = await getTeamTeams(year, map);

    logger.info("Teams loaded.", {
      year,
      map,
      teams,
    });

    populateSelect(document.getElementById("team"), teams, "— Pilih Tim —");
  } catch (error) {
    logger.error("Failed to load teams.", error);
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

function handleTeamChange(event) {
  const team = event.target.value;

  logger.debug("Team changed.", { team });

  updateTeamSummary(team);
}

async function handleGeneralAction() {
  logger.info("General analysis requested.");

  const validation = validateCommonInputs();

  if (!validation.valid) {
    logger.warn("General validation failed.", validation.message);
    showError(validation.message);
    return;
  }

  const payload = {
    map: validation.map,
    year: Number(validation.year),
    agents: validation.agents,
  };

  logger.info("Sending general analysis request.", payload);

  try {
    const result = await analyzeGeneral(payload);

    logger.info("General analysis completed.", result);

    console.log("GENERAL API RESPONSE:", result);
  } catch (error) {
    logger.error("General analysis failed.", error);

    showError(
      error.response?.data?.message ||
        "Gagal melakukan analisis. Silakan coba lagi.",
    );
  }
}

async function handleTeamAction() {
  logger.info("Team prediction requested.");

  const validation = validateTeamInputs();

  if (!validation.valid) {
    logger.warn("Team validation failed.", validation.message);
    showError(validation.message);
    return;
  }

  const payload = {
    team: validation.team,
    map: validation.map,
    year: Number(validation.year),
    agents: validation.agents,
  };

  logger.info("Sending team prediction request.", payload);

  try {
    const result = await predictTeam(payload);

    logger.info("Team prediction completed.", result);

    console.log("TEAM API RESPONSE:", result);
  } catch (error) {
    logger.error("Team prediction failed.", error);

    showError(
      error.response?.data?.message ||
        "Gagal melakukan prediksi. Silakan coba lagi.",
    );
  }
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

function updateTeamSummary(team) {
  const element = document.getElementById("namaTeam");

  if (element) {
    element.textContent = team || "—";
  }
}

function resetSummaries() {
  document.getElementById("namaTahun")?.replaceChildren("—");
  document.getElementById("namaMap")?.replaceChildren("—");
  document.getElementById("namaTeam")?.replaceChildren("—");
}

/* =========================================================
   VALIDATION
   ========================================================= */

function validateCommonInputs() {
  const year = document.getElementById("year")?.value;
  const map = document.getElementById("map")?.value;

  if (!year) {
    return {
      valid: false,
      message: "Silakan pilih tahun terlebih dahulu.",
    };
  }

  if (!map) {
    return {
      valid: false,
      message: "Silakan pilih map terlebih dahulu.",
    };
  }

  if (!isAgentSelectionComplete()) {
    return {
      valid: false,
      message: "Silakan pilih 5 agent terlebih dahulu.",
    };
  }

  return {
    valid: true,
    year,
    map,
    agents: getSelectedAgents(),
  };
}

function validateTeamInputs() {
  const common = validateCommonInputs();

  if (!common.valid) {
    return common;
  }

  const team = document.getElementById("team")?.value;

  if (!team) {
    return {
      valid: false,
      message: "Silakan pilih team terlebih dahulu.",
    };
  }

  return {
    ...common,
    team,
  };
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
