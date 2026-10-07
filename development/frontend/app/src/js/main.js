import "../css/base.css";
import "../css/components.css";
import "../css/results.css";
import "../css/responsive.css";

import { state } from "./state.js";

import { initAgents, resetAgents } from "./agents.js";

import { closeError, hideResults, showError } from "./ui.js";

import { getAvailableYears, getAvailableMaps, getTeamTeams } from "./api.js";

import { handleGeneralAction } from "./general.js";

import { handleTeamAction } from "./team.js";

import { logger } from "./logger.js";

import $ from "jquery";

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
    showError(
      error.apiInfo?.message + " Silakan tunggu beberapa saat" ||
        "Gagal memuat data tahun. Silakan coba lagi.",
    );
  }
}

/* =========================================================
   MODE
   ========================================================= */

function initModeTabs() {
  const $tabs = $(".mode-tab");

  logger.debug(`Found ${$tabs.length} mode tabs.`);

  $tabs.on("click", function () {
    const mode = $(this).data("mode");

    logger.debug("Mode tab clicked.", { mode });

    if (mode === state.currentMode) {
      return;
    }

    state.currentMode = mode;

    updateActiveTab(this);
    updateModeUI(mode);
    hideModeResults();
    handleReset();

    logger.info(`Mode changed to: ${mode}`);
  });
}

function updateActiveTab(activeTab) {
  $(".mode-tab").removeClass("active");
  $(activeTab).addClass("active");
}

function updateModeUI(mode) {
  const isTeamMode = mode === "team";

  $("#teamField").toggle(isTeamMode);
  $("#teamSummary").toggle(isTeamMode);

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
  $("#year").on("change", handleYearChange);
  $("#map").on("change", handleMapChange);
  $("#team").on("change", handleTeamChange);

  $("#btnAction").on("click", handleAction);
  $("#btnReset").on("click", handleReset);

  logger.debug("Shared input elements.", {
    yearFound: Boolean($("#year").length),
    mapFound: Boolean($("#map").length),
    teamFound: Boolean($("#team").length),
    actionButtonFound: Boolean($("#btnAction").length),
    resetButtonFound: Boolean($("#btnReset").length),
  });
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
    showError(
      error.apiInfo?.message + " Silakan tunggu beberapa saat" ||
        "Gagal memuat data map. Silakan coba lagi.",
    );
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

    showError(
      error.apiInfo?.message + " Silakan tunggu beberapa saat" ||
        "Gagal memuat data tim. Silakan coba lagi.",
    );
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

  const $select = $(select);

  $select.empty();
  $select.append(
    $("<option>", {
      value: "",
      text: placeholder,
      hidden: true,
    }),
  );

  items.forEach((item) => {
    $select.append(
      $("<option>", {
        value: item,
        text: item,
      }),
    );
  });

  $select.val("").prop("disabled", false);
}

function resetYear() {
  $("#year").val("");
}

function resetMap() {
  $("#map").val("").attr("disabled", true);
}

function resetTeam() {
  $("#team").val("").attr("disabled", true);
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
