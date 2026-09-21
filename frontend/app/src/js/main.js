import "../css/base.css";
import "../css/components.css";
import "../css/results.css";
import "../css/responsive.css";

import { state } from "./state.js";
import { initAgents, resetAgents } from "./agents.js";
import { closeError, hideResults } from "./ui.js";
import { logger } from "./logger.js";

function init() {
  logger.info("Application initialization started.");

  initModeTabs();
  logger.debug("Mode tabs initialized.");

  initGeneralEvents();
  logger.debug("General events initialized.");

  initTeamEvents();
  logger.debug("Team events initialized.");

  initErrorEvents();
  logger.debug("Error events initialized.");

  initAgents();
  logger.debug("Agent system initialized.");

  logger.info("Application initialization completed.", {
    mode: state.currentMode,
    selectedAgents: state.selectedAgents,
    maxAgents: state.maxAgents,
  });
}

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
      updateModePanels(mode);
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

function updateModePanels(mode) {
  const generalPanel = document.getElementById("formGeneral");
  const teamPanel = document.getElementById("formTeam");

  logger.debug("Updating mode panels.", {
    mode,
    generalPanelFound: Boolean(generalPanel),
    teamPanelFound: Boolean(teamPanel),
  });

  generalPanel?.classList.toggle("active", mode === "general");
  teamPanel?.classList.toggle("active", mode === "team");
}

function hideModeResults() {
  hideResults("generalResult");
  hideResults("teamResult");
}

function initGeneralEvents() {
  const resetButton = document.getElementById("btnResetGeneral");

  logger.debug("General reset button.", {
    found: Boolean(resetButton),
  });

  resetButton?.addEventListener("click", () => {
    logger.info("General reset clicked.");

    resetAgents();

    const year = document.getElementById("yearGeneral");
    const map = document.getElementById("mapGeneral");

    if (year) {
      year.value = "";
    }

    if (map) {
      map.value = "";
    }

    document.getElementById("namaTahunGeneral")?.replaceChildren("—");
    document.getElementById("namaMapGeneral")?.replaceChildren("—");

    hideResults("generalResult");
  });
}

function initTeamEvents() {
  const resetButton = document.getElementById("btnResetTeam");

  logger.debug("Team reset button.", {
    found: Boolean(resetButton),
  });

  resetButton?.addEventListener("click", () => {
    logger.info("Team reset clicked.");

    resetAgents();

    const year = document.getElementById("yearTeam");
    const map = document.getElementById("mapTeam");
    const team = document.getElementById("team");

    if (year) {
      year.value = "";
    }

    if (map) {
      map.value = "";
    }

    if (team) {
      team.value = "";
    }

    map?.setAttribute("disabled", "");
    team?.setAttribute("disabled", "");

    document.getElementById("namaTahunTeam")?.replaceChildren("—");
    document.getElementById("namaMapTeam")?.replaceChildren("—");
    document.getElementById("namaTeam")?.replaceChildren("—");

    hideResults("teamResult");
  });
}

function initErrorEvents() {
  const closeButton = document.getElementById("errorCloseBtn");

  logger.debug("Error close button.", {
    found: Boolean(closeButton),
  });

  closeButton?.addEventListener("click", closeError);
}

document.addEventListener("DOMContentLoaded", init);
