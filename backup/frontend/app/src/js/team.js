import { predictTeam } from "./api.js";
import { getSelectedAgents, isAgentSelectionComplete } from "./agents.js";
import {
  hideLoading,
  showError,
  showLoading,
  showResults,
  revealResults,
} from "./ui.js";
import { logger } from "./logger.js";

export async function handleTeamAction() {
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

  try {
    showLoading("Menganalisis komposisi team...");

    const result = await predictTeam(payload);

    logger.info("Team prediction completed.", result);

    renderTeamResult(result);

    showResults("teamResult");
    revealResults();
  } catch (error) {
    logger.error("Team prediction failed.", error);

    showError(
      error.response?.data?.message ||
        "Gagal melakukan prediksi. Silakan coba lagi.",
    );
  } finally {
    hideLoading();
  }
}

function validateTeamInputs() {
  const year = document.getElementById("year")?.value;
  const map = document.getElementById("map")?.value;
  const team = document.getElementById("team")?.value;

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

  if (!team) {
    return {
      valid: false,
      message: "Silakan pilih team terlebih dahulu.",
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
    team,
    agents: getSelectedAgents(),
  };
}

function renderTeamResult(result) {
  logger.debug("Rendering team result.", result);

  renderTeamPrediction(result.prediction);
  renderTeamHistorical(result.historical);
  renderTeamComposition(result.composition);
  renderTeamBestComposition(result.best_composition);
  renderTeamInference(result.inference);
}

function renderTeamPrediction(prediction) {
  const percentage = prediction?.percentage;

  if (typeof percentage !== "number") {
    logger.warn("Team prediction percentage is unavailable.");
    return;
  }

  const element = document.getElementById("teamPredictionWinrate");

  if (element) {
    element.textContent = `${percentage.toFixed(2)}%`;
  }

  updateTeamGauge(percentage);
}

function updateTeamGauge(percentage) {
  const fill = document.getElementById("teamGaugeFill");
  const status = document.getElementById("teamGaugeStatus");
  const subText = document.getElementById("teamGaugeSubText");

  if (status) {
    status.textContent = `${percentage.toFixed(2)}%`;
  }

  if (subText) {
    subText.textContent = "Predicted Winrate";
  }

  if (!fill) {
    return;
  }

  const clamped = Math.max(0, Math.min(100, percentage));

  const pathLength = fill.getTotalLength();

  fill.style.strokeDasharray = pathLength;
  fill.style.strokeDashoffset = pathLength - (clamped / 100) * pathLength;
}

function renderTeamHistorical(historical) {
  const element = document.getElementById("teamHistoricalWinrate");

  if (!element) {
    return;
  }

  if (!historical?.found || historical.winrate === null) {
    element.textContent = "Belum ada data";
    return;
  }

  element.textContent = `${(historical.winrate * 100).toFixed(2)}%`;
}

function renderTeamComposition(composition) {
  if (!composition) {
    return;
  }

  const rolePattern = document.getElementById("teamRolePattern");

  if (rolePattern) {
    rolePattern.textContent = composition.role_pattern || "—";
  }

  const roleComposition = document.getElementById("teamRoleComposition");

  if (roleComposition) {
    roleComposition.textContent =
      `Duelist ${composition.duelist_count} • ` +
      `Initiator ${composition.initiator_count} • ` +
      `Controller ${composition.controller_count} • ` +
      `Sentinel ${composition.sentinel_count}`;
  }
}

function renderTeamBestComposition(bestComposition) {
  const element = document.getElementById("teamHistoricalStatus");

  if (!element) {
    return;
  }

  if (!bestComposition?.found) {
    element.textContent = "Belum ada historical composition.";
    return;
  }

  const agents = bestComposition.agents?.map(capitalize).join(" • ");

  element.textContent =
    `${agents} | ` +
    `${bestComposition.role_pattern} | ` +
    `${bestComposition.maps_played} Maps | ` +
    `${(bestComposition.winrate * 100).toFixed(2)}% WR`;
}

function renderTeamInference(inference) {
  const element = document.getElementById("teamFeatureCount");

  if (!element) {
    return;
  }

  element.textContent = inference?.feature_count
    ? `${inference.feature_count} features digunakan model`
    : "—";
}

function capitalize(value) {
  if (!value) {
    return "";
  }

  return value.charAt(0).toUpperCase() + value.slice(1);
}
