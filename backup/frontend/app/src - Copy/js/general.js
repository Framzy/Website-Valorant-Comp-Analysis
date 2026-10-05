import { analyzeGeneral } from "./api.js";
import { getSelectedAgents, isAgentSelectionComplete } from "./agents.js";
import {
  hideLoading,
  showError,
  showLoading,
  showResults,
  revealResults,
} from "./ui.js";
import { logger } from "./logger.js";

export async function handleGeneralAction() {
  logger.info("General analysis requested.");

  const validation = validateGeneralInputs();

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
    showLoading("Menganalisis komposisi...");

    const result = await analyzeGeneral(payload);

    logger.info("General analysis completed.", result);

    renderGeneralResult(result);

    showResults("generalResult");
    revealResults();
  } catch (error) {
    logger.error("General analysis failed.", error);

    showError(
      error.response?.data?.message ||
        error.response?.data?.error ||
        "Gagal melakukan analisis. Silakan coba lagi.",
    );
  } finally {
    hideLoading();
  }
}

function validateGeneralInputs() {
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

function renderGeneralResult(result) {
  logger.debug("Rendering general result.", result);

  renderGeneralSummary(result);
  renderGeneralHistorical(result.historical);
  renderGeneralPlaystyle(result.playstyle);
  renderGeneralRecommendations(result.recommendations);
}

function renderGeneralSummary(result) {
  const element = document.getElementById("generalResultSummary");

  if (!element) {
    return;
  }

  element.textContent =
    `${result?.input?.map || "—"} • ` +
    `${result?.input?.year || "—"} • 5 Agents`;
}

function renderGeneralHistorical(historical) {
  const pickRate = document.getElementById("generalPickRate");
  const winrate = document.getElementById("generalWinrate");
  const totalMaps = document.getElementById("generalTotalMaps");

  if (pickRate) {
    pickRate.textContent =
      typeof historical?.pick_rate === "number"
        ? `${(historical.pick_rate * 100).toFixed(2)}%`
        : "—";
  }

  if (winrate) {
    winrate.textContent =
      typeof historical?.winrate === "number"
        ? `${(historical.winrate * 100).toFixed(2)}%`
        : "—";
  }

  if (totalMaps) {
    totalMaps.textContent =
      historical?.total_maps !== undefined ? historical.total_maps : "—";
  }
}

function renderGeneralPlaystyle(playstyle) {
  const playstyleElement = document.getElementById("generalPlaystyle");

  const rolePattern = document.getElementById("generalRolePattern");

  if (playstyleElement) {
    playstyleElement.textContent = playstyle?.name || "—";
  }

  if (rolePattern) {
    rolePattern.textContent = playstyle?.pattern || "—";
  }
}

function renderGeneralRecommendations(recommendations) {
  const container = document.getElementById("recommendationsList");

  if (!container) {
    return;
  }

  container.innerHTML = "";

  if (!Array.isArray(recommendations) || recommendations.length === 0) {
    container.innerHTML = `
      <p class="combo-value">
        Belum ada historical recommendation.
      </p>
    `;

    return;
  }

  recommendations.slice(0, 3).forEach((item, index) => {
    const card = document.createElement("div");

    card.className = "popular-comp-card";

    const agents = Array.isArray(item.agents)
      ? item.agents.map(capitalize).join(" • ")
      : "—";

    const pickRate =
      typeof item.pick_rate === "number"
        ? `${(item.pick_rate * 100).toFixed(2)}% pick`
        : "";

    const winrate =
      typeof item.winrate === "number"
        ? `${(item.winrate * 100).toFixed(2)}% WR`
        : "";

    card.innerHTML = `
      <div class="popular-comp-rank">
        ${item.rank || index + 1}
      </div>

      <div class="popular-comp-info">
        <div class="popular-comp-agents">
          ${agents}
        </div>

        <div class="popular-comp-meta">
          ${item.role_pattern || "—"} •
          ${item.total_maps ?? 0} Maps •
          ${pickRate}
        </div>
      </div>

      <div class="popular-comp-wr">
        ${winrate || "—"}
      </div>
    `;

    container.appendChild(card);
  });
}

function capitalize(value) {
  if (!value) {
    return "";
  }

  return value.charAt(0).toUpperCase() + value.slice(1);
}
