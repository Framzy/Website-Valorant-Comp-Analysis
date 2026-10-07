import { analyzeGeneral } from "./api.js";
import { getSelectedAgents, isAgentSelectionComplete } from "./agents.js";
import { agents as agentData } from "../data/agent.js";
import {
  hideLoading,
  showError,
  showLoading,
  showResults,
  revealResults,
  renderRolePattern,
} from "./ui.js";
import { logger } from "./logger.js";
import $ from "jquery";

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
      error.apiInfo?.message ||
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
  renderGeneralPlaystyle(result.playstyle);
  renderGeneralHistorical(result.historical);
  renderGeneralFallback(result);
  renderGeneralRecommendations(result.recommendations);
}

function renderGeneralSummary(result) {
  const $header = $("#generalResult .result-header");
  const $element = $("#generalResultSummary");

  $element.text(
    `${result?.input?.map || "—"} • ` +
      `${result?.input?.year || "—"} • 5 Agents`,
  );

  if (!$header.length) {
    return;
  }

  let $agentVisual = $header.find(".general-result-agents");

  if (!$agentVisual.length) {
    $agentVisual = $("<div>", {
      class: "general-result-agents reveal-item",
    });

    $header.append($agentVisual);
  }

  renderAgentImages($agentVisual, result?.input?.agents);
}

function renderAgentImages(container, agentNames) {
  const $container = $(container);

  if (!$container.length) {
    return;
  }

  $container.empty();

  if (!Array.isArray(agentNames) || agentNames.length === 0) {
    return;
  }

  agentNames.forEach((name) => {
    const agent = findAgent(name);

    const item = document.createElement("div");
    item.className = "general-agent-visual";

    if (agent?.url) {
      const image = document.createElement("img");
      image.src = agent.url;
      image.alt = capitalize(name);
      image.loading = "lazy";
      image.referrerPolicy = "no-referrer";
      item.appendChild(image);
    }

    const label = document.createElement("span");
    label.textContent = capitalize(name);
    item.appendChild(label);

    $container.append(item);
  });
}

function renderGeneralHistorical(historical) {
  const $pickRate = $("#generalPickRate");
  const $winrate = $("#generalWinrate");
  const $totalMaps = $("#generalTotalMaps");

  const found = Boolean(historical?.found);

  $pickRate.text(
    found && typeof historical?.pick_rate === "number"
      ? `${(historical.pick_rate * 100).toFixed(2)}%`
      : "—",
  );

  $winrate.text(
    found && typeof historical?.winrate === "number"
      ? `${(historical.winrate * 100).toFixed(2)}%`
      : "—",
  );

  $totalMaps.text(
    found && historical?.total_maps !== undefined ? historical.total_maps : "—",
  );

  const statCards = document.querySelectorAll(
    "#generalResult .stat-cards .stat-card",
  );

  statCards.forEach((card) => {
    card.classList.toggle("no-data", !found);
  });
}

function renderGeneralPlaystyle(playstyle) {
  const playstyleElement = document.getElementById("generalPlaystyle");
  const rolePattern = document.getElementById("generalRolePattern");

  const name = playstyle?.name || "UNCLASSIFIED";
  const pattern = playstyle?.pattern || "—";
  const description = playstyle?.description || "";

  if (playstyleElement) {
    playstyleElement.textContent = name;
    playstyleElement.className = `detail-value playstyle-badge playstyle-${normalizeClassName(name)}`;
  }

  if (rolePattern) {
    renderRolePattern(rolePattern, pattern);
  }

  const playstyleCard = playstyleElement?.closest(".detail-card");

  if (playstyleCard) {
    let descriptionElement = playstyleCard.querySelector(
      ".playstyle-description",
    );

    if (!descriptionElement) {
      descriptionElement = document.createElement("p");
      descriptionElement.className = "playstyle-description";
      playstyleCard.appendChild(descriptionElement);
    }

    descriptionElement.textContent = description;
    descriptionElement.hidden = !description;
  }
}

function renderGeneralFallback(result) {
  const $header = $("#generalResult .result-header");

  if (!$header.length) {
    return;
  }

  let $fallback = $header.find(".general-fallback");

  if (!result?.fallback) {
    $fallback.remove();
    return;
  }

  if (!$fallback.length) {
    $fallback = $("<div>", {
      class: "general-fallback reveal-item",
    });

    $header.append($fallback);
  }

  $fallback.html(`
    <span class="general-fallback-badge">Historical composition tidak ditemukan</span>
    <span class="general-fallback-text">
      Rekomendasi menggunakan data ${formatFallbackSource(result.fallback_source)}.
    </span>
  `);
}

function renderGeneralRecommendations(recommendations) {
  const $container = $("#recommendationsList");

  if (!$container.length) {
    return;
  }

  $container.empty();

  if (!Array.isArray(recommendations) || recommendations.length === 0) {
    $container.html(`
      <p class="combo-value">
        Belum ada historical recommendation.
      </p>
    `);
    return;
  }

  recommendations.slice(0, 3).forEach((item, index) => {
    const card = document.createElement("div");

    card.className = "popular-comp-card";

    const rank = document.createElement("div");
    rank.className = "popular-comp-rank";
    rank.textContent = item.rank || index + 1;

    const info = document.createElement("div");
    info.className = "popular-comp-info";

    const agentVisuals = document.createElement("div");
    agentVisuals.className = "recommendation-agents";

    renderAgentImages(agentVisuals, item.agents);

    const playstyle = document.createElement("span");
    const playstyleName = item.playstyle || "UNCLASSIFIED";
    playstyle.className = `recommendation-playstyle playstyle-${normalizeClassName(playstyleName)}`;
    playstyle.textContent = playstyleName;

    const meta = document.createElement("div");
    meta.className = "popular-comp-meta";
    meta.textContent = buildRecommendationMeta(item);

    info.appendChild(agentVisuals);
    info.appendChild(playstyle);
    info.appendChild(meta);

    const winrate = document.createElement("div");
    winrate.className = "popular-comp-wr";
    winrate.textContent =
      typeof item.winrate === "number"
        ? `${(item.winrate * 100).toFixed(2)}% WR`
        : "—";

    card.appendChild(rank);
    card.appendChild(info);
    card.appendChild(winrate);

    $container.append(card);
  });
}

function buildRecommendationMeta(item) {
  const parts = [item.role_pattern || "—", `${item.total_maps ?? 0} Maps`];

  if (typeof item.pick_rate === "number") {
    parts.push(`${(item.pick_rate * 100).toFixed(2)}% Pick`);
  }

  return parts.join(" | ");
}

function findAgent(name) {
  const normalized = String(name || "").toLowerCase();

  return agentData.find((agent) => agent.name.toLowerCase() === normalized);
}

function formatFallbackSource(source) {
  switch (source) {
    case "MAP_YEAR":
      return "konteks map dan tahun yang dipilih";
    case "MAP":
      return "konteks map yang dipilih";
    case "GLOBAL":
      return "data global";
    default:
      return "data fallback";
  }
}

function normalizeClassName(value) {
  return String(value || "unclassified")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-");
}

function capitalize(value) {
  if (!value) {
    return "";
  }

  return value.charAt(0).toUpperCase() + value.slice(1);
}
