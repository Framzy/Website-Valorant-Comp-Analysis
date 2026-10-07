import { predictTeam } from "./api.js";
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
      error.apiInfo?.message ||
        error.response?.data?.message ||
        error.response?.data?.error ||
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

  if (!year)
    return { valid: false, message: "Silakan pilih tahun terlebih dahulu." };
  if (!map)
    return { valid: false, message: "Silakan pilih map terlebih dahulu." };
  if (!team)
    return { valid: false, message: "Silakan pilih team terlebih dahulu." };
  if (!isAgentSelectionComplete())
    return { valid: false, message: "Silakan pilih 5 agent terlebih dahulu." };

  return { valid: true, year, map, team, agents: getSelectedAgents() };
}

function renderTeamResult(result) {
  logger.debug("Rendering team result.", result);

  renderTeamPrediction(result.prediction);
  renderTeamHistorical(result.historical);
  renderTeamComposition(result.composition);
  renderTeamSelectedAgents(result.input?.agents);
  renderTeamPredictionContext(result.prediction, result.confidence);
  renderTeamBestComposition(result.best_composition);
  renderTeamInference(result.inference);
}

function renderTeamPrediction(prediction) {
  const percentage = prediction?.percentage;

  if (typeof percentage !== "number") {
    logger.warn("Team prediction percentage is unavailable.");
    return;
  }

  setText("teamPredictionWinrate", formatPercent(percentage));

  const range = prediction?.range;
  if (typeof range?.low === "number" && typeof range?.high === "number") {
    setText(
      "teamPredictionRange",
      `${formatPercent(range.low * 100)} – ${formatPercent(range.high * 100)}`,
    );
  } else {
    setText("teamPredictionRange", "—");
  }

  const conclusion = assessTeamPrediction(prediction);
  updateTeamGauge(percentage, conclusion);
}

/**
 * Assess the prediction relative to the historical/context prior.
 *
 * The conclusion intentionally does not use the predicted percentage alone.
 * A prediction is considered:
 * - Layak Dicoba: final prediction is >= 5 percentage points above prior
 * - Ragu-Ragu: difference is between -5 and +5 points
 * - Kurang Direkomendasikan: final prediction is <= 5 points below prior
 *
 * Confidence remains supporting information and is not used as the primary
 * classification criterion.
 */
function assessTeamPrediction(prediction) {
  const percentage = prediction?.percentage;
  const prior = prediction?.prior_winrate;

  if (typeof percentage !== "number" || typeof prior !== "number") {
    return {
      key: "unknown",
      label: "—",
      className: "team-conclusion-unknown",
      reason:
        "Kesimpulan belum dapat ditentukan karena data pembanding tidak tersedia.",
    };
  }

  const difference = percentage - prior * 100;
  const threshold = 10;

  if (difference >= threshold) {
    return {
      key: "recommended",
      label: "Potensial",
      className: "team-conclusion-recommended",
      reason: `Prediksi ${formatPercent(percentage)} berada ${formatPercent(Math.abs(difference))} di atas prior ${formatPercent(prior * 100)}.`,
    };
  }

  if (difference <= -threshold) {
    return {
      key: "risk",
      label: "Beresiko",
      className: "team-conclusion-risk",
      reason: `Prediksi ${formatPercent(percentage)} berada ${formatPercent(Math.abs(difference))} di bawah prior ${formatPercent(prior * 100)}.`,
    };
  }

  return {
    key: "stable",
    label: "Stabil",
    className: "team-conclusion-stable",
    reason: `Prediksi ${formatPercent(percentage)} relatif dekat dengan prior ${formatPercent(prior * 100)} (${difference >= 0 ? "+" : "-"}${formatPercent(Math.abs(difference))}).`,
  };
}

function updateTeamGauge(percentage, conclusion) {
  const section = document.getElementById("teamGaugeSection");
  const fill = document.getElementById("teamGaugeFill");
  const status = document.getElementById("teamGaugeStatus");
  const reason = document.getElementById("teamConclusionReason");

  if (section) {
    section.classList.remove(
      "conclusion-recommended",
      "conclusion-stable",
      "conclusion-risk",
      "conclusion-unknown",
    );
    section.classList.add(`conclusion-${conclusion.key}`);
  }

  if (status) {
    status.className = `gauge-status conclusion-${conclusion.key}`;
    status.textContent = conclusion.label;
  }

  // Keep the gauge center concise. The detailed explanation belongs below.
  setText("teamGaugeSubText", `Predicted ${formatPercent(percentage)}`);

  if (reason) {
    reason.textContent = conclusion.reason;
  }

  if (fill) {
    fill.classList.remove(
      "conclusion-recommended",
      "conclusion-stable",
      "conclusion-risk",
      "conclusion-unknown",
    );

    fill.classList.add(`conclusion-${conclusion.key}`);

    const gaugePosition = {
      risk: 30,
      stable: 60,
      recommended: 85,
      unknown: 50,
    };

    const clamped = gaugePosition[conclusion.key] ?? 50;

    const pathLength = fill.getTotalLength();

    fill.style.strokeDasharray = pathLength;
    fill.style.strokeDashoffset = pathLength - (clamped / 100) * pathLength;
  }
}

function renderTeamHistorical(historical) {
  const element = document.getElementById("teamHistoricalCompositionWinrate");
  if (!element) return;

  if (!historical?.found || historical.winrate === null) {
    element.textContent = "Belum ada data";
  } else {
    element.textContent = formatPercent(historical.winrate * 100);
  }

  const strength = historical?.composition_strength;
  setText(
    "teamCompositionStrength",
    typeof strength === "number" ? strength.toFixed(2) : "Belum ada data",
  );
}

function renderTeamComposition(composition) {
  const rolePattern = document.getElementById("teamRolePattern");

  if (!composition) return;

  renderTeamPlaystyle(
    "teamPlaystyleBadge",
    "teamPlaystyleDescription",
    composition.playstyle,
  );

  const pattern = composition?.role_pattern || "—";
  if (rolePattern) {
    renderRolePattern(rolePattern, pattern);
  }
}

function renderTeamSelectedAgents(agentNames) {
  renderAgentVisuals("teamSelectedAgents", agentNames);
}

function renderTeamPredictionContext(prediction, confidence) {
  const adjustment = prediction?.pattern_adjustment;

  if (adjustment && typeof adjustment.points === "number") {
    const points =
      adjustment.points > 0 ? `+${adjustment.points}` : adjustment.points;
    setText("teamPatternAdjustment", `${points} points`);
  } else {
    setText("teamPatternAdjustment", "—");
  }

  setText(
    "teamModelWinrate",
    typeof prediction?.model_winrate === "number"
      ? formatPercent(prediction.model_winrate * 100)
      : "—",
  );

  setText(
    "teamPriorWinrate",
    typeof prediction?.prior_winrate === "number"
      ? formatPercent(prediction.prior_winrate * 100)
      : "—",
  );

  setText(
    "teamConfidenceNote",
    confidence?.note || adjustment?.reason || "Tidak ada konteks tambahan.",
  );

  setText("teamConfidenceLevel", formatConfidence(confidence?.level));
}

function renderTeamBestComposition(bestComposition) {
  const section = document.getElementById("teamHistoricalSection");
  const rolePattern = document.getElementById("teamHistoricalRolePattern");

  if (!bestComposition?.found) {
    if (section) section.classList.add("no-data");
    renderAgentVisuals("teamHistoricalAgents", []);
    renderTeamPlaystyle(
      "teamHistoricalPlaystyleBadge",
      "teamHistoricalPlaystyleDescription",
      null,
    );
    setText("teamHistoricalMaps", "Belum ada data");
    setText("teamHistoricalCompositionWinrate", "—");
    setText("teamHistoricalCompositionStrength", "—");
    return;
  }

  if (section) section.classList.remove("no-data");

  const pattern = bestComposition?.role_pattern || "—";

  renderAgentVisuals("teamHistoricalAgents", bestComposition.agents);
  renderTeamPlaystyle(
    "teamHistoricalPlaystyleBadge",
    "teamHistoricalPlaystyleDescription",
    bestComposition.playstyle,
  );

  if (rolePattern) {
    renderRolePattern(rolePattern, pattern);
  }

  setText(
    "teamHistoricalMaps",
    typeof bestComposition.maps_played === "number"
      ? `${bestComposition.maps_played} Maps`
      : "—",
  );
  setText(
    "teamHistoricalCompositionWinrate",
    typeof bestComposition.winrate === "number"
      ? `${(bestComposition.winrate * 100).toFixed(2)}% WR`
      : "—",
  );
  setText(
    "teamHistoricalCompositionStrength",
    typeof bestComposition.composition_strength === "number"
      ? `Strength ${bestComposition.composition_strength.toFixed(2)}`
      : "—",
  );
}

function renderTeamPlaystyle(badgeId, descriptionId, playstyle) {
  const $badge = $(`#${badgeId}`);
  const $description = $(`#${descriptionId}`);

  if (!playstyle?.name) {
    $badge
      .text("—")
      .attr(
        "class",
        "playstyle-badge team-playstyle-badge playstyle-unclassified",
      );
    $description.text("Belum ada informasi playstyle.");
    return;
  }

  const name = String(playstyle.name).toUpperCase();
  const className = `playstyle-${name.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;

  $badge
    .text(name)
    .attr("class", `playstyle-badge team-playstyle-badge ${className}`);

  $description.text(
    playstyle.description || "Tidak ada deskripsi playstyle yang tersedia.",
  );
}

function renderAgentVisuals(containerId, agentNames = []) {
  const $container = $(`#${containerId}`);
  if (!$container.length) return;

  $container.empty();

  agentNames.forEach((name) => {
    const agent = agentData.find(
      (item) => item.name.toLowerCase() === String(name).toLowerCase(),
    );

    if (!agent) {
      logger.warn("Agent visual data not found.", name);
      return;
    }

    const wrapper = document.createElement("div");
    wrapper.className = "team-agent-visual";
    wrapper.title = capitalize(agent.name);

    const image = document.createElement("img");
    image.src = agent.url;
    image.alt = capitalize(agent.name);
    image.loading = "lazy";

    const label = document.createElement("span");
    label.textContent = capitalize(agent.name);

    wrapper.append(image, label);
    $container.append(wrapper);
  });
}

function renderTeamInference(inference) {
  const $element = $("#teamFeatureCount");
  if (!$element.length) return;

  $element.text(
    inference?.feature_count
      ? `${inference.feature_count} features digunakan model`
      : "—",
  );
}

function setText(id, value) {
  $(`#${id}`).text(value);
}

function formatPercent(value) {
  return `${Number(value).toFixed(2)}%`;
}

function formatConfidence(level) {
  if (!level) return "—";

  const normalized = String(level).toLowerCase();

  const labels = {
    high: "HIGH",
    medium: "MEDIUM",
    low: "LOW",
  };

  return labels[normalized] || String(level).toUpperCase();
}

function capitalize(value) {
  if (!value) return "";
  return value.charAt(0).toUpperCase() + value.slice(1);
}
