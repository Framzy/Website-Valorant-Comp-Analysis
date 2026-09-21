import { agents } from "../data/agent.js";
import { state } from "./state.js";
import { logger } from "./logger.js";

export function initAgents() {
  logger.info("Initializing agent system.");

  state.agents = agents;

  logger.debug("Agent data loaded.", {
    count: state.agents.length,
  });

  renderAgents();
  initRoleFilter();

  updateAgentCounter();
  updateSelectedAgentSummary();

  logger.info("Agent system initialized.");
}

function renderAgents() {
  const container = document.getElementById("Agent");

  logger.debug("Looking for agent container.", {
    found: Boolean(container),
  });

  if (!container) {
    throw new Error("Element #Agent tidak ditemukan.");
  }

  container.innerHTML = "";

  state.agents.forEach((agent) => {
    const element = createAgentElement(agent);
    container.appendChild(element);
  });

  logger.debug("Agent cards rendered.", {
    count: state.agents.length,
  });
}

function createAgentElement(agent) {
  const wrapper = document.createElement("div");

  wrapper.classList.add("agent-item");

  wrapper.dataset.name = agent.name;
  wrapper.dataset.role = agent.role;

  wrapper.style.backgroundImage = `url("${agent.url}")`;

  // Role badge
  const badge = document.createElement("span");

  badge.classList.add("role-badge", agent.role);

  badge.textContent = capitalize(agent.role);

  wrapper.appendChild(badge);

  // Agent name
  const name = document.createElement("h1");

  name.textContent = capitalize(agent.name);

  wrapper.appendChild(name);

  // Selection
  wrapper.addEventListener("click", () => {
    toggleAgent(agent.name, wrapper);
  });

  return wrapper;
}

function toggleAgent(agentName, element) {
  const index = state.selectedAgents.indexOf(agentName);

  // Deselect
  if (index !== -1) {
    state.selectedAgents.splice(index, 1);

    element.classList.remove("selected");

    updateAgentCounter();
    updateSelectedAgentSummary();

    return;
  }

  // Maximum agent
  if (state.selectedAgents.length >= state.maxAgents) {
    const removedAgent = state.selectedAgents.shift();

    const removedElement = document.querySelector(
      `.agent-item[data-name="${removedAgent}"]`,
    );

    removedElement?.classList.remove("selected");
  }

  state.selectedAgents.push(agentName);

  element.classList.add("selected");

  updateAgentCounter();
  updateSelectedAgentSummary();
}

function updateAgentCounter() {
  const counter = document.getElementById("agentCounter");

  if (!counter) {
    return;
  }

  const selected = state.selectedAgents.length;
  const max = state.maxAgents;

  counter.textContent = `${selected} / ${max} Agent Dipilih`;

  counter.classList.toggle("complete", selected === max);
}

function updateSelectedAgentSummary() {
  const text = state.selectedAgents.length
    ? state.selectedAgents.map(capitalize).join(", ")
    : "—";

  const general = document.getElementById("namaAgentGeneral");

  if (general) {
    general.textContent = text;
  }

  const team = document.getElementById("namaAgentTeam");

  if (team) {
    team.textContent = text;
  }
}

function initRoleFilter() {
  const tabs = document.querySelectorAll(".role-tab");

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const role = tab.dataset.role;

      setActiveRoleTab(tab);
      filterAgents(role);
    });
  });
}

function setActiveRoleTab(activeTab) {
  document
    .querySelectorAll(".role-tab")
    .forEach((tab) => tab.classList.remove("active"));

  activeTab.classList.add("active");
}

function filterAgents(role) {
  document.querySelectorAll(".agent-item").forEach((agent) => {
    const isMatch = role === "all" || agent.dataset.role === role;

    agent.classList.toggle("role-hidden", !isMatch);
  });
}

export function resetAgents() {
  state.selectedAgents = [];

  document
    .querySelectorAll(".agent-item.selected")
    .forEach((agent) => agent.classList.remove("selected"));

  resetRoleFilter();

  updateAgentCounter();
  updateSelectedAgentSummary();
}

function resetRoleFilter() {
  const tabs = document.querySelectorAll(".role-tab");

  tabs.forEach((tab) => tab.classList.remove("active"));

  const allTab = document.querySelector('.role-tab[data-role="all"]');

  allTab?.classList.add("active");

  document
    .querySelectorAll(".agent-item")
    .forEach((agent) => agent.classList.remove("role-hidden"));
}

export function getSelectedAgents() {
  return [...state.selectedAgents];
}

export function isAgentSelectionComplete() {
  return state.selectedAgents.length === state.maxAgents;
}

function capitalize(value) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
