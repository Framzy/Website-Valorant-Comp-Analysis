import axios from "axios";
import { logger } from "./logger.js";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
    Accept: "application/json",
  },
});

// =========================================================
// REQUEST INTERCEPTOR
// =========================================================

api.interceptors.request.use(
  (config) => {
    logger.debug("API request.", {
      method: config.method?.toUpperCase(),
      url: `${config.baseURL}${config.url}`,
      params: config.params,
      data: config.data,
    });

    return config;
  },
  (error) => {
    logger.error("API request setup failed.", error);
    return Promise.reject(error);
  },
);

// =========================================================
// RESPONSE INTERCEPTOR
// =========================================================

api.interceptors.response.use(
  (response) => {
    logger.debug("API response.", {
      status: response.status,
      url: response.config.url,
      data: response.data,
    });

    return response;
  },
  (error) => {
    logger.error("API request failed.", {
      status: error.response?.status,
      url: error.config?.url,
      data: error.response?.data,
      message: error.message,
    });

    return Promise.reject(error);
  },
);

// =========================================================
// SHARED OPTIONS
// =========================================================

export async function getAvailableYears() {
  const response = await api.get("/api/options/years");

  return response.data.years;
}

export async function getAvailableMaps(year) {
  const response = await api.get("/api/options/maps", {
    params: {
      year,
    },
  });

  return response.data.maps;
}

// =========================================================
// TEAM OPTIONS
// =========================================================

export async function getTeamTeams(year, map) {
  const response = await api.get("/api/team/options/teams", {
    params: {
      year,
      map,
    },
  });

  return response.data.teams;
}

// =========================================================
// GENERAL ANALYSIS
// =========================================================

export async function analyzeGeneral(payload) {
  const response = await api.post("/api/general/analyze", payload);

  return response.data;
}

// =========================================================
// TEAM PREDICTION
// =========================================================

export async function predictTeam(payload) {
  const response = await api.post("/api/team/predict", payload);

  return response.data;
}
