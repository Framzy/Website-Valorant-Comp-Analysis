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

// ========================================================
// API ERROR HANDLING
// ========================================================

function getApiErrorInfo(error) {
  if (!error.response) {
    if (error.code === "ECONNABORTED") {
      return {
        type: "timeout",
        status: null,
        message: "Request membutuhkan waktu terlalu lama.",
      };
    }

    return {
      type: "network",
      status: null,
      message: "Tidak dapat terhubung ke server.",
    };
  }

  const status = error.response.status;

  switch (status) {
    case 400:
      return {
        type: "http",
        status,
        message: "Permintaan tidak valid.",
      };

    case 404:
      return {
        type: "http",
        status,
        message: "Data atau endpoint tidak ditemukan.",
      };

    case 408:
      return {
        type: "http",
        status,
        message: "Request membutuhkan waktu terlalu lama.",
      };

    case 429:
      return {
        type: "http",
        status,
        message: "Terlalu banyak permintaan. Silakan coba lagi nanti.",
      };

    case 500:
      return {
        type: "http",
        status,
        message: "Terjadi kesalahan pada server.",
      };

    case 502:
    case 503:
    case 504:
      return {
        type: "http",
        status,
        message: "Server sedang tidak tersedia. Silakan coba lagi nanti.",
      };

    default:
      return {
        type: "http",
        status,
        message: `Terjadi kesalahan pada server (${status}).`,
      };
  }
}

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
    const apiInfo = getApiErrorInfo(error);

    logger.error("API request failed.", {
      status: apiInfo.status,
      type: apiInfo.type,
      url: error.config?.url,
      data: error.response?.data,
      message: apiInfo.message,
      axiosMessage: error.message,
    });

    error.apiInfo = apiInfo;

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
