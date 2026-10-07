const PREFIX = "[Valorant Analyzer]";

const isDevelopment = import.meta.env.DEV;

export const logger = {
  info(message, data = null) {
    if (!isDevelopment) return;

    if (data !== null) {
      console.info(`${PREFIX} ${message}`, data);
      return;
    }

    console.info(`${PREFIX} ${message}`);
  },

  debug(message, data = null) {
    if (!isDevelopment) return;

    if (data !== null) {
      console.debug(`${PREFIX} ${message}`, data);
      return;
    }

    console.debug(`${PREFIX} ${message}`);
  },

  warn(message, data = null) {
    if (!isDevelopment) return;

    if (data !== null) {
      console.warn(`${PREFIX} ${message}`, data);
      return;
    }

    console.warn(`${PREFIX} ${message}`);
  },

  error(message, error = null) {
    if (!isDevelopment) return;

    if (error !== null) {
      console.error(`${PREFIX} ${message}`, error);
      return;
    }

    console.error(`${PREFIX} ${message}`);
  },
};
