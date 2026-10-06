export function showLoading(message = "Memproses data...") {
  const overlay = document.getElementById("overlay");
  const loading = document.getElementById("loading");
  const messageElement = document.getElementById("loadingMessage");

  if (messageElement) {
    messageElement.textContent = message;
  }

  if (overlay) {
    overlay.style.display = "block";
  }

  if (loading) {
    loading.style.display = "flex";
  }
}

export function hideLoading() {
  const overlay = document.getElementById("overlay");
  const loading = document.getElementById("loading");

  if (overlay) {
    overlay.style.display = "none";
  }

  if (loading) {
    loading.style.display = "none";
  }
}

export function showError(message) {
  const messageElement = document.getElementById("errorMessage");
  const overlay = document.getElementById("errorOverlay");
  const popup = document.getElementById("errorPopup");

  if (messageElement) {
    messageElement.textContent = message;
  }

  if (overlay) {
    overlay.style.display = "block";
  }

  if (popup) {
    popup.style.display = "flex";

    requestAnimationFrame(() => {
      popup.classList.add("show");
    });
  }
}

export function closeError() {
  const overlay = document.getElementById("errorOverlay");
  const popup = document.getElementById("errorPopup");

  if (!popup) {
    return;
  }

  popup.classList.remove("show");

  setTimeout(() => {
    if (overlay) {
      overlay.style.display = "none";
    }

    popup.style.display = "none";
  }, 250);
}

export function showResults(elementId) {
  document.getElementById(elementId)?.classList.add("active", "visible");
}

export function hideResults(elementId) {
  document.getElementById(elementId)?.classList.remove("active", "visible");
}

export function revealResults() {
  const items = document.querySelectorAll(".reveal-item");

  items.forEach((item, index) => {
    item.classList.remove("revealed");

    setTimeout(() => {
      item.classList.add("revealed");
    }, 150 * index);
  });
}
