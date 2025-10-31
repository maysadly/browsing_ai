/* global window, document, fetch, EventSource */

const state = {
  tasks: new Map(),
  currentTaskId: null,
  stream: null,
};

const form = document.querySelector("#task-form");
const taskList = document.querySelector("#tasks");
const logStream = document.querySelector("#log-stream");
const screenshotEl = document.querySelector("#screenshot");
const confirmPane = document.querySelector("#confirm-pane");
const confirmSummary = document.querySelector("#confirm-summary");
const confirmApprove = document.querySelector("#confirm-approve");
const confirmReject = document.querySelector("#confirm-reject");

async function loadTasks() {
  const response = await fetch("/api/tasks");
  if (!response.ok) {
    return;
  }
  const tasks = await response.json();
  tasks.forEach((task) => state.tasks.set(task.id, task));
  renderTasks();
}

async function createTask(goal, allowHighRisk) {
  const response = await fetch("/api/tasks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ goal, allow_high_risk: allowHighRisk }),
  });
  const payload = await response.json();
  if (response.ok && payload.id) {
    await refreshTask(payload.id);
    attachStream(payload.id);
  } else {
    appendLog(`Task creation failed: ${payload.error || "unknown error"}`);
  }
}

async function refreshTask(taskId) {
  const response = await fetch(`/api/tasks/${taskId}`);
  if (!response.ok) {
    return;
  }
  const task = await response.json();
  state.tasks.set(task.id, task);
  renderTasks();
}

function renderTasks() {
  const entries = [...state.tasks.values()].sort(
    (a, b) => new Date(b.created_at) - new Date(a.created_at),
  );
  taskList.innerHTML = "";
  entries.forEach((task) => {
    const item = document.createElement("li");
    item.dataset.taskId = task.id;
    item.innerHTML = `
      <strong>${task.goal}</strong>
      <span>Status: ${task.status}</span>
      ${task.result_summary ? `<p>${task.result_summary}</p>` : ""}
    `;
    item.addEventListener("click", () => attachStream(task.id));
    taskList.appendChild(item);
  });
}

function attachStream(taskId) {
  if (state.stream) {
    state.stream.close();
  }
  state.currentTaskId = taskId;
  logStream.innerHTML = "";
  confirmPane.classList.add("hidden");

  state.stream = new EventSource(`/api/tasks/${taskId}/events`);
  state.stream.onmessage = (event) => {
    if (!event.data) {
      return;
    }
    const payload = JSON.parse(event.data);
    handleEvent(taskId, payload);
  };
  state.stream.onerror = () => {
    appendLog("Event stream error, retrying soon...");
    state.stream.close();
    setTimeout(() => attachStream(taskId), 3000);
  };

  refreshTask(taskId);
}

function appendLog(text) {
  const entry = document.createElement("div");
  entry.className = "entry";
  entry.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
  logStream.appendChild(entry);
  logStream.scrollTop = logStream.scrollHeight;
}

function handleEvent(taskId, event) {
  const task = state.tasks.get(taskId) || { id: taskId };
  if (event.kind === "status") {
    task.status = event.payload.status;
    if (task.status === "Done" && task.result_summary) {
      appendLog(`Task finished: ${task.result_summary}`);
    } else {
      appendLog(`Status -> ${task.status}`);
    }
  }
  if (event.kind === "step" && event.payload.observation) {
    const obs = event.payload.observation;
    appendLog(`${event.payload.role}: ${event.payload.note || ""} | ${obs.note}`);
    if (obs.screenshot_path) {
      screenshotEl.src = obs.screenshot_path;
    }
  }
  if (event.kind === "summary" && event.payload.text) {
    task.result_summary = event.payload.text;
    appendLog(`Summary ready: ${event.payload.text}`);
  }
  if (event.kind === "confirm_request") {
    showConfirmation(taskId, event.payload);
  }
  state.tasks.set(taskId, task);
  renderTasks();
}

function showConfirmation(taskId, payload) {
  confirmPane.classList.remove("hidden");
  confirmSummary.textContent = payload.summary;
  if (payload.screenshot_path) {
    screenshotEl.src = payload.screenshot_path;
  }
  confirmApprove.onclick = () => sendConfirmation(taskId, true);
  confirmReject.onclick = () => sendConfirmation(taskId, false);
}

async function sendConfirmation(taskId, approve) {
  await fetch(`/api/tasks/${taskId}/confirm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ approve }),
  });
  confirmPane.classList.add("hidden");
  appendLog(approve ? "High risk action approved." : "High risk action rejected.");
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const goal = document.querySelector("#goal").value.trim();
  const allowHighRisk = document.querySelector("#allow-high-risk").checked;
  if (!goal) {
    appendLog("Enter a goal before submitting.");
    return;
  }
  createTask(goal, allowHighRisk);
  form.reset();
});

window.addEventListener("load", () => {
  loadTasks();
});
