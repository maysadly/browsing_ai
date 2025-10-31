from __future__ import annotations

import atexit
import json
import signal
from typing import Dict, Optional, TYPE_CHECKING

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
    stream_with_context,
)

if TYPE_CHECKING:  # pragma: no cover
    from autobrowser.core.config import Settings

try:
    from autobrowser.core.config import get_settings
except ModuleNotFoundError:  # pragma: no cover
    def get_settings():  # type: ignore
        raise RuntimeError("Settings module unavailable; ensure pydantic-settings is installed")
from autobrowser.core.logger import configure_logging
from autobrowser.core.orchestrator import TaskOrchestrator
from autobrowser.core.schemas import Task, TaskStatus


def create_app(settings: Optional["Settings"] = None) -> Flask:
    settings = settings or get_settings()
    configure_logging(settings)

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["JSON_SORT_KEYS"] = False

    orchestrator = TaskOrchestrator(settings)

    @app.route("/", methods=["GET"])
    def index() -> str:
        return render_template(
            "index.html",
            config={
                "model": settings.anthropic_model,
                "tokenBudget": settings.token_budget,
                "maxSteps": settings.max_steps,
            },
        )

    @app.route("/api/config", methods=["GET"])
    def api_config():
        payload = {
            "model": settings.anthropic_model,
            "embedding_model": settings.embedding_model,
            "token_budget": settings.token_budget,
            "max_steps": settings.max_steps,
            "max_concurrent_tasks": settings.max_concurrent_tasks,
        }
        return jsonify(payload)

    @app.route("/api/tasks", methods=["POST"])
    def create_task_endpoint():
        body = request.get_json(force=True)
        goal = body.get("goal", "").strip()
        allow_high_risk = bool(body.get("allow_high_risk", False))
        if not goal:
            return jsonify({"error": "Goal is required."}), 400

        task = orchestrator.create_task(goal, allow_high_risk=allow_high_risk)
        return jsonify({"id": task.id})

    @app.route("/api/tasks", methods=["GET"])
    def list_tasks_endpoint():
        tasks = [ _serialize_task(task) for task in orchestrator.list_tasks() ]
        return jsonify(tasks)

    @app.route("/api/tasks/<task_id>", methods=["GET"])
    def get_task(task_id: str):
        task = orchestrator.get_task(task_id)
        if not task:
            return jsonify({"error": "Task not found."}), 404
        return jsonify(_serialize_task(task))

    @app.route("/api/tasks/<task_id>/events", methods=["GET"])
    def task_events(task_id: str):
        def event_stream():
            for event in orchestrator.stream_events(task_id):
                payload = event.model_dump(mode="json")
                yield f"data: {json.dumps(payload)}\n\n"

        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        return Response(
            stream_with_context(event_stream()),
            mimetype="text/event-stream",
            headers=headers,
        )

    @app.route("/api/tasks/<task_id>/confirm", methods=["POST"])
    def confirm_action(task_id: str):
        body = request.get_json(force=True)
        approve = bool(body.get("approve"))
        ok = orchestrator.confirm_task_action(task_id, approve)
        return jsonify({"ack": ok})

    @app.route("/artifacts/<path:filename>", methods=["GET"])
    def artifacts(filename: str):
        return send_from_directory(settings.artifacts_dir, filename)

    @app.after_request
    def add_cors_headers(response: Response):
        response.headers["Access-Control-Allow-Origin"] = request.headers.get(
            "Origin", "*"
        )
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        if request.method == "OPTIONS":
            response.status_code = 200
            response.headers["Content-Length"] = "0"
        return response

    def handle_exit(*_: int) -> None:
        orchestrator.shutdown()

    signal.signal(signal.SIGTERM, handle_exit)
    signal.signal(signal.SIGINT, handle_exit)
    atexit.register(orchestrator.shutdown)

    return app


def _serialize_task(task: Task) -> Dict[str, object]:
    return {
        "id": task.id,
        "goal": task.goal,
        "status": task.status.value,
        "created_at": task.created_at.isoformat(),
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "allow_high_risk": task.allow_high_risk,
        "risk_level": task.risk_level.value,
        "result_summary": task.result_summary,
        "last_error": task.last_error,
        "logs": [event.model_dump() for event in task.logs[-50:]],
    }


app = create_app()
