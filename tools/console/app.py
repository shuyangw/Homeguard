"""FastAPI app: the page, the htmx panel fragments, and a Host check (127.0.0.1 only)."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.trustedhost import TrustedHostMiddleware

from tools.console import views

HERE = Path(__file__).resolve().parent


def create_app(poller, region: str, clock: Callable[[], datetime], start_polling: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(poller.run()) if start_polling else None
        yield
        if task is not None:
            task.cancel()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    # Defeats DNS rebinding: a page on another origin cannot reach the app under its own hostname.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")

    def render(request: Request, template: str, context: dict) -> HTMLResponse:
        return templates.TemplateResponse(request, template, context)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return render(request, "index.html", views.page_context(poller.state, clock(), region))

    @app.get("/panels/{panel}", response_class=HTMLResponse)
    def panel(request: Request, panel: str) -> HTMLResponse:
        if panel not in views.PANELS:
            raise HTTPException(status_code=404)
        return render(request, f"panels/{panel}.html", views.page_context(poller.state, clock(), region))

    @app.get("/panels/gates/{strategy}", response_class=HTMLResponse)
    def gates(request: Request, strategy: str) -> HTMLResponse:
        context = views.gates_context(poller.state, strategy)
        if context is None:
            raise HTTPException(status_code=404)
        return render(request, "panels/gates.html", context)

    return app
