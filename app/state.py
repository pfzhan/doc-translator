"""State shared by the web server's modules: the stores, the running jobs and the model downloads. Modules read it
as state.jobs and so on, so replacing a value here (as the tests do) is seen everywhere."""

import asyncio

from . import history, layout_model, quantize
from .glossaries import GlossaryStore
from .history import Job, JobStore
from .services import ServiceStore
from .settings import Settings

store = ServiceStore()

glossaries = GlossaryStore()

settings = Settings()

jobs = JobStore(history.JOBS_DIR)

# Running translation jobs; keeps them from being garbage collected and is used to cancel on pause or delete
running: dict[str, asyncio.Task] = {}

# Jobs waiting for a free place (settings.max_jobs), in the order they start
waiting: dict[str, Job] = {}

# Requests generating the other version. Deletion and expiry cleanup wait until they finish writing; a second
# request is rejected.
variant_tasks: dict[str, asyncio.Future[None]] = {}

# Jobs being cleaned: those expiry cleanup has decided to delete and not finished deleting, or those whose cache
# the user asked to clear and not finished clearing.
# Generating the other version and resuming translation are rejected for them, so nothing writes during cleanup.
cleaning: set[str] = set()

# Jobs the user paused: on cancellation their status is recorded as "已暂停" instead of "已中断"
pause_requested: set[str] = set()

# Layout model downloads (one at a time)
model_downloads = {kind: layout_model.Downloader(spec) for kind, spec in layout_model.MODELS.items()}

# int8 versions of both models are generated locally (see quantize.py), one at a time
quantizers = {kind: quantize.Quantizer() for kind in layout_model.MODELS}

ACTIVE = ("queued", "running")
