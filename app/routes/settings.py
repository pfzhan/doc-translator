"""Routes of the application settings."""

from fastapi import APIRouter, Body, HTTPException

from .. import state
from ..job_control import _cleanup, _dispatch
from ..settings import SettingsError

router = APIRouter()

# ---------- Settings ----------


@router.get("/api/settings")
def get_settings():
    return state.settings.public()


@router.put("/api/settings")
async def update_settings(data: dict = Body(...)):
    try:
        result = state.settings.update(data)
    except SettingsError as e:
        raise HTTPException(400, str(e)) from e
    _dispatch()  # A higher limit starts queued jobs at once
    await _cleanup()  # The retention period may have been shortened; clean up at once
    return result
