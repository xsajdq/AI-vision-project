from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import history
from .audio.inspector import AudioInspector
from .config import settings
from .schemas import (
    AudioInspectionResponse,
    HealthResponse,
    HistoryEntry,
    InspectionKind,
    VisionInspectionResponse,
)
from .vision.inspector import VisionInspector

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
SAMPLES_DIR = ROOT_DIR / "data" / "samples"
MAX_UPLOAD_BYTES = settings.max_upload_mb * 1024 * 1024

app = FastAPI(title=settings.app_name)

_vision_inspector: VisionInspector | None = None
_audio_inspector: AudioInspector | None = None
_vision_load_error: str | None = None
_audio_load_error: str | None = None

try:
    _vision_inspector = VisionInspector()
except FileNotFoundError as exc:
    _vision_load_error = str(exc)

try:
    _audio_inspector = AudioInspector()
except FileNotFoundError as exc:
    _audio_load_error = str(exc)


async def _read_upload(upload: UploadFile) -> bytes:
    data = await upload.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB limit")
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    return data


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        vision_model_loaded=_vision_inspector is not None,
        audio_model_loaded=_audio_inspector is not None,
    )


@app.post("/api/inspect/image", response_model=VisionInspectionResponse)
async def inspect_image(file: UploadFile = File(...)) -> VisionInspectionResponse:
    if _vision_inspector is None:
        raise HTTPException(status_code=503, detail=_vision_load_error)

    data = await _read_upload(file)
    try:
        result = _vision_inspector.inspect(data)
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as a 400
        raise HTTPException(status_code=400, detail=f"Could not process image: {exc}") from exc

    verdict = "DEFECT DETECTED" if result.is_defective else "OK"
    history.record(InspectionKind.VISION, file.filename or "upload", verdict, result.anomaly_score, result.threshold)

    return VisionInspectionResponse(
        filename=file.filename or "upload",
        anomaly_score=result.anomaly_score,
        threshold=result.threshold,
        is_defective=result.is_defective,
        verdict=verdict,
        heatmap_png_base64=result.heatmap_png_base64,
    )


@app.post("/api/inspect/audio", response_model=AudioInspectionResponse)
async def inspect_audio(file: UploadFile = File(...)) -> AudioInspectionResponse:
    if _audio_inspector is None:
        raise HTTPException(status_code=503, detail=_audio_load_error)

    data = await _read_upload(file)
    try:
        result = _audio_inspector.inspect(data)
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as a 400
        raise HTTPException(status_code=400, detail=f"Could not process audio: {exc}") from exc

    verdict = "ANOMALY DETECTED" if result.is_anomalous else "NORMAL"
    history.record(InspectionKind.AUDIO, file.filename or "upload", verdict, result.anomaly_score, result.threshold)

    return AudioInspectionResponse(
        filename=file.filename or "upload",
        anomaly_score=result.anomaly_score,
        threshold=result.threshold,
        is_anomalous=result.is_anomalous,
        verdict=verdict,
        spectrogram_png_base64=result.spectrogram_png_base64,
    )


@app.get("/api/history", response_model=list[HistoryEntry])
def get_history() -> list[HistoryEntry]:
    return history.list_recent()


@app.get("/api/samples")
def list_samples() -> dict:
    if not SAMPLES_DIR.exists():
        return {"vision": {"normal": [], "defective": []}, "audio": {"normal": [], "anomalous": []}}

    def listing(*parts: str) -> list[str]:
        directory = SAMPLES_DIR.joinpath(*parts)
        if not directory.exists():
            return []
        return sorted(f"/samples/{'/'.join(parts)}/{p.name}" for p in directory.iterdir() if p.is_file())

    return {
        "vision": {"normal": listing("vision", "normal"), "defective": listing("vision", "defective")},
        "audio": {"normal": listing("audio", "normal"), "anomalous": listing("audio", "anomalous")},
    }


if SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")
