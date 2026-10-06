"""Video model providers behind one interface.

ReplicateProvider calls a hosted text-to-video model. MockProvider renders real mp4
files with ffmpeg and injects seeded failures and defects, so the harness (retries,
evals, failover, resume) can be exercised for free and deterministically in tests
and in demo mode.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from . import media
from .registry import ToolFailed, TransientError
from .settings import env, env_float, parse_model

DEFECTS = ("black", "frozen", "off_palette", "short")


@dataclass
class GenRequest:
    prompt: str
    duration: float
    width: int
    height: int
    palette: list[str]
    seed: int
    shot_id: str
    noise_key: str = ""  # ties the mock's failure draws to the brief, not the run id
    attempt: int = 0     # tool-level retry index
    image_path: str | None = None  # start the clip from this still, when the model can


class Provider(Protocol):
    name: str
    price_per_s: float
    simulated: bool

    def generate(self, req: GenRequest, out_path: str) -> None: ...


def _draw(*parts) -> float:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


@dataclass
class MockProvider:
    name: str
    price_per_s: float
    p_error: float = 0.0
    p_defect: float = 0.0
    force_defect: str | None = None  # tests use this to produce one specific defect
    delay_s: float = 0.0             # demo mode: pretend generation takes a moment
    simulated: bool = True

    def generate(self, req: GenRequest, out_path: str) -> None:
        if self.delay_s:
            time.sleep(self.delay_s)
        if _draw(self.name, req.noise_key, req.shot_id, req.seed, "err", req.attempt) < self.p_error:
            raise TransientError(f"{self.name}: 503 upstream timeout")
        defect = self.force_defect
        if defect is None and _draw(self.name, req.noise_key, req.shot_id, req.seed, "defect") < self.p_defect:
            defect = DEFECTS[int(_draw(self.name, req.noise_key, req.shot_id, req.seed, "kind") * len(DEFECTS))]
        self._render(req, out_path, defect)

    def _render(self, req: GenRequest, out_path: str, defect: str | None) -> None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        size, dur = f"{req.width}x{req.height}", req.duration
        c0, c1 = (c.lstrip("#") for c in req.palette[:2])
        if defect == "short":
            dur = req.duration * 0.4
        if defect == "off_palette":  # style drift: complementary colours
            c0, c1 = ("%06x" % (0xFFFFFF - int(c, 16)) for c in (c0, c1))
        if defect == "black":
            src = f"color=c=black:s={size}:r=12:d={dur}"
        elif defect == "frozen":
            r0, r1 = media.hex_rgb(c0), media.hex_rgb(c1)
            mid = "%02x%02x%02x" % tuple((a + b) // 2 for a, b in zip(r0, r1))
            src = f"color=c=0x{mid}:s={size}:r=12:d={dur}"
        else:
            # seeded and with fixed endpoints, so the same request always renders the same pixels
            seed = int(_draw(self.name, req.noise_key, req.shot_id, req.seed, "pix") * 2 ** 31)
            src = (f"gradients=s={size}:r=12:d={dur}:c0=0x{c0}:c1=0x{c1}:nb_colors=2:speed=0.2:"
                   f"seed={seed}:x0=0:y0=0:x1={req.width}:y1={req.height}")
        args = ["-f", "lavfi", "-i", src]
        if os.path.exists(media.FONT) and defect not in ("black", "frozen"):
            label = out_path + ".txt"
            with open(label, "w") as f:
                f.write(f"{req.shot_id}: {req.prompt[:60]}")
            args += ["-vf", f"drawtext=fontfile={media.FONT}:textfile={label}:fontcolor=white:"
                            f"fontsize={max(10, req.height // 16)}:x=8:y=h-th-8:box=1:boxcolor=black@0.35"]
        args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", out_path]
        media.ffmpeg(args)
        if os.path.exists(out_path + ".txt"):
            os.remove(out_path + ".txt")


class ReplicateClient:
    """Runs one model on Replicate and downloads its output file.

    Models name their inputs differently, so the client reads the model's published
    input schema once; callers build inputs from what that model actually accepts.
    """

    API = "https://api.replicate.com/v1"

    def __init__(self, model: str, name: str, token: str | None = None, transport=None,
                 poll_s: float = 3.0, timeout_s: float = 600):
        self.model, self.name = model, name
        self.token = token if token is not None else os.environ.get("REPLICATE_API_TOKEN", "")
        self.transport, self.poll_s, self.timeout_s = transport, poll_s, timeout_s
        self._schema: dict | None = None

    def _input_schema(self, http: httpx.Client) -> dict:
        if self._schema is None:
            self._schema = {}
            try:
                r = http.get(f"{self.API}/models/{self.model}")
                if r.status_code == 200:
                    version = r.json().get("latest_version") or {}
                    schemas = version.get("openapi_schema", {}).get("components", {}).get("schemas", {})

                    def resolve(spec: dict) -> dict:
                        # allowed values often live in a referenced schema, not on the property itself
                        out = dict(spec)
                        refs = [spec.get("$ref")] + [s.get("$ref") for s in spec.get("allOf", []) + spec.get("anyOf", [])]
                        for ref in filter(None, refs):
                            out.update(schemas.get(ref.rsplit("/", 1)[-1], {}))
                        return out

                    self._schema = {k: resolve(v) for k, v in schemas.get("Input", {}).get("properties", {}).items()}
            except httpx.TransportError:
                pass  # fall back to prompt-only inputs
        return self._schema

    def run(self, build_inputs, out_path: str) -> dict:
        """build_inputs(schema) -> inputs. Returns the inputs that were accepted."""
        if not self.token:
            raise ToolFailed("REPLICATE_API_TOKEN is not set")
        try:
            with httpx.Client(timeout=60, transport=self.transport, follow_redirects=True,
                              headers={"Authorization": f"Bearer {self.token}"}) as http:
                inputs = build_inputs(self._input_schema(http))
                url = f"{self.API}/models/{self.model}/predictions"
                r = http.post(url, json={"input": inputs})
                if r.status_code == 422:
                    # The model refused some input values. Drop exactly those fields so the model
                    # uses its own defaults, and try once more.
                    try:
                        refused = {f.get("field", "").split(".", 1)[-1] for f in r.json().get("invalid_fields", [])}
                    except ValueError:
                        refused = set()
                    refused = (refused & inputs.keys()) - {"prompt"}
                    if refused:
                        inputs = {k: v for k, v in inputs.items() if k not in refused}
                        r = http.post(url, json={"input": inputs})
                pred = self._ok(r)
                deadline = time.time() + self.timeout_s
                while pred["status"] not in ("succeeded", "failed", "canceled"):
                    if time.time() > deadline:
                        raise TransientError(f"{self.name}: generation timed out")
                    time.sleep(self.poll_s)
                    pred = self._ok(http.get(pred["urls"]["get"]))
                if pred["status"] != "succeeded":
                    raise TransientError(f"{self.name}: generation {pred['status']}: {pred.get('error')}")
                output = pred.get("output")
                url = output[0] if isinstance(output, list) else output
                if not isinstance(url, str) or not url.startswith("http"):
                    raise ToolFailed(f"{self.name}: unexpected output {json.dumps(output)[:120]}")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                # the file host is not Replicate's API: do not send it the API token
                with httpx.Client(timeout=120, transport=self.transport, follow_redirects=True) as files:
                    dl = files.get(url)
                    if dl.status_code != 200:
                        raise TransientError(f"{self.name}: download HTTP {dl.status_code}")
                    with open(out_path, "wb") as f:
                        f.write(dl.content)
                return inputs
        except httpx.TransportError as e:
            raise TransientError(f"{self.name}: {e}") from e

    def _ok(self, r: httpx.Response) -> dict:
        if r.status_code in (408, 429) or r.status_code >= 500:
            try:
                wait = float(r.headers.get("retry-after", ""))
            except ValueError:
                wait = None
            if r.status_code == 429 and wait is None:
                wait = 10.0  # rate limited without a hint: a short pause is the minimum useful wait
            raise TransientError(f"{self.name}: HTTP {r.status_code}: {r.text[:160]}", retry_after=wait)
        if r.status_code >= 400:
            raise ToolFailed(f"{self.name}: HTTP {r.status_code}: {r.text[:200]}")
        return r.json()


def _offer(inputs: dict, schema: dict, key: str, value) -> None:
    """Set an input only if this model has it and allows the value."""
    spec = schema.get(key)
    if spec is None:
        return
    allowed = spec.get("enum")
    if allowed and value not in allowed:
        return
    inputs[key] = value


def _attach_image(inputs: dict, schema: dict, names: tuple[str, ...], image_path: str) -> bool:
    """Put a still into whichever image input this model has. Small JPEGs travel inline."""
    field = next((n for n in names if n in schema), None)
    if field is None:
        return False
    with open(image_path, "rb") as f:
        uri = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
    inputs[field] = [uri] if schema[field].get("type") == "array" else uri
    return True


VIDEO_IMAGE_FIELDS = ("image", "first_frame_image", "start_image", "first_frame", "input_image")
STILL_IMAGE_FIELDS = ("input_image", "image_input", "image", "reference_image", "reference_images", "input_images")


class ReplicateProvider:
    """A video model hosted on Replicate: text-to-video, or image-to-video when the
    model has an image input and the request carries a start image."""

    simulated = False

    def __init__(self, model: str, price_per_s: float, name: str | None = None, token: str | None = None,
                 extra_inputs: dict | None = None, transport=None, poll_s: float = 3.0, timeout_s: float = 600):
        self.model, self.price_per_s = model, price_per_s
        self.name = name or model.split("/")[-1]
        self.extra_inputs = extra_inputs or {}
        self.client = ReplicateClient(model, self.name, token, transport, poll_s, timeout_s)

    def build_inputs(self, req: GenRequest, schema: dict) -> dict:
        inputs: dict = {"prompt": req.prompt}
        _offer(inputs, schema, "seed", req.seed)
        _offer(inputs, schema, "aspect_ratio", "16:9")
        _offer(inputs, schema, "resolution", f"{req.height}p")
        spec = schema.get("duration")
        if spec is not None:
            want = int(-(-req.duration // 1))  # ceil: generate at least as long as the shot, trim later
            allowed = spec.get("enum")
            if allowed:
                longer = [d for d in allowed if d >= want]
                inputs["duration"] = min(longer) if longer else max(allowed)
            else:
                inputs["duration"] = max(spec.get("minimum", want), min(spec.get("maximum", want), want))
        if req.image_path:  # a text-only model simply ignores the start image
            _attach_image(inputs, schema, VIDEO_IMAGE_FIELDS, req.image_path)
        inputs.update(self.extra_inputs)
        return inputs

    def generate(self, req: GenRequest, out_path: str) -> None:
        self.client.run(lambda schema: self.build_inputs(req, schema), out_path)


# ---------------------------------------------------------------------------
# Still-image models: the subject reference and per-shot keyframes
# ---------------------------------------------------------------------------
class ReplicateImageModel:
    """A still-image model on Replicate. With `image_path` it must accept a reference image."""

    simulated = False

    def __init__(self, model: str, price_per_image: float, name: str | None = None, token: str | None = None,
                 transport=None, poll_s: float = 2.0, timeout_s: float = 300):
        self.model, self.price_per_image = model, price_per_image
        self.name = name or model.split("/")[-1]
        self.client = ReplicateClient(model, self.name, token, transport, poll_s, timeout_s)

    def generate(self, prompt: str, seed: int, image_path: str | None, out_path: str) -> None:
        def build(schema: dict) -> dict:
            inputs: dict = {"prompt": prompt}
            _offer(inputs, schema, "seed", seed)
            _offer(inputs, schema, "aspect_ratio", "16:9")
            _offer(inputs, schema, "output_format", "jpg")
            if image_path and not _attach_image(inputs, schema, STILL_IMAGE_FIELDS, image_path):
                raise ToolFailed(f"{self.name}: this model has no reference-image input "
                                 f"(looked for {', '.join(STILL_IMAGE_FIELDS)})")
            return inputs

        self.client.run(build, out_path)


@dataclass
class MockImageModel:
    """Renders a flat still so the reference and keyframe flow can be tested for free."""

    name: str = "mock-image"
    price_per_image: float = 0.01
    fail: bool = False
    simulated: bool = True

    def generate(self, prompt: str, seed: int, image_path: str | None, out_path: str) -> None:
        if self.fail:
            raise TransientError(f"{self.name}: 503 upstream timeout")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        colour = "%06x" % int(_draw(self.name, prompt, seed, bool(image_path)) * 0xFFFFFF)
        media.ffmpeg(["-f", "lavfi", "-i", f"color=c=0x{colour}:s=640x360:d=1", "-frames:v", "1", "-f", "image2", "-c:v", "mjpeg", out_path])


def default_image_models() -> dict:
    """The still-image models, each `owner/name=price_per_image`:

    DIRECTORCUT_IMAGE_MODEL     text-to-image, draws the subject reference
    DIRECTORCUT_KEYFRAME_MODEL  image editing, places the referenced subject in each shot

    Setting DIRECTORCUT_IMAGE_MODEL turns on the reference workflow. Without the keyframe
    model every shot starts from the reference image itself.
    """
    models = {"reference": None, "keyframe": None}
    if not os.environ.get("REPLICATE_API_TOKEN"):
        if env("DEMO_REFERENCE"):  # try the workflow in demo mode with flat placeholder stills
            models = {"reference": MockImageModel("demo-reference"), "keyframe": MockImageModel("demo-keyframe")}
    else:
        for role, var in (("reference", "IMAGE_MODEL"), ("keyframe", "KEYFRAME_MODEL")):
            parsed = parse_model(env(var), 0.04)
            if parsed:
                models[role] = ReplicateImageModel(*parsed)
    return models


DEFAULT_MODELS = "wan-video/wan-2.2-t2v-fast=0.02,wan-video/wan-2.5-t2v-fast=0.06"


def demo_providers(delay_s: float = 0.0) -> dict[str, Provider]:
    return {
        "fastgen": MockProvider("fastgen", price_per_s=0.04, p_error=0.15, p_defect=0.25, delay_s=delay_s),
        "proline": MockProvider("proline", price_per_s=0.12, p_error=0.03, p_defect=0.04, delay_s=delay_s),
    }


def default_providers() -> dict[str, Provider]:
    """Real models when REPLICATE_API_TOKEN is set, otherwise the simulated pair.

    DIRECTORCUT_VIDEO_MODELS is a comma-separated list of owner/name=price_per_second, in
    order of preference. The price is your estimate from the model's page and is what
    budgets are counted against.
    """
    if not os.environ.get("REPLICATE_API_TOKEN"):
        return demo_providers(env_float("DEMO_DELAY", 0.0))
    providers: dict[str, Provider] = {}
    for item in env("VIDEO_MODELS", DEFAULT_MODELS).split(","):
        parsed = parse_model(item, 0.05)
        if parsed:
            p = ReplicateProvider(*parsed)
            providers[p.name] = p
    return providers
