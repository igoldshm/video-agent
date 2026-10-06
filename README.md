# DirectorCut AI

A director agent for AI video creation.

Describe a video in a sentence or two and get a finished short video back. An agent
plans the shots, generates each one with a text-to-video model, reviews the result,
retakes what is not good enough, cuts the shots together and waits for you to approve
before publishing.

The app is a FastAPI service with a small web front end. The interesting part is the
harness between the creator and the models.

## How a video gets made

```
brief ─ plan ─ choose models ─ pick assets ─┬─ shot 1: generate ─ check ─ judge ─┐
        (LLM)   (eval history)              ├─ shot 2: generate ─ check ─ judge ─┤─ assemble ─ final check ─ publish
                                            └─ ...        ▲ retake / fail over ◄─┘                (needs approval)
```

- **Planning and creative context.** An LLM turns the brief into a title, a recurring
  subject, a look and a shot list. Subject, look and the creator's standing notes go
  into every generation prompt and to the judge, so shot 4 belongs to the same video as shot 1.
- **Tools.** Each capability is a typed tool with the permission scope it needs
  (`directorcut/tools.py`, `directorcut/registry.py`). Transient provider errors are retried
  with backoff; a model that keeps failing is left for the next one.
- **State.** Every tool call is checkpointed in SQLite under a stable key such as
  `gen:s2:r1:a0` (shot 2, revision 1, attempt 0). Resuming a video replays what is
  recorded and only runs, and only pays for, what is new. A restarted service picks up
  unfinished videos by itself.
- **Consistent subjects (optional).** With an image model configured, the agent first
  draws a reference image of the recurring subject and stops until the creator approves
  it (`reference:approve`). Each shot then gets its own opening frame, made by an
  image-editing model from that reference, and the clip is generated from that frame
  with an image-to-video model. The opening frame is reviewed before any video is
  generated from it, because a still costs a fraction of a clip. No shot is filmed from
  the reference itself.
- **Evals.** Two layers. Technical checks measure the file: decodable, right length,
  no black frames, has motion. Then a vision model looks at three frames, plus a
  reference frame from the first accepted shot, and scores whether the clip shows the
  shot, looks clean and matches the rest. A rejection carries issue tags that become
  corrective instructions in the next take.
- **Learning.** Eval results are kept per model. Once there is enough history, models
  are tried in order of expected cost per accepted clip, and an issue a model keeps
  being rejected for becomes a standing instruction in every prompt sent to it
  (`directorcut/learning.py`). A creator's change request can be remembered and applied to
  their later videos.
- **Permissions and limits.** Publishing needs the `deliver:publish` scope, granted
  when the creator presses Publish; changing a shot withdraws it. Each video has a
  spend budget that is checked before a clip is generated, and the service has daily caps.

Creators see none of this. The app shows shots, plain-language progress and a player.
The full trace of a video is at `/ops/<video id>/trace?code=<access code>`.

## Run it locally

Python 3.11+ and ffmpeg.

```bash
pip install -e ".[dev]"
pytest
python -m directorcut serve            # http://localhost:8000
```

Without API keys the app runs in **demo mode**: two simulated models render
placeholder clips and fail at fixed rates, which is enough to try every flow. The page
says so in its footer.

To use real models, copy `.env.example` to `.env`, fill it in and export it:

| Variable | What it does |
|---|---|
| `REPLICATE_API_TOKEN` | Turns on real video generation through Replicate |
| `ANTHROPIC_API_KEY` | Turns on the LLM planner and the vision judge |
| `DIRECTORCUT_VIDEO_MODELS` | `owner/name=price_per_second,...` in order of preference |
| `DIRECTORCUT_IMAGE_MODEL` | Optional `owner/name=price_per_image`. Turns on the subject reference and its approval step |
| `DIRECTORCUT_KEYFRAME_MODEL` | Optional `owner/name=price_per_image`. An image-editing model that takes a reference image; gives each shot its own opening frame |
| `DIRECTORCUT_ACCESS_CODE` | Required to make videos. Set it on anything public |
| `DIRECTORCUT_BUDGET_USD`, `DIRECTORCUT_DAILY_USD`, `DIRECTORCUT_DAILY_VIDEOS` | Spend limits |
| `DIRECTORCUT_MAX_SECONDS`, `DIRECTORCUT_MAX_SHOT_SECONDS` | Length limits |
| `DIRECTORCUT_HOME` | Where the database and media live |

Settings are also accepted under the earlier `CUTROOM_` prefix, so an existing deployment keeps
working unchanged.

The prices in `DIRECTORCUT_VIDEO_MODELS` are your own estimates from each model's page.
Budgets are counted against them, so set them before trusting a budget. LLM calls are
not counted in the budget.

The Replicate adapter reads each model's published input schema and sends only the
fields that model accepts, so most text-to-video models work without code changes.

## Deploy

The repository has a `Dockerfile` and a Render blueprint (`render.yaml`).

1. Push this repository to GitHub.
2. In Render: New, Blueprint, select the repository.
3. Fill in `DIRECTORCUT_ACCESS_CODE`, `REPLICATE_API_TOKEN` and `ANTHROPIC_API_KEY`.
4. Open the service URL and enter the access code.

The blueprint mounts a persistent disk at `/data`. Without one, videos and history are
lost on every deploy. Any host that runs a container with a mounted volume works the same way.

## Command line

```bash
python -m directorcut run "A 12s calm film about a sleep tea" --crash-after 6
python -m directorcut resume <id>      # continues from the checkpoint
python -m directorcut approve <id>
python -m directorcut feedback <id> s2 "Slower, warmer light." --remember
python -m directorcut trace <id>       # writes an HTML trace
python -m directorcut bench
```

## Benchmark of the harness logic

`python -m directorcut bench` runs twelve fixed briefs through three configurations using
the **simulated** models, which draw their failures from the brief so every
configuration faces the same ones. A delivery is clean when every clip in it passes
all checks, measured afterwards for all three.

| Configuration | Delivered | Clean | Shipped with a defect | Cost per clean video |
|---|---|---|---|---|
| No retries, no evals | 8/12 | 2/12 | 6 | $2.46 |
| Retries and eval gate | 10/12 | 10/12 | 0 | $0.79 |
| Plus a second model | 12/12 | 12/12 | 0 | $0.84 |

This measures the harness against injected faults with invented rates and prices. It
says nothing about any real model.

## Status

- Tested: 32 tests cover the harness, the creator API and the real-model path. The
  Replicate adapter, the planner and the judge are tested against stubbed HTTP
  responses with real video files.
- Verified live: planning, text-to-video generation, the vision judge, retakes, change
  requests and publishing.
- Not yet verified live: the reference and keyframe workflow (image models and image-to-video).
- If no take of a shot passes review, the closest one is kept and flagged to the creator;
  a change request that cannot be made keeps the take it was meant to replace.
- Known limits: one process with a thread pool and SQLite; one shared access code
  instead of accounts; videos are silent unless you add music to `$DIRECTORCUT_HOME/assets/library.json`;
  the judge sees three frames per clip, so it can miss problems between them.
