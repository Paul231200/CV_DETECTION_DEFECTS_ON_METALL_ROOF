# Sauron Pilot — Defect Detection System

Industrial video analytics pipeline that monitors an RTSP stream with **YOLO**-based defect detection, **sheet-presence** gating (SSIM / MAD vs. reference frames), optional **AMQP** event publishing, and **MQTT** signalling for automation gear (for example relay controls).

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=flat&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=flat&logo=opencv&logoColor=white" alt="OpenCV"/>
  <img src="https://img.shields.io/badge/Ultralytics-YOLO-00FFFF?style=flat" alt="YOLO"/>
</p>

---

## Highlights

| Area | Detail |
|------|--------|
| **Inference** | YOLO detection with ROI masking, ONNX-friendly workflow, optional tracking |
| **Sheet logic** | Defect inference runs only after a configurable delay once a sheet is detected versus “no-sheet” templates |
| **Events** | JSON events over RabbitMQ (AMQP); fast MQTT pulses for PLC / relay lines |
| **Ops** | Dockerized runtime, GitLab CI build template included |

---

## Repository layout

| Path | Role |
|------|------|
| `main.py` | Stream loop, detection, MQTT/AMQP integration, persistence |
| `publisher.py` | AMQP (`pika`) publisher thread |
| `mqtt_publisher.py` | MQTT client (topics/credentials via environment) |
| `settings.py` | Optional loader: fetch YOLO weights from GitLab registry metadata via ClickHouse |
| `Dockerfile` / `docker-compose.yml` | Container build and orchestration |

---

## Requirements

- **Python 3.10+** (matching the Dockerfile)
- **FFmpeg-compatible** RTSP endpoint (OpenCV uses FFmpeg for RTSP)
- **RabbitMQ** URL if you enable AMQP
- Optional **MQTT** broker credentials and control topics (`K1`, `K2` style relays)

Dependencies are listed in [`requirements.txt`](requirements.txt).

---

## Configuration & secrets

**Do not commit real credentials.** The repository ships with [`.env.example`](.env.example) as the only tracked template.

1. Copy the example file:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and set:

   - **`RTSP_URL`** — camera or NVR RTSP URI (credentials belong here, not in Git)
   - **`AMQP_URL`** — `amqp://user:password@host:5672/vhost`
   - **`UUID_Publisher`** — zone/session id passed to AMQP payloads
   - **`MQTT_BROKER`**, **`MQTT_PORT`**, **`MQTT_USER`**, **`MQTT_PASSWORD`**
   - **`MQTT_TOPIC_K1`** / **`MQTT_TOPIC_K2`** — your controller topics (supports both `/devices/…` and `devices/…` variants via automatic fallbacks)

If **`MQTT_BROKER`** is empty, MQTT is disabled and the app continues with AMQP only (where configured).

Additional tuning (sheet detector thresholds, paths) is documented via variables in `.env.example`.

---

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export $(grep -v '^#' .env | xargs)
python main.py --use-original --model models/your_model.pt --source "$RTSP_URL"
```

CLI flags are defined in `main.py` (`argparse`): `--model`, `--output`, `--test-image`, etc.

---

## Run with Docker Compose

Ensure `.env` exists next to `docker-compose.yml`:

```bash
docker compose build
docker compose up -d
```

Images are saved under the mounted volume **`defect_pilot_defects_volume`** (see `docker-compose.yml`).

---

## Optional: model bootstrap from GitLab (`settings.py`)

[`settings.py`](settings.py) connects to ClickHouse (`DB_*` env vars) and downloads a pinned model artifact using a GitLab `PRIVATE-TOKEN`. Use **`MODEL_UUID`** to select the row. This path is orthogonal to **`main.py`**’s local `--model` file workflow.

---

## GitLab CI

[`.gitlab-ci.yml`](.gitlab-ci.yml) includes a Docker build job for **`main`** and a placeholder deploy stub. Adapt registry variables and SSH/deploy steps to your environment; keep secrets in **CI/CD variables**, never in YAML.

---

## Security checklist before pushing to GitHub

- [ ] Rotate any password that ever appeared in a local file committed by mistake (`RTSP_URL`, MQTT, AMQP, GitLab tokens)
- [ ] Keep **`.env` untracked** (already ignored); only **`.env.example`** is public-facing placeholders  
- [ ] Review `docker-compose` history — no inline secrets  

---

## Author

<table>
  <tr>
    <td align="center">
      <strong>Barakhnin Pavel Ivanovich</strong><br/>
      AI & Computer Vision Engineer<br/>
      <sub>Architecting production vision pipelines — detection, tracking, deployment</sub>
    </td>
  </tr>
</table>

Questions or collaboration: open an issue or reach out via your preferred channel linked from your GitHub profile.

---

## License

No license file is bundled in this snapshot. Before sharing publicly, add a `LICENSE` that matches how you intend others to use the code.
