import json

from project_config import OUTPUT_ROOT
from restoration.io import atomic_json

SUMMARY_PATH = OUTPUT_ROOT / "readiness/ffhq_full_restoration/summary.json"
OUTPUT_PATH = OUTPUT_ROOT / "readiness/ffhq_full_restoration_smoke_validation.json"


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    gates = dict(summary["gates"])
    gates["overall"] = bool(summary["passed"])
    payload = {"status": "ready" if all(gates.values()) else "blocked", "gates": gates}
    atomic_json(OUTPUT_PATH, payload)
    print(json.dumps(payload, indent=2))
    if not all(gates.values()):
        raise RuntimeError("FFHQ full restoration smoke validation failed")


if __name__ == "__main__":
    main()
