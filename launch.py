import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download


APP_DIR = Path("/opt/RC-stable-audio-tools")
MODELS_ROOT = Path(os.environ.get("FOUNDATION_MODELS_ROOT", "/models")).resolve()
MODEL_DIR = MODELS_ROOT / "Foundation-1"
OUTPUT_DIR = Path(os.environ.get("FOUNDATION_OUTPUT_DIR", "/outputs")).resolve()
MODEL_REPO = "RoyalCities/Foundation-1"


def ensure_config() -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "models_directory": str(MODELS_ROOT),
        "generations_directory": str(OUTPUT_DIR),
        "hffs": [
            {
                "path": str(MODELS_ROOT),
                "options": [MODEL_REPO],
            }
        ],
    }
    (APP_DIR / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def ensure_model_files() -> tuple[Path, Path]:
    config_path = Path(
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename="model_config.json",
            local_dir=MODEL_DIR,
        )
    )
    ckpt_path = Path(
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename="Foundation_1.safetensors",
            local_dir=MODEL_DIR,
        )
    )
    return config_path, ckpt_path


def main() -> None:
    ensure_config()
    model_config_path, ckpt_path = ensure_model_files()

    os.chdir(APP_DIR)

    import torch
    from stable_audio_tools.interface.gradio import create_ui

    port = int(os.environ.get("PORT", "7860"))
    ui = create_ui(
        model_config_path=str(model_config_path),
        ckpt_path=str(ckpt_path),
        model_half=torch.cuda.is_available(),
        gradio_title="Foundation-1",
    )
    ui.queue()
    ui.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=port,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()

