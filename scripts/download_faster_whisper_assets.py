"""Acquire only the pinned files used by the faster-whisper Small benchmark."""

from huggingface_hub import snapshot_download


MODEL_REVISION = "536b0662742c02347bc0e980a01041f333bce120"
REQUIRED_FILES = (
    "config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.txt",
)


def main() -> None:
    snapshot_download(
        repo_id="Systran/faster-whisper-small",
        revision=MODEL_REVISION,
        allow_patterns=list(REQUIRED_FILES),
    )
    print(
        {
            "status": "downloaded",
            "revision": MODEL_REVISION,
            "file_count": len(REQUIRED_FILES),
        }
    )


if __name__ == "__main__":
    main()
