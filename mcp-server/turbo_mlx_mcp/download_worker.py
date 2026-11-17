"""Downloads one repository snapshot; run as a child process so the server can stop it at any time.

    python -m turbo_mlx_mcp.download_worker <repo> <revision> <patterns as JSON>

Files already finished stay in the Hugging Face cache; the one in progress is usually lost when the process is
stopped and is removed by cancel_download, so the next download fetches it again.
"""

import json
import os
import sys

from huggingface_hub import snapshot_download


def main() -> None:
    repo, revision, patterns = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    snapshot_download(repo, revision=revision, allow_patterns=patterns, token=os.environ.get("HF_TOKEN") or None)


if __name__ == "__main__":
    main()
