"""Download pinned upstream YAM assets and retain provenance/license."""

import concurrent.futures
import hashlib
import json
import pathlib
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1] / "third_party" / "yam"


def get(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=90) as f:
                return f.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def fetch_repo(repo, prefix, dest, commit):
    tree = json.loads(
        get(f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1")
    )
    sha = tree["sha"]
    paths = [
        x["path"]
        for x in tree["tree"]
        if x["type"] == "blob"
        and (x["path"].startswith(prefix) or x["path"] == "LICENSE")
    ]

    def one(path):
        rel = path[len(prefix) :] if path.startswith(prefix) else path
        out = ROOT / dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        content = (
            out.read_bytes()
            if out.exists()
            else get(f"https://raw.githubusercontent.com/{repo}/{sha}/{path}")
        )
        out.write_bytes(content)
        return {
            "path": rel,
            "upstream_path": path,
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        files = list(ex.map(one, paths))
    (ROOT / dest / "provenance.json").write_text(
        json.dumps({"repo": repo, "commit": sha, "files": files}, indent=2)
    )
    print(dest, sha, len(files), flush=True)


if __name__ == "__main__":
    fetch_repo(
        "i2rt-robotics/i2rt",
        "i2rt/robot_models/arm/yam/v1/",
        "i2rt",
        "5b72c47239bd056d0fa6c1a39edeb0537c89443c",
    )
    fetch_repo(
        "google-deepmind/mujoco_menagerie",
        "i2rt_yam/",
        "menagerie",
        "8161bba264d7fa7c99ca301e91e7fb44737676ad",
    )
    fetch_repo(
        "i2rt-robotics/i2rt",
        "robot_models/yam/",
        "i2rt_legacy",
        "d4efb66d81bd8bde42909880b16591d4af82e8c0",
    )
