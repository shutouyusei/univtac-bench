"""Check the installed Isaac Sim 4.5 environment: package versions, torch's CUDA and the GPU. Exit 1 on any mismatch."""

from importlib.metadata import PackageNotFoundError, version

import torch

EXPECTED = {
    "isaacsim": "4.5.0.0", "torch": "2.7.0", "torchvision": "0.22.0", "numpy": "1.26.4",
    "tacex": "0.1.0", "tacex-assets": "0.1.0", "tacex-uipc": "0.1.0",
    "isaaclab": None, "nvidia-curobo": None, "warp-lang": None, "torch-scatter": None, "pyuipc": None,
}


def main() -> int:
    failed = False
    for package, wanted in EXPECTED.items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            print(f"[missing] {package}")
            failed = True
            continue
        if package == "torch-scatter":
            ok = installed.endswith("+pt27cu128")
        else:
            ok = wanted is None or installed.split("+")[0] == wanted
        print(f"[{'ok' if ok else 'wrong'}] {package}=={installed}")
        failed |= not ok
    cuda_ok = torch.version.cuda == "12.8"
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"[{'ok' if cuda_ok else 'wrong'}] torch CUDA {torch.version.cuda}; device {device}")
    return 1 if failed or not cuda_ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
