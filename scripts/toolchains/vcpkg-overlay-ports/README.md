# vcpkg overlay ports

Ports here shadow the ones in the pinned vcpkg registry (`VCPKG_COMMIT` in
`scripts/install.sh`). `install.sh` exports this directory as
`VCPKG_OVERLAY_PORTS` before building libuipc.

- `tinygltf`: same port as the registry, with the SHA512 of the
  `syoyo/tinygltf` v2.9.6 GitHub archive updated to what GitHub serves now.
  The pinned registry hash no longer matches the regenerated archive, so
  `vcpkg install` fails on this header-only dependency.
