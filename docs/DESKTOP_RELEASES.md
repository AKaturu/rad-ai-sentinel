# Desktop Releases

GitHub Actions can build downloadable desktop artifacts for users who do not want to install Python.

## Release Artifacts

| Platform | Artifact | Contents |
|---|---|---|
| Windows | `rad-ai-sentinel-windows.zip` | `RadAISentinel.exe` and bundled runtime files |
| macOS | `rad-ai-sentinel-macos.dmg` | `RadAISentinel.app` |
| Linux | `rad-ai-sentinel-linux.tar.gz` | `RadAISentinel` executable bundle |

The desktop launcher starts the Streamlit dashboard locally in the user's browser. It does not send data to a hosted service.

## Build Locally

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[app,packaging]"
python scripts/build_native.py
```

The build output is written to `dist/native/`.

## CI Release Build

The workflow in `.github/workflows/native-release.yml` runs on:

- manual `workflow_dispatch`
- tags that match `v*`

For a public release, create a tag such as `v0.1.0`, let the workflow produce the artifacts, then attach the artifacts to a GitHub Release.

## Notes

- The packaged app includes synthetic demo mode and CSV upload mode.
- No Census, OpenRouteService, or other API key is required for this project.
- Real monitoring CSVs should only be used under appropriate institutional approval and privacy controls.
