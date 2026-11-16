import argparse
import html
import shutil
from pathlib import Path
from urllib.parse import quote

from anomaly.service.demo import render_demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="static_demo")
    parser.add_argument("samples", nargs="*", help="Overlay PNGs from save_overlays.py")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gallery = ""
    if args.samples:
        samples_dir = out / "samples"
        samples_dir.mkdir(exist_ok=True)
        figures = []
        for sample in args.samples:
            path = Path(sample)
            shutil.copy(path, samples_dir / path.name)
            figures.append(
                f'<figure><img src="samples/{quote(path.name)}" alt="{html.escape(path.stem)}">'
                f"<figcaption>{html.escape(path.stem)}</figcaption></figure>"
            )
        gallery = (
            "<h2>Saved example results</h2>"
            "<p>Red marks where the model sees an anomaly. These were produced by the live service.</p>"
            f'<div class="images">{"".join(figures)}</div>'
        )

    page = render_demo(static=True).replace("</body>", gallery + "\n</body>")
    (out / "index.html").write_text(page, encoding="utf-8")
    print(f"Static demo written to {out}/index.html")


if __name__ == "__main__":
    main()