"""
Bake the InsightFace `buffalo_l` model pack into the container image so
the first request isn't a 30-second cold start.

Invoked from the Dockerfile during build; harmless to run locally too.
"""

from insightface.app import FaceAnalysis


def main() -> None:
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("buffalo_l prepared OK")


if __name__ == "__main__":
    main()
