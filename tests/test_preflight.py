import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "preflight", Path(__file__).resolve().parents[1] / "scripts" / "preflight.py"
)
preflight = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(preflight)


def test_detects_common_secret_shapes():
    cases = {
        "Groq key": "gsk_" + "a" * 30,
        "AWS access key id": "AKIA" + "A" * 16,
        "Hugging Face token": "hf_" + "b" * 34,
        "private key block": "-----BEGIN " + "RSA PRIVATE KEY-----",
        "generic key assignment": 'api_key = "' + "x" * 24 + '"',
    }
    for name, text in cases.items():
        assert name in [hit[0] for hit in preflight.find_secrets(text)], name


def test_placeholders_are_not_flagged():
    clean = [
        "AD_API_KEY: ${AD_API_KEY:?set AD_API_KEY in your environment}",
        "credentials: dev-secret",
        "api_key: SecretStr | None = None",
        'headers = {"X-API-Key": args.api_key}',
    ]
    for text in clean:
        assert preflight.find_secrets(text) == [], text


def test_secret_location_is_reported_without_the_value():
    text = "line one\n" + 'token = "' + "z" * 24 + '"\n'
    assert preflight.find_secrets(text) == [("generic key assignment", 2)]


def test_forbidden_paths():
    assert preflight.forbidden(".env")
    assert preflight.forbidden(".env.local")
    assert preflight.forbidden(".env.example") is None
    assert preflight.forbidden("releases/v1/bottle.pt")
    assert preflight.forbidden("releases/.gitkeep") is None
    assert preflight.forbidden("data/mvtec_ad/bottle/train/good/000.png")
    assert preflight.forbidden("artifacts/patchcore/bottle.pt")
    assert preflight.forbidden("src/anomaly/data/mvtec.py") is None
    assert preflight.forbidden("docs/img/demo_defect.png") is None