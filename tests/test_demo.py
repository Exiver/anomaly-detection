from anomaly.service.demo import render_demo


def test_static_variant_has_the_api_url_field():
    assert 'id="api"' in render_demo(static=True)
    assert 'id="api"' not in render_demo(static=False)


def test_both_variants_are_complete():
    for static in (True, False):
        page = render_demo(static=static)
        assert "Visual anomaly detection" in page
        assert "<!--API_ROW-->" not in page
        assert "/*API_BASE_JS*/" not in page