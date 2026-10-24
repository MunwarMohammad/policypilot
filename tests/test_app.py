from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_demo_ui_answers_with_sources():
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py", default_timeout=15
    ).run()
    assert not app.exception
    app.chat_input[0].set_value("supply chain").run()
    assert not app.exception
    assert any("demo-1" in item.value for item in app.markdown)
