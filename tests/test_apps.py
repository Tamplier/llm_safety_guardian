from apps.gradio.app import app

def test_gradio():
    text = 'Test'
    output = app(text)
    assert output.root == {'class': text, 'confidence': 1.0}
