import html

def decode_html_entities(X):
    return X.apply(html.unescape)
