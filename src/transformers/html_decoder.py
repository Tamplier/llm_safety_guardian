import html

def decode_html_entities(X):
    if isinstance(X, list):
        return [html.unescape(x) for x in X]

    return X.apply(html.unescape)
