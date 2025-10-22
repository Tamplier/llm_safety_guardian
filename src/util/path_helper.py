from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

class PathConfig:
    def __init_subclass__(cls):
        parts = cls.__qualname__.split('.')[1:]
        base_path = PROJECT_ROOT / Path(*parts) if parts else PROJECT_ROOT

        for attr_name, attr_value in cls.__dict__.items():
            if not attr_name.startswith('_') and isinstance(attr_value, str):
                setattr(cls, attr_name, base_path / attr_value)

class PathHelper(PathConfig):
    project_root = PROJECT_ROOT
    class models(PathConfig):
        label_encoder = 'label_encoder.joblib'
        light_text_preprocessor = 'light_text_preprocessor.joblib'
        vectorizer = 'vectorizer.joblib'
        sbert_classifier = 'sbert_classifier.joblib'
    class data(PathConfig):
        class raw(PathConfig):
            data_set = 'Suicide_Detection.csv'
        class processed(PathConfig):
            x_train = 'X_train_transformed.csv'
            x_test = 'X_test_transformed.csv'
            y_train = 'y_train.csv'
            y_test = 'y_test.csv'
    class logs(PathConfig):
        train = 'train.log'
