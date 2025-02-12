import os 

def get_available_models():
    models_dir = "models"
    if not os.path.exists(models_dir):
        return []
    files = os.listdir(models_dir)
    model_names = [os.path.splitext(f)[0] for f in files if f.endswith(".pkl")]
    return model_names