from invoke import task

@task
def retrain_model(c, skip_preprocessing=False, sample_n=None, opt_trials=30):
    """Retrain the model."""

    cmd = [
        'python', '-m', 'src.scripts.train',
    ]
    if skip_preprocessing:
        cmd.append('--skip_preprocessing')
    if sample_n:
        cmd.append(f'--sample_n={sample_n}')
    cmd.append(f'--optimization_trials={opt_trials}')
    command_str = ' '.join(cmd)

    c.run(command_str, pty=True)

@task
def cli(c):
    c.run('python -m apps.cli.__main__')

@task
def gradio(c):
    c.run('python -m apps.gradio')

@task
def start_telegram_bot(c):
    c.run('python -m apps.telegram.bot')

@task
def collect_messages(c):
    c.run('python -m src.scripts.scrap_reddit')
