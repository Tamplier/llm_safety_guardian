from invoke import task

@task
def relabel_manually(c, sample_n=1000):
    c.run(f'python -m src.scripts.relabel_manually --sample_n={sample_n}', pty=True)

@task
def prepare_input(c, skip_preprocessing=False, skip_vectorization=False, sample_n=None, disale_new_sources=False):
    """Fix typos, generate extra features, vectorize text"""

    cmd = [
        'python', '-m', 'src.scripts.prepare_input',
    ]
    if skip_preprocessing:
        cmd.append('--skip_preprocessing')
    if skip_vectorization:
        cmd.append('--skip_vectorization')
    if disale_new_sources:
        cmd.append('--disale_new_sources')
    if sample_n:
        cmd.append(f'--sample_n={sample_n}')
    command_str = ' '.join(cmd)

    c.run(command_str, pty=True)

@task
def retrain_model(c, opt_trials=30, frac_noise=1.0):
    """Retrain the model."""
    cmd = ['python', '-m', 'src.scripts.train']
    cmd.append(f'--optimization_trials={opt_trials}')
    cmd.append(f'--frac_noise={frac_noise}')
    command_str = ' '.join(cmd)

    c.run(command_str, pty=True)

@task
def cli(c):
    c.run('python -m apps.cli.__main__', pty=True)

@task
def gradio(c):
    c.run('python -m apps.gradio')

@task
def start_telegram_bot(c):
    c.run('python -m apps.telegram.bot')

@task
def collect_messages(c):
    c.run('python -m src.scripts.scrap_reddit')
