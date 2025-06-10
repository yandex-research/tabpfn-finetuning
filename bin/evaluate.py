import shutil
from copy import deepcopy
from pathlib import Path

from loguru import logger

import lib


def main(
    path: Path,
    n_seeds: int = 15,
    function: None | str = None,
    *,
    force: bool = False,
):
    path = lib.get_path(path)
    assert path.is_dir()

    if path.name.endswith('tuning'):
        assert function is None
        assert (path / 'DONE').exists()
        from_tuning = True
        tuning_report = lib.load_report(path)
        function_qualname = tuning_report['config']['function']
        template_config = tuning_report['best']['config']
        evaluation_dir = path.with_name(path.name.replace('tuning', 'evaluation'))
        evaluation_dir.mkdir(exist_ok=True)

    elif path.name.endswith('evaluation'):
        assert function is not None
        from_tuning = False
        function_qualname = function
        evaluation_dir = path
        template_config = lib.load_config(evaluation_dir / '0')

    else:
        raise ValueError(f'Bad input path: {path}')
    del path

    function_: lib.MainFunction = lib.import_(function_qualname)
    for seed in range(n_seeds):
        output = evaluation_dir / str(seed)
        config_path = output.with_suffix('.toml')
        done = (output / 'DONE').exists()
        if config_path.exists() and not done:
            if output.exists():
                logger.warning(f'Removing the incomplete output {output}')
                shutil.rmtree(output)
            if from_tuning or seed > 0:
                config_path.unlink()

        config = deepcopy(template_config)
        config['seed'] = seed
        if 'catboost' in function_qualname:
            if config['model']['task_type'] == 'GPU':
                config['model']['task_type'] = 'CPU'  # this is crucial for good results
                thread_count = config['model'].get('thread_count', 1)
                config['model']['thread_count'] = max(thread_count, 4)

        try:
            if (from_tuning or seed > 0) and not (config_path.exists() and done):
                lib.dump_config(output, config)
            function_(config, output, force=force)
        except Exception:
            if from_tuning or seed > 0:
                config_path.unlink(True)
            shutil.rmtree(output, True)
            raise


if __name__ == '__main__':
    lib.configure_libraries()
    lib.run_cli(main)
