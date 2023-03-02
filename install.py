import os

import launch

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

with open(req_file) as file:
    for lib in map(lambda l: l.strip(), file):
        if launch.is_installed(lib):
            continue
        launch.run_pip(f"install {lib}", f"sd-optuna-prompt-weights requirement: {lib}")
