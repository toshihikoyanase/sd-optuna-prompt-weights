import math
import os

import gradio as gr

import modules.processing
import modules.scripts as scripts
from modules import shared
from modules.processing import process_images, Processed
from modules.prompt_parser import parse_prompt_attention


class OptunaControlNetWeightScript(scripts.Script):
    def __init__(self) -> None:
        super().__init__()

    def title(self) -> str:
        return "Optuna ControlNet weights"
    
    def show(self, is_img2img) -> bool:
        return True
 
    def ui(self, is_img2img):
        n_trials_per_iter = gr.Slider(minimum=1, maximum=20, step=1, value=2, label="Number of trials per iteration")
        lower = gr.Slider(minimum=0, maximum=1, step=0.1, value=0, label="Lower bound of suggestion")
        upper = gr.Slider(minimum=0, maximum=1, step=0.1, value=1, label="Upper bound of suggestion")
        storage = gr.Textbox(placeholder="Input DB URL of storage.", value="sqlite:///optuna.db", label="Storage")
        study_name = gr.Textbox(placeholder="Name of study", label="Study name")
        artifact_dir = gr.Textbox(placeholder="Path to artifact dir for Optuna dashboard", value="./artifact", label="Artifact_dir")
        excluded_keywords = gr.Textbox(placeholder="Keywords to be excluded for optuna. Comma separated.", label="Excluded keywords")
        return (n_trials_per_iter, lower, upper, storage, study_name, artifact_dir, excluded_keywords)

    def run(self, p, n_trials_per_iter, lower, upper, storage, study_name, artifact_dir, excluded_keywords):
        modules.processing.fix_seed(p)

        print("prompt", p.prompt)
        print("n_trials_per_iter: ", n_trials_per_iter)
        print("storage: ", storage)
        print("study_name", study_name)
        import optuna
        from optuna.distributions import FloatDistribution
        from optuna_dashboard import ObjectiveChoiceWidget
        from optuna_dashboard import register_objective_form_widgets
        from optuna_dashboard import set_objective_names
        from optuna_dashboard.artifact import upload_artifact
        from optuna_dashboard.artifact.file_system import FileSystemBackend

        study = optuna.create_study(
            storage=storage or None,
            study_name=study_name or None,
            sampler=optuna.samplers.TPESampler(multivariate=True, constant_liar=True),
            load_if_exists=True,
        )
        if len(study.trials) == 0:
            set_objective_names(study, ["Human Perception"])
            register_objective_form_widgets(study, widgets=[
                ObjectiveChoiceWidget(
                    choices=["Good 👍", "So so 👋", "Bad 👎"],
                    values=[-1, 0, 1],
                    description="Choose Good 👍, So so 👋 or Bad 👎.",
                ),
            ])
        os.makedirs(artifact_dir, exist_ok=True)
        artifact_backend = FileSystemBackend(artifact_dir)

        original_prompt = p.prompt
        if len(study.trials) != 0:
            assert study.user_attrs.get("prompt", "") == original_prompt, "Please set a new study_name when you modify the prompt."
        else:
            study.set_user_attr("prompt", original_prompt)

        # Add controlnet weight.
        default_params = {"control_net_weight": 1}

        fixed_distributions = {name: FloatDistribution(lower, upper) for name in default_params}

        trials = []
        control_net_weights = []
        # Try the default weights first as a baseline.
        if len(study.trials) == 0:
            study.enqueue_trial(default_params)
            default_trial = study.ask(fixed_distributions=fixed_distributions)
            trials.append(default_trial)
            control_net_weights.append(1)

        while len(control_net_weights) < n_trials_per_iter:
            trial = study.ask(fixed_distributions=fixed_distributions)
            params = trial.params
            suggested = []
            control_net_weights.append(params["control_net_weight"])
            print(params["control_net_weight"])
            trials.append(trial)

        p.n_iter = 1
        p.seed = [int(p.seed) for _ in range(len(control_net_weights))]
        shared.opts.data["control_net_allow_script_control"] = True
        for idx, (weight, trial) in enumerate(zip(control_net_weights, trials)):
            setattr(p, "control_net_weight", weight)
            processed = process_images(p)
            image = processed.images[0]
            file_path = f"/tmp/{trial.number}.png"
            image.save(file_path)
            upload_artifact(artifact_backend, trial, file_path)
        return processed
