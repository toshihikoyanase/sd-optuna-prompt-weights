import os
import gradio as gr

import modules.processing
import modules.scripts as scripts
from modules.processing import process_images, Processed
from modules.prompt_parser import parse_prompt_attention


class Script(scripts.Script):
    def __init__(self) -> None:
        super().__init__()

    def title(self) -> str:
        return "Optuna"
    
    def show(self, is_img2img) -> bool:
        return True
 
    def ui(self, is_img2img):
        gr.Label()
        n_trials_per_iter = gr.Slider(minimum=1, maximum=20, step=1, value=2, label="n_trials per iteration")
        lower = gr.Slider(minimum=0.5, maximum=1, step=0.1, value=0.5, label="lower bound of suggestion")
        upper = gr.Slider(minimum=1.0, maximum=2, step=0.1, value=2, label="upper bound of suggestion")
        storage = gr.Textbox(placeholder="Input DB URL of storage.", value="sqlite:///optuna.db", label="storage")
        study_name = gr.Textbox(placeholder="Name of study", label="study_name")
        artifact_dir = gr.Textbox(placeholder="Path to artifact dir for Optuna dashboard", value="./artifact", label="artifact_dir")
        excluded_keywords = gr.Textbox(placeholder="Keywords to be excluded for optuna. Comma separated.", label="excluded_keywords")
        return (n_trials_per_iter, lower, upper, storage, study_name, artifact_dir, excluded_keywords)

    def run(self, p, n_trials_per_iter, lower, upper, storage, study_name, artifact_dir, excluded_keywords):
        modules.processing.fix_seed(p)

        print("prompt", p.prompt)
        print("n_trials_per_iter: ", n_trials_per_iter)
        print("storage: ", storage, type(storage))
        print("study_name", study_name, type(study_name))
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

        ek = set([k.strip() for k in excluded_keywords.split(",")])
        print(f"Keywords excluded from weight optimization: {ek}")

        original_prompt = p.prompt
        if len(study.trials) != 0:
            assert study.user_attrs.get("prompt", "") == original_prompt, "Please set a new study_name when you modify the prompt."
        else:
            study.set_user_attr("prompt", original_prompt)

        all_prompts = []

        parse_result = parse_prompt_attention(original_prompt)
        print(f"prompt was pased as follows:", parse_result)
        default_params = {
            f"{i}:{token}": weight
            for i, (token, weight) in enumerate(parse_result)
            if weight != 1 and token not in ek
        }

        fixed_distributions = {name: FloatDistribution(lower, upper) for name in default_params}

        trials = []
        # Try the default weights first as a baseline.
        if len(study.trials) == 0:
            all_prompts.append(original_prompt)
            study.enqueue_trial(default_params)
            default_trial = study.ask(fixed_distributions=fixed_distributions)
            trials.append(default_trial)

        while len(all_prompts) < n_trials_per_iter:
            trial = study.ask(fixed_distributions=fixed_distributions)
            params = trial.params
            suggested = []
            for i, (token, weight) in enumerate(parse_result):
                if f"{i}:{token}" not in default_params:
                    suggested.append(token)
                    continue
                new_weight = params[f"{i}:{token}"]
                suggested.append(f"({token}:{new_weight})")
            new_prompt = " ".join(suggested)
            print(f"{original_prompt} --> {new_prompt}")
            all_prompts.append(new_prompt)
            trials.append(trial)

        p.n_iter = len(all_prompts)
        print(f"Optuna trial will create {len(all_prompts)} images using a total of {p.n_iter} batches.")
        p.prompt = all_prompts
        p.seed = [int(p.seed) for _ in range(len(all_prompts))]
        
        processed = process_images(p)
        for trial, image in zip(trials, processed.images[1:]):
            file_path = f"/tmp/{trial.number}.png"
            image.save(file_path)
            upload_artifact(artifact_backend, trial, file_path)
        return processed
