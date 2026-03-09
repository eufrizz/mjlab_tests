import torch
from mjlab.rl import MjlabOnPolicyRunner
from rsl_rl.runners import DistillationRunner


class MjlabDistillationRunner(DistillationRunner):
    """DistillationRunner compatible with mjlab conventions.

    - Strips cnn_cfg=None from student/teacher configs before construction so
      MLPModel doesn't receive an unexpected kwarg (mirrors MjlabOnPolicyRunner).
    - save() stores only student_state_dict (no teacher weights) so checkpoints
      are small and directly loadable by MjlabStudentOnPolicyRunner / play.py.
    - load() remaps play.py's load_cfg={"actor": True} to None so Distillation.load()
      auto-detects the checkpoint type, and relaxes strict to handle the stochastic
      PPO actor's "std" key being absent in the non-stochastic teacher model.
    """

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        for key in ("student", "teacher"):
            if key in train_cfg and train_cfg[key].get("cnn_cfg") is None:
                train_cfg[key].pop("cnn_cfg", None)
        super().__init__(env, train_cfg, log_dir, device)

    def save(self, path, infos=None):
        # Match OnPolicyRunner.save() exactly, replacing self.alg.save() with
        # student-only weights. Teacher weights are large and not needed for
        # deployment (they can be reloaded from the original PPO checkpoint).
        saved_dict = {
            "student_state_dict": self.alg.student.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        # play.py always passes load_cfg={"actor": True}, which Distillation.load()
        # doesn't understand. Pass None instead so Distillation.load() auto-detects
        # the checkpoint type from its keys:
        #   - distillation checkpoint (has student_state_dict) → loads student + teacher
        #   - PPO checkpoint (has actor_state_dict)            → loads as teacher only
        # Also relax strict: a PPO actor is stochastic (has "std" in state_dict) but
        # our teacher model is non-stochastic, so strict=True would raise a KeyError.
        if load_cfg == {"actor": True}:
            load_cfg = None
            strict = False
        return super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)


class MjlabStudentOnPolicyRunner(MjlabOnPolicyRunner):
    """OnPolicyRunner for the regular RGB env that accepts distillation checkpoints.

    Architecturally identical to MjlabOnPolicyRunner (same PPO actor/critic config),
    but load() transparently remaps student_state_dict → actor_state_dict when given
    a distillation checkpoint. This lets the trained student be played in the regular
    RGB env (no privileged obs group overhead from distillation training).

    Also accepts regular PPO checkpoints, falling back to normal loading.
    """

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
        if "student_state_dict" not in loaded_dict:
            # Regular PPO checkpoint — delegate to parent (re-reads from disk).
            return super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)

        # Distillation checkpoint: remap student → actor so PPO alg.load() picks it up.
        loaded_dict["actor_state_dict"] = loaded_dict.pop("student_state_dict")
        loaded_dict.pop("teacher_state_dict", None)  # not needed for inference

        load_iteration = self.alg.load(loaded_dict, {"actor": True}, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict.get("iter", 0)

        infos = loaded_dict.get("infos", {})
        if infos and "env_state" in infos:
            self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
        return infos
