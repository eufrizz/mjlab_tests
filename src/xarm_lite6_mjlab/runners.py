from rsl_rl.runners import DistillationRunner


class MjlabDistillationRunner(DistillationRunner):
    """DistillationRunner compatible with play.py's load_cfg={"actor": True}.

    play.py always calls runner.load(..., load_cfg={"actor": True}). For a
    distillation checkpoint that key is meaningless, so we remap it to
    {"student": True} before delegating to Distillation.load().
    """

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        if load_cfg == {"actor": True}:
            load_cfg = {"student": True}
        return super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
