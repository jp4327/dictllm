"""Architecture-independent projection metadata."""

PROJECTION_PATHS = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}

CALIBRATION_FAMILIES = {
    "attention_input": PROJECTION_PATHS["k"],
    "attention_output": PROJECTION_PATHS["o"],
    "mlp_input": PROJECTION_PATHS["up"],
    "mlp_output": PROJECTION_PATHS["down"],
}

TARGET_TO_FAMILY = {
    "q": "attention_input",
    "k": "attention_input",
    "v": "attention_input",
    "o": "attention_output",
    "gate": "mlp_input",
    "up": "mlp_input",
    "down": "mlp_output",
}

STAGE_ORDER = ("factorize", "prune", "refit")
STAGE_DIRECTORIES = {
    "factorize": "01_factorized",
    "prune": "02_pruned",
    "refit": "03_refit",
}
MANIFEST_NAME = "dictllm_manifest.json"

