"""The built-in models, ported from the removed app's ModelCatalog.swift.

Each entry pins the Hugging Face commit it was checked with, and lists the files its family reads, so a
download fetches exactly what the engine needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Families by what they make.
IMAGE, VIDEO, AUDIO, MODEL3D = "image", "video", "audio", "3d"


@dataclass(frozen=True)
class Companion:
    """A second checkpoint a family needs (LTX-2.3: Gemma 3 12B as its text encoder)."""

    repo: str
    revision: str
    patterns: tuple[str, ...]


@dataclass(frozen=True)
class Model:
    id: str  # short id used by the MCP tools
    name: str
    detail: str
    family: str
    repo: str
    revision: str
    size_gb: float
    memory_gb: int
    license: str
    variant: str | None = None
    quantize: int | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    # ---- family traits (ModelFamily in the app) ----
    @property
    def is_ltx(self) -> bool:
        return self.family in ("ltx-2", "ltx-2.5")

    @property
    def media(self) -> str:
        if self.is_ltx:
            return VIDEO
        if self.family == "idol":
            return MODEL3D
        if self.family in ("minimax-music3", "moss-soundeffect"):
            return AUDIO
        return IMAGE

    @property
    def is_upscaler(self) -> bool:
        return self.family == "seedvr2"

    @property
    def is_klein_base(self) -> bool:
        return self.family in ("flux2-klein", "flux3-klein") and "base" in (self.variant or self.repo).lower()

    @property
    def takes_reference_image(self) -> bool:
        return self.is_ltx or self.family in (
            "flux2-klein", "flux3-klein", "qwen-image-edit", "sensenova", "seedvr2", "idol",
        )

    @property
    def requires_reference_image(self) -> bool:
        return self.family in ("qwen-image-edit", "seedvr2", "idol")

    @property
    def takes_loras(self) -> bool:
        return self.family in ("z-image-turbo", "flux2-klein", "flux3-klein", "qwen-image", "qwen-image-edit", "ltx-2", "ltx-2.5")

    @property
    def produces_alpha(self) -> bool:
        return self.family == "ming"

    @property
    def size_multiple(self) -> int:
        if self.is_ltx:
            return 64
        return 32 if self.family == "sensenova" else 16

    @property
    def supports_guidance(self) -> bool:
        if self.family in ("ming", "moss-image", "qwen-image", "qwen-image-edit", "idol", "moss-soundeffect"):
            return True
        if self.family in ("flux2-klein", "flux3-klein"):
            return self.is_klein_base
        return False

    @property
    def supports_negative_prompt(self) -> bool:
        return self.supports_guidance and self.family not in ("ming", "flux3-klein", "idol")

    @property
    def supports_half_precision(self) -> bool:
        return self.family in ("z-image-turbo", "qwen-image", "qwen-image-edit")

    def default_steps(self, quality: bool = False) -> int:
        if quality and self.is_ltx:
            return 30
        return {
            "ming": 12, "z-image-turbo": 9, "flux2-klein": 50 if self.is_klein_base else 4,
            "flux3-klein": 50 if self.is_klein_base else 16, "moss-image": 50, "qwen-image": 20,
            "qwen-image-edit": 20, "sensenova": 8, "ltx-2": 8, "ltx-2.5": 8, "seedvr2": 1, "idol": 50,
            "minimax-music3": 30, "moss-soundeffect": 100,
        }[self.family]

    def step_range(self, quality: bool = False) -> tuple[int, int]:
        if quality and self.is_ltx:
            return (10, 50)
        return {
            "ming": (4, 40), "z-image-turbo": (4, 20), "flux2-klein": (10, 60) if self.is_klein_base else (2, 12),
            "flux3-klein": (10, 60) if self.is_klein_base else (4, 32), "moss-image": (10, 60),
            "qwen-image": (10, 50), "qwen-image-edit": (10, 50), "sensenova": (4, 16), "ltx-2": (4, 8),
            "ltx-2.5": (4, 8), "seedvr2": (1, 1), "idol": (10, 100), "minimax-music3": (8, 30),
            "moss-soundeffect": (10, 200),
        }[self.family]

    def default_guidance(self, quality: bool = False) -> float:
        if quality and self.is_ltx:
            return 3.0
        if self.family == "flux3-klein":
            return 3.5 if self.is_klein_base else 1.0
        if self.family in ("moss-image", "idol"):
            return 3.0
        if self.family in ("qwen-image", "qwen-image-edit", "moss-soundeffect") or self.is_klein_base:
            return 4.0
        return 1.0

    # ---- what to download ----
    @property
    def components(self) -> list[str]:
        f = self.family
        if f == "ming":
            return ["mllm", "connector", "mlp", "transformer", "vae"]
        if f in ("z-image-turbo", "flux2-klein", "flux3-klein", "moss-image", "qwen-image", "qwen-image-edit"):
            return ["transformer", "text_encoder", "vae"]
        if f == "moss-soundeffect":
            return ["text_encoder"]
        return []

    @property
    def download_patterns(self) -> list[str]:
        weights = [p for c in self.components for p in (f"{c}/*.safetensors", f"{c}/*.json")]
        f = self.family
        if f == "ming":
            return weights
        if f == "z-image-turbo":
            return weights + ["tokenizer/*"]
        if f in ("flux2-klein", "qwen-image", "qwen-image-edit"):
            return weights + ["tokenizer/**", "added_tokens.json", "chat_template.jinja"]
        if f == "flux3-klein":
            return weights + ["LICENSE.md", "NOTICE.md", "README.md"]
        if f == "moss-image":
            return weights + ["tokenizer/*", "scheduler/*.json", "model_index.json", "LICENSE", "README.md"]
        if f == "ltx-2":
            return ["LICENSE", "README.md", "config.json", "embedded_config.json", "quantize_config.json",
                    "split_model.json", "transformer-distilled-1.1.safetensors", "connector.safetensors",
                    "vae_decoder.safetensors", "vae_encoder.safetensors", "audio_vae.safetensors",
                    "vocoder.safetensors", "spatial_upscaler_x2_v1_1.safetensors",
                    "spatial_upscaler_x2_v1_1_config.json"]
        if f == "ltx-2.5":
            return ["LICENSE", "README.md", "embedded_config.json", "quantize_config.json", "split_model.json",
                    "transformer-distilled.safetensors", "connector.safetensors", "text_encoder.safetensors",
                    "text_encoder_config.json", "tokenizer.json", "tokenizer_config.json",
                    "vae_decoder_conv.safetensors", "vae_encoder_conv.safetensors", "audio_vae.safetensors",
                    "vocoder.safetensors", "spatial_upscaler_x2_v1_0.safetensors",
                    "spatial_upscaler_x2_v1_0_config.json"]
        if f == "seedvr2":
            return ["README.md", "seedvr2_ema_3b_fp16.safetensors", "ema_vae_fp16.safetensors"]
        if f == "sensenova":
            return ["README.md", "config.json", "model.safetensors.index.json", "model-*.safetensors", "tokenizer.json"]
        if f == "idol":
            return ["config.json", "denoiser.safetensors", "decoder.safetensors", "decoder_gs.safetensors",
                    "birefnet.safetensors", "dinov3/config.json", "dinov3/model.safetensors", "vae/config.json",
                    "vae/model.safetensors", "LICENSE", "NOTICE", "README.md", "dinov3/LICENSE.md"]
        if f == "minimax-music3":
            return ["LICENSE", "README.md", "config.json", "model.safetensors.index.json", "model-*.safetensors",
                    "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]
        if f == "moss-soundeffect":
            return weights + ["README.md", "model_index.json", "mlx/*.safetensors", "transformer/config.json",
                              "vae/config.json", "scheduler/*.json", "tokenizer/*"]
        raise ValueError(f"no download patterns for {f}")

    @property
    def quality_files(self) -> list[str]:
        """LTX's dev transformer, only needed for quality=True."""
        return ["transformer-dev.safetensors"] if self.is_ltx else []

    @property
    def companion(self) -> Companion | None:
        if self.family == "ltx-2":
            return Companion(
                repo="mlx-community/gemma-3-12b-it-4bit",
                revision="86cc6a8dedbc456dd0e4af01a9d09f396f77e558",
                patterns=("*.json", "*.safetensors", "tokenizer.model"),
            )
        return None


def _m(**kw) -> Model:
    return Model(**kw)


CATALOG: list[Model] = [
    _m(id="ming-image", name="Ming-Image 0.1 Design", family="ming",
       detail="Graphic design and typography with a transparent background: posters, cards, UI, logos.",
       repo="joeynyc/Ming-Image-0.1-Design-mflux-q8-te5", revision="3adb8aaef779f9b3fa4621acebeba97bb4010d42",
       size_gb=19.2, memory_gb=24, license="MIT"),
    _m(id="z-image-turbo-q4", name="Z-Image Turbo (4-bit)", family="z-image-turbo",
       detail="Photorealistic images with legible text, in 9 steps. The lighter 4-bit version.",
       repo="mflux-community/z-image-turbo-mflux-q4", revision="f427e257d8e6ffa03edd4d9ac554a05809da456c",
       size_gb=5.9, memory_gb=16, license="Apache 2.0"),
    _m(id="z-image-turbo-q8", name="Z-Image Turbo (8-bit)", family="z-image-turbo",
       detail="Photorealistic images with legible text, in 9 steps. The 8-bit version, closer to the original.",
       repo="mflux-community/z-image-turbo-mflux-q8", revision="4ccff28917346aa9daae49cd3c477cf5260a242c",
       size_gb=11.0, memory_gb=24, license="Apache 2.0"),
    _m(id="flux2-klein-4b", name="FLUX.2 Klein 4B", family="flux2-klein", variant="flux2-klein-4b",
       detail="Black Forest Labs' distilled model: an image in 4 steps, the fastest here. Edits reference images.",
       repo="mflux-community/flux2-klein-4b-mflux-q4", revision="794cd159538149ad9830848508c31f0ea7088e58",
       size_gb=4.6, memory_gb=24, license="Apache 2.0"),
    _m(id="flux3-klein", name="FLUX 3 Image Klein", family="flux3-klein", variant="flux3-image-klein", quantize=8,
       detail="Black Forest Labs' newest model: photos and legible text in 16 steps; edits a reference image. "
              "Gated repo: accept the licence on Hugging Face and set HF_TOKEN.",
       repo="black-forest-labs/flux-3-image-klein", revision="337337621b8526e120f2275d250242f4a6bedd75",
       size_gb=20.3, memory_gb=16, license="FLUX Kommunity"),
    _m(id="moss-image", name="MOSS-Image", family="moss-image", quantize=8,
       detail="OpenMOSS's 5B model with Qwen3.5: long text in English and Chinese, layouts from JSON prompts. Slow.",
       repo="mlx-community/MOSS-Image-SFT-bf16", revision="ea698e99312015505884559ebad94b4a5a0d0753",
       size_gb=27.2, memory_gb=16, license="Apache 2.0"),
    _m(id="sensenova-u1.5", name="SenseNova-U1.5", family="sensenova",
       detail="SenseTime's unified model: photos, posters and infographics with legible text, in 8 steps; edits a reference image.",
       repo="mlx-community/SenseNova-U1.5-8B-MoT-8step-4bit", revision="ff6d0c2dfe21b19891ae4551e11fcc99f6aa82ae",
       size_gb=11.8, memory_gb=16, license="Apache 2.0"),
    _m(id="ltx-2.3-q4", name="LTX-2.3 (4-bit)", family="ltx-2",
       detail="Lightricks' video model with sound: a clip from a prompt, or from images as first/last frames.",
       repo="dgrauet/ltx-2.3-mlx-q4", revision="56a5866d638ecfe37c54d348e88938235185c2d4",
       size_gb=28.5, memory_gb=32, license="LTX-2 Community"),
    _m(id="ltx-2.3-q8", name="LTX-2.3 (8-bit)", family="ltx-2",
       detail="Lightricks' video model with sound. The 8-bit version, closer to the original.",
       repo="dgrauet/ltx-2.3-mlx-q8", revision="6671a7572a530862d1d60ce393b5d93491e3f76b",
       size_gb=37.8, memory_gb=64, license="LTX-2 Community"),
    _m(id="ltx-2.5-q4", name="LTX-2.5 (4-bit)", family="ltx-2.5",
       detail="Lightricks' newer video model with sound; follows multi-shot prompts and can pick the clip length. "
              "Gated repo: accept the licence on Hugging Face and set HF_TOKEN.",
       repo="dgrauet/ltx-2.5-mlx-q4", revision="e9a20add8c5937fcab635fb8e7a52926487a5776",
       size_gb=28.8, memory_gb=32, license="LTX-2 Community"),
    _m(id="qwen-image-2512", name="Qwen-Image 2512", family="qwen-image",
       detail="Alibaba's 20B model: rich scenes and long text inside the image. Thorough but slow: 20 steps.",
       repo="mflux-community/qwen-image-2512-mflux-q4", revision="f30c76ff30977000ffdc27fafdd4516bb1c3b44e",
       size_gb=27.6, memory_gb=32, license="Apache 2.0"),
    _m(id="qwen-image-edit-2511", name="Qwen-Image Edit 2511", family="qwen-image-edit",
       detail="Alibaba's 20B editor: changes the reference picture as the prompt says and keeps the rest.",
       repo="mflux-community/qwen-image-edit-2511-mflux-q4", revision="3c74aab7dd79e6a8148c329cbca78c00c049ec3b",
       size_gb=29.0, memory_gb=32, license="Apache 2.0"),
    _m(id="seedvr2-3b", name="SeedVR2 Upscaler 3B", family="seedvr2",
       detail="ByteDance's upscaler: enlarges a picture 2-4x with sharper, faithful detail, in one step. No prompt.",
       repo="numz/SeedVR2_comfyUI", revision="09ced71023636e9bc8cdf9cdecfb2625d1e691e8",
       size_gb=7.3, memory_gb=16, license="Apache 2.0"),
    _m(id="idol", name="Idol", family="idol",
       detail="Sperid Labs' Idol: one picture of one object becomes a 3D object (Gaussian splats) as a GLB. No prompt.",
       repo="ddalcu/Idol-MLX-Serve-8bit", revision="12fbe4b06ef7cc377100b8f60238a6824546cacc",
       size_gb=3.0, memory_gb=16, license="Apache 2.0 / DINOv3 License"),
    _m(id="minimax-music3-4bit", name="MiniMax-Music3 (4-bit)", family="minimax-music3",
       detail="MiniMax's music model: a description and lyrics become a complete song with vocals, up to 6 minutes.",
       repo="mlx-community/MiniMax-Music3-4bit", revision="c7ea32923b245fe5afc22d740a1936ad2ac590f3",
       size_gb=9.2, memory_gb=16, license="MiniMax-Music3 Community License"),
    _m(id="minimax-music3-8bit", name="MiniMax-Music3 (8-bit)", family="minimax-music3",
       detail="MiniMax's music model, 8-bit version, closer to the original.",
       repo="mlx-community/MiniMax-Music3-8bit", revision="10aa4ca578d04c6f5256c1bc22fc8405a09602b5",
       size_gb=14.2, memory_gb=24, license="MiniMax-Music3 Community License"),
    _m(id="moss-soundeffect", name="MOSS-SoundEffect", family="moss-soundeffect",
       detail="OpenMOSS's sound-effect model: a description becomes up to 30 s of sound. No music or speech.",
       repo="mlx-community/MOSS-SoundEffect-v2.0-bf16", revision="71a08a420dcac61b73fc1f721d56ac41078973bf",
       size_gb=8.4, memory_gb=16, license="Apache 2.0"),
]

BY_ID: dict[str, Model] = {m.id: m for m in CATALOG}


def find(model_id: str) -> Model:
    if model_id in BY_ID:
        return BY_ID[model_id]
    for m in CATALOG:  # also accept the Hugging Face repo id
        if m.repo == model_id:
            return m
    raise KeyError(f"unknown model {model_id!r}; call list_models for the ids")
