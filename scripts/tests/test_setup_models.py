from scripts import setup_models


def test_guide_lists_download_only_models_with_sources_and_keys():
    guide = setup_models.build_guide()
    # Silero VAD (入手元 + config キー)
    assert "snakers4/silero-vad" in guide
    assert "vad_model_file" in guide
    # rmvpe (入手元 + GPL + NOTICES 導線 + config キー)
    assert "wok000/weights_gpl" in guide
    assert "GPL-3.0" in guide
    assert "THIRD_PARTY_NOTICES" in guide
    assert "rmvpe_model_file" in guide


def test_guide_points_derived_models_to_their_own_poe_tasks():
    guide = setup_models.build_guide()
    # FCPE / HuBERT は各自の poe タスク --help が詳細の持ち主
    assert "poe export-fcpe-onnx" in guide
    assert "fcpe_model_file" in guide
    assert "poe convert-hubert" in guide
    assert "poe export-hubert-onnx" in guide
    assert "hubert_model_file" in guide
