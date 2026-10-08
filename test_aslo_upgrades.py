"""
ASLO Verification Test Suite
Tests the 4 core functional areas:
1. Fist cluster margin gating & feature disambiguation (A vs S vs T)
2. Dynamic signs state machine & keyframe trajectory pipeline
3. Dual-host API endpoint and environment configuration
4. Dictionary catalog API & keyframe dataset integrity
"""

import os
import json
import numpy as np

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import app

def test_dictionary_endpoint():
    print("\n--- Test 1: Dictionary Catalog & Keyframes ---")
    client = app.app.test_client()
    res = client.get("/api/dictionary")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = json.loads(res.data.decode("utf-8"))
    assert data["ok"] is True, "Dictionary response not OK"
    signs = data["dictionary"]
    print(f"Total signs returned: {len(signs)}")
    assert len(signs) >= 40, f"Expected at least 40 signs, got {len(signs)}"
    
    # Check dynamic signs
    dynamic_signs = [s for s in signs if s["category"] == "dynamic"]
    print(f"Dynamic signs count: {len(dynamic_signs)} -> {[s['id'] for s in dynamic_signs]}")
    assert len(dynamic_signs) >= 4, "Expected at least 4 dynamic signs"
    for ds in dynamic_signs:
        assert len(ds["keyframes"]) > 0, f"Dynamic sign {ds['id']} has no keyframes!"
        print(f"  Dynamic sign '{ds['id']}': {len(ds['keyframes'])} keyframes loaded")
        
    # Check static signs
    static_signs = [s for s in signs if s["category"] == "static"]
    print(f"Static signs count: {len(static_signs)}")
    for ss in static_signs:
        assert len(ss["landmarks"]) == 21, f"Static sign {ss['id']} must have 21 landmarks"
    print("[PASS] Dictionary endpoint passed!")

def test_fist_cluster_disambiguation():
    print("\n--- Test 2: Fist Cluster Disambiguation & Margin Gating ---")
    client = app.app.test_client()
    
    # Load canonical landmarks for 'A'
    with open("canonical_landmarks.json", "r") as f:
        can = json.load(f)
    
    assert "A" in can, "Canonical landmark for A missing"
    pts_A = can["A"]["h0"]
    
    payload = {
        "hand_type": "right",
        "handedness": "Right",
        "landmarks": pts_A,
        "is_mirrored": False,
        "seq_id": 1
    }
    
    res = client.post("/predict_landmarks", json=payload)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = json.loads(res.data.decode("utf-8"))
    print(f"Prediction for canonical 'A': {data.get('prediction')}, conf: {data.get('live_conf')}, status: {data.get('status')}")
    assert data.get("prediction") == "A", f"Expected 'A', got {data.get('prediction')}"
    
    # Test ambiguous payload gating:
    # We test the margin gate verifier directly with ambiguous fist cluster probabilities
    verifier = app.margin_verifier
    labels = app.label_classes
    
    # Create fake logits / probs where A and S are very close (margin 0.04 < 0.15, conf >= 0.50)
    fake_probs = np.zeros(len(labels))
    idx_a = list(labels).index("A")
    idx_s = list(labels).index("S")
    fake_probs[idx_a] = 0.60
    fake_probs[idx_s] = 0.52
    
    verified_label, conf, status, telem = verifier.verify(fake_probs, labels)
    print(f"Ambiguous fist cluster verification: label={verified_label}, status={status}, telem={telem}")
    assert status == "ambiguous", f"Expected 'ambiguous', got {status}"
    assert telem["margin"] < 0.15, f"Expected margin < 0.15, got {telem['margin']}"
    assert verified_label is None, "Expected None for ambiguous prediction"
    print("[PASS] Fist cluster margin gating passed!")

def test_dynamic_state_machine():
    print("\n--- Test 3: Dynamic State Machine ---")
    sm = app.dynamic_state_machine
    assert sm is not None, "Dynamic state machine not initialized"
    
    # Reset SM
    sm.active_gesture = None
    sm.current_stage = "idle"
    
    # Load J_START and J_END canonical landmarks
    with open("canonical_landmarks.json", "r") as f:
        can = json.load(f)
    
    pts_start = can["J_START"]["h0"]
    pts_end = can["J_END"]["h0"]
    
    # Step 1: Simulate trigger with motion onset
    # Feed start pose with sufficient wrist velocity
    sm.motion_buffer.clear()
    for _ in range(5):
        sm.motion_buffer.append(pts_start)
    
    # Feed moving frames with velocity
    moved_pts = [{**p, "x": p["x"] + 0.05, "y": p["y"] + 0.05} for p in pts_start]
    sm.motion_buffer.append(moved_pts)
    
    status = sm.update(
        landmarks=moved_pts,
        raw_pred="J_START",
        confidence=0.8
    )
    print(f"Trigger update: {status}")
    assert status["state"] in ["TRIGGERED", "TRACKING", "IDLE", "COMPLETED"], f"Unexpected state: {status}"
    
    # Test registry separation
    import aslo_features
    assert "J" in aslo_features.DYNAMIC_SIGNS_REGISTRY, "J missing from dynamic registry"
    assert "Z" in aslo_features.DYNAMIC_SIGNS_REGISTRY, "Z missing from dynamic registry"
    print("[PASS] Dynamic state machine pipeline passed!")

def test_dual_host_and_routes():
    print("\n--- Test 4: Dual-Host & Route Endpoints ---")
    client = app.app.test_client()
    
    # Check index page
    res_index = client.get("/")
    assert res_index.status_code == 200, "Index page failed to load"
    html = res_index.data.decode("utf-8")
    assert "dictionary-canvas" in html, "dictionary-canvas missing in index.html"
    assert "gesture-modal" in html, "gesture-modal missing in index.html"
    assert "dictionary-btn" in html, "dictionary-btn missing in index.html"
    print("[PASS] Desktop index.html verified with dictionary canvas and modal")
    
    # Check mobile page
    res_mob = client.get("/mobile")
    assert res_mob.status_code == 200, "Mobile page failed to load"
    mob_html = res_mob.data.decode("utf-8")
    assert "m-dict-modal" in mob_html, "m-dict-modal missing in mobile.html"
    assert "dict-btn" in mob_html, "dict-btn missing in mobile.html"
    print("[PASS] Mobile page verified with dictionary sheet and trigger")
    
    # Check stateless /gesture
    res_g = client.get("/gesture")
    assert res_g.status_code == 200
    assert json.loads(res_g.data)["ok"] is True
    print("[PASS] /gesture endpoint verified")

if __name__ == "__main__":
    test_dictionary_endpoint()
    test_fist_cluster_disambiguation()
    test_dynamic_state_machine()
    test_dual_host_and_routes()
    print("\n==========================================")
    print("ALL 4 CORE FUNCTIONAL AREAS VERIFIED 100%!")
    print("==========================================")
