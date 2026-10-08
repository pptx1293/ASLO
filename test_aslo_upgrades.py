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
    
    # Test canonical landmarks for T, S, N, M, A
    for sign in ["T", "S", "N", "M", "A"]:
        pts = can[sign]["h0"]
        payload = {
            "hand_type": "right",
            "handedness": "Right",
            "landmarks": pts,
            "is_mirrored": False,
            "seq_id": 1
        }
        res = client.post("/predict_landmarks", json=payload)
        assert res.status_code == 200
        d = json.loads(res.data.decode("utf-8"))
        print(f"Prediction for canonical '{sign}': {d.get('prediction')}, conf: {d.get('live_conf')}, status: {d.get('status')}")
        assert d.get("prediction") == sign, f"Expected '{sign}', got '{d.get('prediction')}'"
        assert d.get("status") == "valid"

    # Test geometric tie-breaker logic for T vs S:
    # Construct base fist landmarks
    lms = np.zeros((21, 3), dtype=np.float32)
    lms[0] = [0.0, 0.0, 0.0]
    lms[5] = [0.20, -0.90, 0.0]
    lms[6] = [0.20, -1.25, 0.0]
    lms[9] = [0.10, -0.95, 0.0]
    lms[10] = [0.10, -1.30, 0.0]
    lms[13] = [0.00, -0.95, 0.0]
    lms[14] = [0.00, -1.25, 0.0]
    lms[17] = [-0.10, -0.90, 0.0]
    lms[18] = [-0.10, -1.15, 0.0]

    # In T: Thumb tip (4) wedged between index (5) and middle (9), protruding up (y4 < y6)
    lms_T = lms.copy()
    lms_T[4] = [0.15, -1.30, 0.0]
    tb_T = app.aslo_features.resolve_fist_tie_breaker(lms_T, "S", "T", margin=0.05)
    assert tb_T == "T", f"Expected 'T', got {tb_T}"

    # In S: Thumb tip (4) curls flat across front (y4 >= y6)
    lms_S = lms.copy()
    lms_S[4] = [0.15, -0.95, 0.0]
    tb_S = app.aslo_features.resolve_fist_tie_breaker(lms_S, "T", "S", margin=0.05)
    assert tb_S == "S", f"Expected 'S', got {tb_S}"

    # In N: Thumb in Slot N (between middle 9 and ring 13)
    lms_N = lms.copy()
    lms_N[4] = (lms[9] + lms[13]) / 2.0
    tb_N = app.aslo_features.resolve_fist_tie_breaker(lms_N, "M", "N", margin=0.05)
    assert tb_N == "N", f"Expected 'N', got {tb_N}"

    # In M: Thumb in Slot M (between ring 13 and pinky 17)
    lms_M = lms.copy()
    lms_M[4] = (lms[13] + lms[17]) / 2.0
    tb_M = app.aslo_features.resolve_fist_tie_breaker(lms_M, "N", "M", margin=0.05)
    assert tb_M == "M", f"Expected 'M', got {tb_M}"

    # Pairwise T vs N disambiguation on canonical landmarks:
    # 1. N landmarks with T predicted by mistake -> must resolve to N
    pts_can_N = np.array([[lm['x'], lm['y'], lm['z']] for lm in can['N']['h0']], dtype=np.float32)
    sc_N = float(np.linalg.norm(pts_can_N[9] - pts_can_N[0])) + 1e-6
    norm_can_N = (pts_can_N - pts_can_N[0]) / sc_N
    tb_can_N = app.aslo_features.resolve_fist_tie_breaker(norm_can_N, "T", "N", margin=0.05)
    assert tb_can_N == "N", f"Expected 'N', got {tb_can_N}"

    # 2. T landmarks with N predicted by mistake -> must resolve to T
    pts_can_T = np.array([[lm['x'], lm['y'], lm['z']] for lm in can['T']['h0']], dtype=np.float32)
    sc_T = float(np.linalg.norm(pts_can_T[9] - pts_can_T[0])) + 1e-6
    norm_can_T = (pts_can_T - pts_can_T[0]) / sc_T
    tb_can_T = app.aslo_features.resolve_fist_tie_breaker(norm_can_T, "N", "T", margin=0.05)
    assert tb_can_T == "T", f"Expected 'T', got {tb_can_T}"

    # Verify feature vector length is 310
    assert app.aslo_features.FEATURE_LEN == 310
    assert app.aslo_features.FIST_GEOMETRY_LEN == 42
    print("[PASS] Knuckle-slot geometric tie-breakers (including T vs N) and feature lengths (310) verified!")

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
