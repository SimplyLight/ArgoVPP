import numpy as np

# Test direct avec la nouvelle formule
V_TW = 16  # m/s
V_B = 3    # m/s
TWA_deg = 50  # degrés

TWA_rad = np.radians(TWA_deg)

# Vecteur vent réel (axe Y)
V_TW_x = 0.0
V_TW_y = V_TW

# Vecteur bateau avec cap TWA
V_B_x = V_B * np.cos(TWA_rad)
V_B_y = V_B * np.sin(TWA_rad)

# Vent apparent = vent réel + bateau
V_AW_x = V_TW_x + V_B_x
V_AW_y = V_TW_y + V_B_y
V_AW = np.sqrt(V_AW_x**2 + V_AW_y**2)

# Angle décalage
beta_AW_rad = np.arctan(V_AW_x / V_AW_y) if abs(V_AW_y) > 1e-9 else 0.0
beta_AW_deg = np.degrees(beta_AW_rad)

print("╔════════════════════════════════════════╗")
print("║  TEST NOUVELLE FORMULE VECTORIELLE  ║")
print("╚════════════════════════════════════════╝")
print(f"\n📋 Paramètres:")
print(f"   V_vent_réel = {V_TW} m/s")
print(f"   V_bateau = {V_B} m/s")
print(f"   Cap (TWA) = {TWA_deg}°")
print(f"\n📐 Calcul vectoriel:")
print(f"   V_aw = ({V_AW_x:.4f}, {V_AW_y:.4f}) m/s")
print(f"\n✅ Résultats:")
print(f"   |V_aw| = {V_AW:.2f} m/s")
print(f"   Décalage = {beta_AW_deg:.2f}°")
print(f"\n📊 Vérification avec valeurs attendues:")
expected_v = 18.41
expected_angle = 6.03
v_ok = "✓" if abs(V_AW - expected_v) < 0.1 else "✗"
angle_ok = "✓" if abs(beta_AW_deg - expected_angle) < 0.1 else "✗"
print(f"   Expected |V_aw| ≈ {expected_v} m/s → {V_AW:.2f} m/s {v_ok}")
print(f"   Expected décalage ≈ {expected_angle}° → {beta_AW_deg:.2f}° {angle_ok}")
