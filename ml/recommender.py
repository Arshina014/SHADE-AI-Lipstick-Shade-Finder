import pandas as pd
import numpy as np
import cv2

def get_palette(hex_color):
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)

    if r > 200 and g > 160 and b > 140:
        return "Nude"
    elif r > g and g > b:
        return "Brown"
    elif r > 180 and b > 150:
        return "Pink"
    elif r > 150 and g < 100:
        return "Red"
    elif b > r:
        return "Mauve"
    else:
        return "Coral"
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return np.array([b, g, r])

def recommend_top3(cheek_path, undertone, selected_brand=None):
    data = pd.read_csv("data/lipstick_data.csv")

    # Filter by undertone
    data = data[data["undertone"] == undertone]

    # Filter by brand
    if selected_brand and selected_brand != "All":
        data = data[data["brand"] == selected_brand]

    cheek = cv2.imread(cheek_path)
    avg_color = cheek.mean(axis=0).mean(axis=0)

    recommendations = []

    for _, row in data.iterrows():
        lipstick_color = hex_to_bgr(row["hex_color"])
        distance = np.linalg.norm(avg_color - lipstick_color)

        recommendations.append({
            "brand": row["brand"],
            "shade": row["shade_name"],
            "distance": distance
        })

    # Sort by distance
    recommendations = sorted(recommendations, key=lambda x: x["distance"])

    # Normalize distance → confidence
    max_dist = recommendations[-1]["distance"] if recommendations else 1

    final_results = []
    for r in recommendations[:3]:
        confidence = round(100 * (1 - (r["distance"] / max_dist)), 1)
        final_results.append({
    "brand": r["brand"],
    "shade": r["shade"],
    "confidence": confidence,
    "hex": row["hex_color"],
    "palette": get_palette(row["hex_color"])
})
        

    return final_results